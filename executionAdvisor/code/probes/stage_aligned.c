

#define _GNU_SOURCE

#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}


static const char *REC_ARRAYS[] = { "itemid.i32", "t.f32", "value.f32", NULL };
static int is_record_array(const char *n) {
    for (int i = 0; REC_ARRAYS[i]; i++) if (!strcmp(n, REC_ARRAYS[i])) return 1;
    return 0;
}

static void *map_ro(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    *len = (size_t)st.st_size;
    void *p = mmap(NULL, *len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return (p == MAP_FAILED) ? NULL : p;
}

static void *map_rw_new(const char *path, size_t len) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NULL;
    if (ftruncate(fd, (off_t)len) != 0) { close(fd); return NULL; }
    /* Deliberately NOT MAP_POPULATE: prefaulting here would place every page
     * from this thread and defeat the entire purpose. */
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return (p == MAP_FAILED) ? NULL : p;
}

static void plain_copy(const char *src, const char *dst) {
    size_t n = 0;
    void *s = map_ro(src, &n);
    if (!s) die("map src");
    void *d = map_rw_new(dst, n);
    if (!d) die("map dst");
    memcpy(d, s, n);
    if (msync(d, n, MS_SYNC) != 0) die("msync");
    munmap(d, n); munmap(s, n);
}

int main(int argc, char **argv)
{
    const char *src = NULL, *dst = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--src") && i + 1 < argc) src = argv[++i];
        else if (!strcmp(argv[i], "--dst") && i + 1 < argc) dst = argv[++i];
        else if (!strcmp(argv[i], "--partitioner") && i + 1 < argc) {
            if (strcmp(argv[++i], "block"))
                die("only --partitioner block is supported; the alignment must "
                    "match the kernel's partition and block is what the "
                    "reported configurations use");
        } else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!src || !dst) die("--src DIR --dst DIR required");
    if (mkdir(dst, 0755) != 0 && access(dst, W_OK) != 0) die("cannot create --dst");

    char p[4096];
    snprintf(p, sizeof p, "%s/offsets.i64", src);
    size_t olen = 0;
    const int64_t *offsets = map_ro(p, &olen);
    if (!offsets) die("open offsets.i64");
    const int64_t n_stays = (int64_t)(olen / sizeof(int64_t)) - 1;
    const int64_t n_records = offsets[n_stays];

    int T = omp_get_max_threads();
    if (T < 1) T = 1;

    int64_t *cut = malloc((size_t)(T + 1) * sizeof *cut);
    if (!cut) die("malloc cut");
    for (int j = 0; j <= T; j++)
        cut[j] = (int64_t)((double)n_stays * j / T + 0.5);
    cut[T] = n_stays;

    printf("stage_aligned: n_stays=%" PRId64 "  n_records=%" PRId64 "  T=%d\n",
           n_stays, n_records, T);

    /* Report the mapping so a misbinding is visible rather than silent. */
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp critical
        printf("  thread %2d  cpu %3d  stays [%8" PRId64 ",%8" PRId64 ")"
               "  records [%9" PRId64 ",%9" PRId64 ")\n",
               tid, sched_getcpu(), cut[tid], cut[tid + 1],
               offsets[cut[tid]], offsets[cut[tid + 1]]);
    }

    /* offsets first, plainly */
    snprintf(p, sizeof p, "%s/offsets.i64", dst);
    { char s2[4096]; snprintf(s2, sizeof s2, "%s/offsets.i64", src);
      plain_copy(s2, p); }
    printf("  offsets.i64  plain copy (stay-indexed, 0.5 MB)\n");

    DIR *d = opendir(src);
    if (!d) die("opendir src");
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        if (!strcmp(e->d_name, "offsets.i64")) continue;
        char sp[4096], dp[4096];
        snprintf(sp, sizeof sp, "%s/%s", src, e->d_name);
        snprintf(dp, sizeof dp, "%s/%s", dst, e->d_name);
        struct stat st;
        if (stat(sp, &st) != 0 || !S_ISREG(st.st_mode)) continue;

        if (!is_record_array(e->d_name)) {
            plain_copy(sp, dp);
            printf("  %-12s plain copy (%.1f MiB)\n", e->d_name,
                   (double)st.st_size / 1048576.0);
            continue;
        }

        size_t slen = 0;
        const char *s = map_ro(sp, &slen);
        if (!s) die("map record array");
        char *dd = map_rw_new(dp, slen);
        if (!dd) die("map dst record array");
        const size_t esz = slen / (size_t)n_records;
        if (esz * (size_t)n_records != slen)
            die("record array size is not a multiple of n_records");

        const double t0 = now_s();
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const size_t b0 = (size_t)offsets[cut[tid]]     * esz;
            const size_t b1 = (size_t)offsets[cut[tid + 1]] * esz;
       
            if (b1 > b0) memcpy(dd + b0, s + b0, b1 - b0);
        }
        if (msync(dd, slen, MS_SYNC) != 0) die("msync");
        const double dt = now_s() - t0;
        printf("  %-12s aligned  %.1f MiB  %.2f s  %.2f GB/s  elem=%zuB\n",
               e->d_name, (double)slen / 1048576.0, dt,
               (double)slen / dt / 1e9, esz);
        munmap(dd, slen); munmap((void *)s, slen);
    }
    closedir(d);
    free(cut);
    munmap((void *)offsets, olen);
    printf("stage_aligned: done. Verify with numa_pages, and run the kernel\n"
           "with the SAME --threads and OMP_PLACES or the alignment is void.\n");
    return 0;
}
