
#define _GNU_SOURCE

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAXCPU 512
#define MAXNODE 16
#define BATCH 4096

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }

static long move_pages_q(void **pages, unsigned long count, int *status) {
    return syscall(__NR_move_pages, 0, count, pages, NULL, status, 0);
}


static int cpu_node[MAXCPU];
static int derived_from_sysfs = 0;

static int parse_cpulist(const char *s, int node) {
    int n = 0;
    while (*s) {
        char *end;
        long a = strtol(s, &end, 10);
        if (end == s) break;
        long b = a;
        if (*end == '-') { s = end + 1; b = strtol(s, &end, 10); }
        for (long c = a; c <= b && c < MAXCPU; c++) { cpu_node[c] = node; n++; }
        s = (*end == ',') ? end + 1 : end;
        if (*s == '\0') break;
    }
    return n;
}

static void build_cpu_node_map(void) {
    for (int i = 0; i < MAXCPU; i++) cpu_node[i] = -1;
    int total = 0;
    for (int node = 0; node < MAXNODE; node++) {
        char p[256], buf[4096];
        snprintf(p, sizeof p, "/sys/devices/system/node/node%d/cpulist", node);
        int fd = open(p, O_RDONLY);
        if (fd < 0) continue;
        ssize_t n = read(fd, buf, sizeof buf - 1);
        close(fd);
        if (n <= 0) continue;
        buf[n] = '\0';
        total += parse_cpulist(buf, node);
    }
    if (total > 0) { derived_from_sysfs = 1; return; }
    /* Fallback, flagged loudly in the output. */
    for (int i = 0; i < MAXCPU; i++) cpu_node[i] = i & 1;
}


static void *map_ro(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    *len = (size_t)st.st_size;
    void *p = mmap(NULL, *len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return (p == MAP_FAILED) ? NULL : p;
}

static const char *REC_ARRAYS[] = { "itemid.i32", "t.f32", "value.f32", NULL };

int main(int argc, char **argv)
{
    const char *csr = NULL, *cpulist = NULL;
    int T = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csr") && i + 1 < argc) csr = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpus") && i + 1 < argc) cpulist = argv[++i];
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!csr || !T || !cpulist) die("--csr DIR --threads T --cpus c0,c1,... required");

    build_cpu_node_map();

    int cpus[MAXCPU], ncpu = 0;
    { char *dup = strdup(cpulist), *tok = strtok(dup, ",");
      while (tok && ncpu < MAXCPU) { cpus[ncpu++] = atoi(tok); tok = strtok(NULL, ","); }
      free(dup); }
    if (ncpu != T) die("--cpus must list exactly --threads entries, in thread order");

    for (int j = 0; j < T; j++)
        if (cpus[j] < 0 || cpus[j] >= MAXCPU || cpu_node[cpus[j]] < 0) {
            fprintf(stderr, "FATAL: cpu %d has no node in sysfs -- the --cpus "
                    "list does not match this machine\n", cpus[j]);
            exit(1);
        }

    const long PS = sysconf(_SC_PAGESIZE);

    char p[4096];
    snprintf(p, sizeof p, "%s/offsets.i64", csr);
    size_t olen = 0;
    const int64_t *offsets = map_ro(p, &olen);
    if (!offsets) die("open offsets.i64");
    const int64_t n_stays = (int64_t)(olen / sizeof(int64_t)) - 1;
    const int64_t n_records = offsets[n_stays];

    /* Same cut as stage_aligned.c and the kernel's `block` partitioner. */
    int64_t *cut = malloc((size_t)(T + 1) * sizeof *cut);
    if (!cut) die("malloc");
    for (int j = 0; j <= T; j++)
        cut[j] = (int64_t)((double)n_stays * j / T + 0.5);
    cut[T] = n_stays;

    printf("numa_verify: %s\n", csr);
    printf("  n_stays=%" PRId64 "  n_records=%" PRId64 "  T=%d  page=%ldB\n",
           n_stays, n_records, T, PS);
    printf("  cpu->node map %s\n", derived_from_sysfs
           ? "derived from /sys/devices/system/node/*/cpulist"
           : "*** SYSFS UNREADABLE, assuming cpu&1 -- DO NOT TRUST ***");

    for (int a = 0; REC_ARRAYS[a]; a++) {
        snprintf(p, sizeof p, "%s/%s", csr, REC_ARRAYS[a]);
        size_t slen = 0;
        const char *base = map_ro(p, &slen);
        if (!base) { fprintf(stderr, "  %-12s MISSING\n", REC_ARRAYS[a]); continue; }
        const size_t esz = slen / (size_t)n_records;

        printf("\n  %s  (%.1f MiB, elem=%zuB)\n", REC_ARRAYS[a],
               (double)slen / 1048576.0, esz);
        printf("    tid  cpu node       pages   local   local_frac\n");

        void **pv = malloc(BATCH * sizeof *pv);
        int *st = malloc(BATCH * sizeof *st);
        if (!pv || !st) die("malloc batch");

        int64_t tot_pages = 0, tot_local = 0, tot_unres = 0;

        for (int j = 0; j < T; j++) {
            const size_t b0 = (size_t)offsets[cut[j]]     * esz;
            const size_t b1 = (size_t)offsets[cut[j + 1]] * esz;
            const int want = cpu_node[cpus[j]];

            /* Interior pages only: a page straddling a cut belongs to no
             * single thread and is counted separately, not charged to one. */
            size_t pa = ((b0 + (size_t)PS - 1) / (size_t)PS) * (size_t)PS;
            size_t pb = (b1 / (size_t)PS) * (size_t)PS;

            int64_t pages = 0, local = 0, unres = 0;
            size_t off = pa;
            while (off < pb) {
                unsigned long n = 0;
                for (; n < BATCH && off < pb; n++, off += (size_t)PS) {
                    /* Read-only fault: maps the existing page-cache page. */
                    volatile char sink = base[off];
                    (void)sink;
                    pv[n] = (void *)(base + off);
                }
                if (move_pages_q(pv, n, st) != 0) die("move_pages query");
                for (unsigned long k = 0; k < n; k++) {
                    if (st[k] < 0) { unres++; continue; }
                    pages++;
                    if (st[k] == want) local++;
                }
            }
            printf("    %3d  %3d   %2d  %10" PRId64 "  %10" PRId64 "   %6.4f%s\n",
                   j, cpus[j], want, pages, local,
                   pages ? (double)local / (double)pages : 0.0,
                   unres ? "  (unresolved pages present)" : "");
            tot_pages += pages; tot_local += local; tot_unres += unres;
        }

        printf("    %-13s %10" PRId64 "  %10" PRId64 "   %6.4f  <== page_local_frac\n",
               "TOTAL", tot_pages, tot_local,
               tot_pages ? (double)tot_local / (double)tot_pages : 0.0);
        if (tot_unres)
            printf("    WARNING: %" PRId64 " pages unresolved by move_pages\n", tot_unres);


        int crossing = 0;
        for (int j = 1; j < T; j++)
            if (cpu_node[cpus[j - 1]] != cpu_node[cpus[j]]) crossing++;
        printf("    cuts=%d  node-crossing cuts=%d  (straddling-page exposure)\n",
               T - 1, crossing);

        free(pv); free(st);
        munmap((void *)base, slen);
    }

    free(cut);
    munmap((void *)offsets, olen);
    return 0;
}
