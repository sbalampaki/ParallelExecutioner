
#define _GNU_SOURCE
#define THP_ALIGN (2UL << 20)   /* huge-page size: what the grid must clear */
#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static size_t BLOCK = 2UL << 20;  

static const char *FILES[] = {
    "offsets.i64", "itemid.i32", "t.f32", "value.f32"
};

static int stage_one(const char *src, const char *dst, int interleave)
{
    int sfd = open(src, O_RDONLY);
    if (sfd < 0) { fprintf(stderr, "open %s: %s\n", src, strerror(errno)); return -1; }

    struct stat st;
    if (fstat(sfd, &st) < 0) { perror("fstat"); close(sfd); return -1; }
    size_t len = (size_t)st.st_size;

    const char *sp = mmap(NULL, len, PROT_READ, MAP_SHARED, sfd, 0);
    if (sp == MAP_FAILED) { perror("mmap src"); close(sfd); return -1; }

    unlink(dst);
    int dfd = open(dst, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (dfd < 0) { fprintf(stderr, "open %s: %s\n", dst, strerror(errno)); return -1; }
    if (ftruncate(dfd, (off_t)len) < 0) { perror("ftruncate"); close(dfd); return -1; }

    char *dp = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, dfd, 0);
    if (dp == MAP_FAILED) { perror("mmap dst"); close(dfd); return -1; }

    double t0 = omp_get_wtime();

    if (interleave) {
  
        size_t head = (THP_ALIGN - ((uintptr_t)dp % THP_ALIGN)) % THP_ALIGN;
        if (head > len) head = len;
        if (head) memcpy(dp, sp, head);

        size_t body    = len - head;
        size_t nblocks = (body + BLOCK - 1) / BLOCK;

 
        #pragma omp parallel for schedule(static, 1)
        for (size_t b = 0; b < nblocks; b++) {
            size_t off = head + b * BLOCK;
            size_t n   = (len - off < BLOCK) ? (len - off) : BLOCK;
            memcpy(dp + off, sp + off, n);
        }
    } else {
        size_t nblocks = (len + BLOCK - 1) / BLOCK;
        for (size_t b = 0; b < nblocks; b++) {
            size_t off = b * BLOCK;
            size_t n   = (len - off < BLOCK) ? (len - off) : BLOCK;
            memcpy(dp + off, sp + off, n);
        }
    }

    if (msync(dp, len, MS_SYNC) < 0) perror("msync");
    double dt = omp_get_wtime() - t0;

    fprintf(stderr, "staged %-14s %8.1f MiB  %6.2f s  %6.2f GB/s  mode=%s\n",
            dst, (double)len / 1048576.0, dt,
            dt > 0 ? (double)len / dt / 1e9 : 0.0,
            interleave ? "interleave" : "serial");

    munmap(dp, len); munmap((void *)sp, len);
    close(dfd); close(sfd);
    return 0;
}

int main(int argc, char **argv)
{
    const char *src = NULL, *dst = NULL;
    int interleave = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--src")  && i + 1 < argc) src = argv[++i];
        else if (!strcmp(argv[i], "--dst")  && i + 1 < argc) dst = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) interleave = !strcmp(argv[++i], "interleave");
        else if (!strcmp(argv[i], "--block-kib") && i + 1 < argc)
            BLOCK = (size_t)atol(argv[++i]) << 10;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!src || !dst) { fprintf(stderr, "--src and --dst required\n"); return 2; }

    if (mkdir(dst, 0700) < 0 && errno != EEXIST) { perror("mkdir dst"); return 1; }

    for (size_t i = 0; i < sizeof FILES / sizeof *FILES; i++) {
        char s[4096], d[4096];
        snprintf(s, sizeof s, "%s/%s", src, FILES[i]);
        snprintf(d, sizeof d, "%s/%s", dst, FILES[i]);
        if (stage_one(s, d, interleave) < 0) return 1;
    }
    return 0;
}
