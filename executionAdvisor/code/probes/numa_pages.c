
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAXNODE 16
#define BATCH   4096

static int n_numa_nodes(void)
{
    int n = 0;
    for (int i = 0; i < MAXNODE; i++) {
        char p[80];
        snprintf(p, sizeof p, "/sys/devices/system/node/node%d", i);
        if (access(p, F_OK) == 0) n++;
    }
    return n ? n : 1;
}

static int report(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "numa_pages: open %s: %s\n", path, strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "numa_pages: fstat %s: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    size_t len = (size_t)st.st_size;
    if (len == 0) { close(fd); return 0; }

    char *base = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "numa_pages: mmap %s: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }

    long   ps     = sysconf(_SC_PAGESIZE);
    size_t npages = (len + (size_t)ps - 1) / (size_t)ps;


    {
        volatile unsigned char sink = 0;
        for (size_t i = 0; i < npages; i++) sink ^= (unsigned char)base[i * (size_t)ps];
        (void)sink;
    }

    void **pages  = malloc((size_t)BATCH * sizeof *pages);
    int   *status = malloc((size_t)BATCH * sizeof *status);
    if (!pages || !status) { fprintf(stderr, "numa_pages: OOM\n"); return -1; }

    long counts[MAXNODE];
    memset(counts, 0, sizeof counts);
    long unknown = 0;

    for (size_t off = 0; off < npages; off += BATCH) {
        size_t n = (npages - off < (size_t)BATCH) ? (npages - off) : (size_t)BATCH;
        for (size_t i = 0; i < n; i++) pages[i] = base + (off + i) * (size_t)ps;
        memset(status, 0, n * sizeof *status);

        long r = syscall(SYS_move_pages, 0, (unsigned long)n, pages,
                         (const int *)NULL, status, 0);
        if (r < 0) {
            fprintf(stderr, "numa_pages: move_pages: %s\n", strerror(errno));
            free(pages); free(status);
            munmap(base, len); close(fd);
            return -1;
        }
        for (size_t i = 0; i < n; i++) {
            if (status[i] >= 0 && status[i] < MAXNODE) counts[status[i]]++;
            else unknown++;
        }
    }

    int nn = n_numa_nodes();
    const char *bn = strrchr(path, '/');
    bn = bn ? bn + 1 : path;

    for (int nd = 0; nd < nn; nd++) {
        double frac = npages ? (double)counts[nd] / (double)npages : 0.0;
        printf("NUMA,%s,%s,node%d,%ld,%zu,%.4f,%.1f\n",
               path, bn, nd, counts[nd], npages, frac,
               (double)counts[nd] * (double)ps / 1048576.0);
    }
    if (unknown)
        printf("NUMA,%s,%s,unknown,%ld,%zu,%.4f,0.0\n",
               path, bn, unknown, npages,
               npages ? (double)unknown / (double)npages : 0.0);

    free(pages); free(status);
    munmap(base, len);
    close(fd);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s FILE [FILE ...]\n", argv[0]);
        return 2;
    }
    printf("NUMA,path,file,node,pages,total_pages,frac,mib\n");
    int rc = 0;
    for (int i = 1; i < argc; i++)
        if (report(argv[i]) < 0) rc = 1;
    return rc;
}
