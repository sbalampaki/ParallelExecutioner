
#ifndef NUMA_MAP_H
#define NUMA_MAP_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const int64_t  *offsets;    /* n_stays + 1 */
    const int32_t  *itemid;     /* n_records, dense-remapped */
    const float    *t;          /* n_records, hours since intime */
    const float    *value;      /* n_records */
    int64_t         n_stays;
    int64_t         n_records;

    /* bookkeeping for the timing CSV */
    const char     *mempolicy;  /* "interleave" | "default" -- what took effect */
    int             mbind_ok;   /* 1 if every mbind succeeded */
    char            note[128];  /* errno text when mbind fails */

    void           *_base[4];   /* for munmap */
    size_t          _len[4];
} csr_t;


int  csr_open(csr_t *c, const char *dir, int interleave);
void csr_close(csr_t *c);
int  numa_node_count(void);

#endif /* NUMA_MAP_H */

#ifdef NUMA_MAP_IMPLEMENTATION

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>


extern long syscall(long number, ...);


#ifndef MAP_POPULATE
#define MAP_POPULATE 0x08000
#endif

#define MPOL_INTERLEAVE 3
#define MPOL_MF_MOVE    (1 << 1)

int numa_node_count(void)
{
    int n = 0;
    for (int i = 0; i < 64; i++) {
        char p[80];
        snprintf(p, sizeof p, "/sys/devices/system/node/node%d", i);
        if (access(p, F_OK) != 0) break;
        n++;
    }
    return n ? n : 1;
}

static int nm__mbind_interleave(void *addr, size_t len, char *err, size_t errlen)
{
    int nn = numa_node_count();
    unsigned long mask = (nn >= 64) ? ~0UL : ((1UL << nn) - 1UL);

    long r = syscall(SYS_mbind, addr, (unsigned long)len, MPOL_INTERLEAVE,
                     &mask, (unsigned long)(nn + 1), MPOL_MF_MOVE);
    if (r != 0) {
        snprintf(err, errlen, "mbind: %s", strerror(errno));
        return -1;
    }
    return 0;
}

static void *nm__map(const char *path, size_t *len_out)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "csr_open: %s: %s\n", path, strerror(errno)); return NULL; }
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t len = (size_t)st.st_size;

    void *p = mmap(NULL, len, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { fprintf(stderr, "csr_open: mmap %s: %s\n", path, strerror(errno)); return NULL; }
    *len_out = len;
    return p;
}

int csr_open(csr_t *c, const char *dir, int interleave)
{
    static const char *names[4] = { "offsets.i64", "itemid.i32", "t.f32", "value.f32" };
    memset(c, 0, sizeof *c);
    c->mempolicy = "default";
    c->mbind_ok = 1;

    for (int i = 0; i < 4; i++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        c->_base[i] = nm__map(path, &c->_len[i]);
        if (!c->_base[i]) return -1;
    }

    if (interleave) {
        for (int i = 0; i < 4; i++)
            if (nm__mbind_interleave(c->_base[i], c->_len[i],
                                     c->note, sizeof c->note) != 0) {
                c->mbind_ok = 0;
                break;
            }
        c->mempolicy = c->mbind_ok ? "interleave" : "default";
        if (!c->mbind_ok)
            fprintf(stderr, "csr_open: WARNING %s -- falling back to "
                            "first-touch placement from stage_csr. Placement "
                            "will drift under AutoNUMA (D33).\n", c->note);
    }

    c->offsets   = (const int64_t *)c->_base[0];
    c->itemid    = (const int32_t *)c->_base[1];
    c->t         = (const float   *)c->_base[2];
    c->value     = (const float   *)c->_base[3];
    c->n_stays   = (int64_t)(c->_len[0] / 8) - 1;
    c->n_records = c->offsets[c->n_stays];

    if ((size_t)c->n_records * 4 != c->_len[1]
        || c->_len[1] != c->_len[2] || c->_len[2] != c->_len[3]) {
        fprintf(stderr, "csr_open: size mismatch in %s\n", dir);
        return -1;
    }
    return 0;
}

void csr_close(csr_t *c)
{
    for (int i = 0; i < 4; i++)
        if (c->_base[i]) munmap(c->_base[i], c->_len[i]);
    memset(c, 0, sizeof *c);
}

#endif 
