
#define _POSIX_C_SOURCE 200809L

#define NUMA_MAP_IMPLEMENTATION
#include "numa_map.h"
#define PARTITION_IMPLEMENTATION
#include "partition.h"
#include "kernel_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RHO_REF 5.31    

int main(int argc, char **argv)
{
    const char *csr_dir = NULL, *lookup_path = NULL, *variant = NULL;
    int interleave = 1;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--csr")     && i + 1 < argc) csr_dir = argv[++i];
        else if (!strcmp(argv[i], "--lookup")  && i + 1 < argc) lookup_path = argv[++i];
        else if (!strcmp(argv[i], "--variant") && i + 1 < argc) variant = argv[++i];
        else if (!strcmp(argv[i], "--no-interleave")) interleave = 0;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!csr_dir || !lookup_path) {
        fprintf(stderr, "usage: --csr DIR --lookup FILE [--variant NAME] "
                        "[--no-interleave]\n");
        return 2;
    }
    if (!variant) {   /* default: basename of the CSR dir */
        const char *b = strrchr(csr_dir, '/');
        variant = b ? b + 1 : csr_dir;
    }

    /* ---- numa_map.h  */
    csr_t c;
    if (csr_open(&c, csr_dir, interleave) != 0) return 1;

    fprintf(stderr, "numa_map: nodes=%d  n_stays=%lld  n_records=%lld\n",
            numa_node_count(), (long long)c.n_stays, (long long)c.n_records);
    fprintf(stderr, "numa_map: mempolicy=%s  mbind_ok=%d%s%s\n",
            c.mempolicy, c.mbind_ok, c.note[0] ? "  " : "", c.note);
    if (interleave && !c.mbind_ok)
        fprintf(stderr, "numa_map: ** mbind did NOT take effect. Placement "
                        "will drift under AutoNUMA (D33). Investigate before "
                        "any timed run.\n");

    /* ---- per-stay cost vectors  */
    lookup_t L = load_lookup(lookup_path);
    fprintf(stderr, "lookup  : K=%d  slots=%d\n", L.k, L.n_slots);

    double *n_rec = malloc((size_t)c.n_stays * sizeof *n_rec);
    double *work  = malloc((size_t)c.n_stays * sizeof *work);
    if (!n_rec || !work) { fprintf(stderr, "OOM\n"); return 1; }

    int64_t tot_k = 0;
    for (int64_t i = 0; i < c.n_stays; i++) {
        int64_t r0 = c.offsets[i], r1 = c.offsets[i + 1];
        int64_t k = 0;
        for (int64_t r = r0; r < r1; r++)
            if (L.slot[c.itemid[r]] >= 0) k++;
        n_rec[i] = (double)(r1 - r0);
        work[i]  = (double)(r1 - r0) + RHO_REF * (double)k;
        tot_k += k;
    }
    fprintf(stderr, "cost    : pooled hit rate = %.4f  (D34 rho = %.2f)\n\n",
            (double)tot_k / (double)c.n_records, RHO_REF);

    /* ---- partition.h  */
    const int plist[6] = { 8, 16, 32, 64, 80, 96 };
    const int n_plist = (int)(sizeof plist / sizeof *plist);
    const part_kind_t ks[7] = { PART_BLOCK, PART_CYCLIC, PART_BLOCK_CYCLIC,
                                PART_NZBALANCED, PART_WBALANCED,
                                PART_CONTIG_OPT, PART_GREEDY };

    printf("variant,p,scored_on,strategy,efficiency\n");
    for (int ci = 0; ci < 2; ci++) {
        const double *score = ci ? work : n_rec;          /* score column */
        const char   *label = ci ? "work" : "records";
        for (int a = 0; a < n_plist; a++) {
            int p = plist[a];
            if (p > c.n_stays) continue;
            for (int b = 0; b < 7; b++) {
                const double *key = score;
                if (ks[b] == PART_NZBALANCED)      key = n_rec;
                else if (ks[b] == PART_WBALANCED)  key = work;

                partition_t pt;
                if (partition_build(&pt, ks[b], p, key, c.n_stays, 64) != 0) {
                    fprintf(stderr, "partition_build failed: %s p=%d\n",
                            part_name(ks[b]), p);
                    return 1;
                }
                printf("%s,%d,%s,%s,%.12f\n", variant, p, label,
                       part_name(ks[b]), partition_efficiency(&pt, score));
                partition_free(&pt);
            }
        }
    }

    free(n_rec); free(work);
    csr_close(&c);
    return 0;
}
