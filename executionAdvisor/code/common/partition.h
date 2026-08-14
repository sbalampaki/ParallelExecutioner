
#ifndef PARTITION_H
#define PARTITION_H

#include <stdint.h>

typedef enum {
    PART_BLOCK = 0,     /* contiguous, equal STAY count        */
    PART_CYCLIC,        /* stay i -> worker i % p             */
    PART_BLOCK_CYCLIC,  /* chunks of `chunk` stays, round-robin    */
    PART_NZBALANCED,    /* contiguous, equal RECORD count      */
    PART_WBALANCED,     /* contiguous, equal WORK             */
    PART_CONTIG_OPT,    /* optimal contiguous, on work        */
    PART_GREEDY,        /* LPT: sort desc, least-loaded first       */
    PART_NONE           /* runtime-determined; no precomputed map  */
} part_kind_t;

typedef struct {
    part_kind_t kind;
    int         p;
    int         contiguous;   /* 1 -> order is identity; walk offs[j]..offs[j+1] */
    int64_t    *offs;         /* p + 1 */
    int64_t    *order;        /* n_stays, NULL when contiguous */
    int64_t     n_stays;
} partition_t;

const char *part_name(part_kind_t k);
int         part_parse(const char *s, part_kind_t *out);


int    partition_build(partition_t *pt, part_kind_t kind, int p,
                       const double *cost, int64_t n_stays, int64_t chunk);
void   partition_free(partition_t *pt);


double partition_efficiency(const partition_t *pt, const double *cost);


void   partition_loads(const partition_t *pt, const double *cost, double *loads);

#endif /* PARTITION_H */


#ifdef PARTITION_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *PART_NAMES[] = {
    "block", "cyclic", "block_cyclic", "nzbalanced",
    "wbalanced", "contig_opt", "greedy", "none"
};

const char *part_name(part_kind_t k) { return PART_NAMES[(int)k]; }

int part_parse(const char *s, part_kind_t *out)
{
    for (int i = 0; i < 8; i++)
        if (!strcmp(s, PART_NAMES[i])) { *out = (part_kind_t)i; return 0; }
    return -1;
}

/* ---- helpers ------ */

static void pt__linspace_cuts(int64_t n, int p, int64_t *cuts)
{
    double step = (double)n / (double)p;
    for (int j = 0; j < p; j++) cuts[j] = (int64_t)((double)j * step);
    cuts[p] = n;
    for (int j = 1; j <= p; j++)
        if (cuts[j] < cuts[j - 1]) cuts[j] = cuts[j - 1];
}

/* first index i with cs[i] >= v, i.e. numpy searchsorted(side='left') */
static int64_t pt__lower_bound(const double *cs, int64_t n, double v)
{
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (cs[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static void pt__balanced_cuts(const double *key, int64_t n, int p, int64_t *cuts)
{
    double *cs = (double *)malloc((size_t)(n + 1) * sizeof *cs);
    cs[0] = 0.0;
    for (int64_t i = 0; i < n; i++) cs[i + 1] = cs[i] + key[i];

    for (int j = 0; j <= p; j++) {
        double target = cs[n] * (double)j / (double)p;
        int64_t idx = pt__lower_bound(cs, n + 1, target);
        int64_t lo = idx > 0 ? idx - 1 : 0;
        int64_t hi = idx <= n ? idx : n;
        double dlo = cs[lo] - target; if (dlo < 0) dlo = -dlo;
        double dhi = cs[hi] - target; if (dhi < 0) dhi = -dhi;
        cuts[j] = (dlo <= dhi) ? lo : hi;
    }
    cuts[0] = 0; cuts[p] = n;
    for (int j = 1; j <= p; j++)
        if (cuts[j] < cuts[j - 1]) cuts[j] = cuts[j - 1];
    cuts[p] = n;
    free(cs);
}


static int pt__feasible(const double *cost, int64_t n, int p, double cap,
                        int64_t *cuts)
{
    int blocks = 1; double cur = 0.0;
    if (cuts) cuts[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        if (cost[i] > cap) return 0;
        if (cur + cost[i] > cap) {
            if (cuts && blocks <= p) cuts[blocks] = i;
            blocks++; cur = cost[i];
            if (blocks > p) return 0;
        } else cur += cost[i];
    }
    if (cuts) { for (int j = blocks; j <= p; j++) cuts[j] = n; }
    return 1;
}

static void pt__contig_opt_cuts(const double *cost, int64_t n, int p, int64_t *cuts)
{
    double sum = 0.0, mx = 0.0;
    for (int64_t i = 0; i < n; i++) { sum += cost[i]; if (cost[i] > mx) mx = cost[i]; }
    double lo = (mx > sum / p) ? mx : sum / p, hi = sum;
    for (int it = 0; it < 60; it++) {
        double mid = 0.5 * (lo + hi);
        if (pt__feasible(cost, n, p, mid, NULL)) hi = mid; else lo = mid;
    }
    pt__feasible(cost, n, p, hi, cuts);
    cuts[0] = 0; cuts[p] = n;
    for (int j = 1; j <= p; j++)
        if (cuts[j] < cuts[j - 1]) cuts[j] = cuts[j - 1];
    cuts[p] = n;
}

typedef struct { double c; int64_t i; } pt_pair_t;
static int pt__cmp_desc(const void *a, const void *b)
{
    const pt_pair_t *x = (const pt_pair_t *)a, *y = (const pt_pair_t *)b;
    if (x->c > y->c) return -1;
    if (x->c < y->c) return  1;
    return (x->i < y->i) ? -1 : (x->i > y->i);   /* stable: index breaks ties */
}


int partition_build(partition_t *pt, part_kind_t kind, int p,
                    const double *cost, int64_t n, int64_t chunk)
{
    memset(pt, 0, sizeof *pt);
    pt->kind = kind; pt->p = p; pt->n_stays = n;
    pt->offs = (int64_t *)calloc((size_t)p + 1, sizeof *pt->offs);
    if (!pt->offs) return -1;

    switch (kind) {
    case PART_BLOCK:
        pt->contiguous = 1; pt__linspace_cuts(n, p, pt->offs); return 0;

    case PART_NZBALANCED:
    case PART_WBALANCED:
        pt->contiguous = 1; pt__balanced_cuts(cost, n, p, pt->offs); return 0;

    case PART_CONTIG_OPT:
        pt->contiguous = 1; pt__contig_opt_cuts(cost, n, p, pt->offs); return 0;

    case PART_CYCLIC:
    case PART_BLOCK_CYCLIC: {
        int64_t c = (kind == PART_CYCLIC) ? 1 : (chunk > 0 ? chunk : 1);
        pt->contiguous = 0;
        pt->order = (int64_t *)malloc((size_t)n * sizeof *pt->order);
        if (!pt->order) return -1;
        /* count per worker, then fill */
        for (int64_t i = 0; i < n; i++) {
            int j = (int)((i / c) % (int64_t)p);
            pt->offs[j + 1]++;
        }
        for (int j = 0; j < p; j++) pt->offs[j + 1] += pt->offs[j];
        int64_t *cur = (int64_t *)malloc((size_t)p * sizeof *cur);
        for (int j = 0; j < p; j++) cur[j] = pt->offs[j];
        for (int64_t i = 0; i < n; i++) {
            int j = (int)((i / c) % (int64_t)p);
            pt->order[cur[j]++] = i;
        }
        free(cur);
        return 0;
    }

    case PART_GREEDY: {
        pt->contiguous = 0;
        pt->order = (int64_t *)malloc((size_t)n * sizeof *pt->order);
        pt_pair_t *v = (pt_pair_t *)malloc((size_t)n * sizeof *v);
        if (!pt->order || !v) return -1;
        for (int64_t i = 0; i < n; i++) { v[i].c = cost[i]; v[i].i = i; }
        qsort(v, (size_t)n, sizeof *v, pt__cmp_desc);

        double *load = (double *)calloc((size_t)p, sizeof *load);
        int64_t *cnt = (int64_t *)calloc((size_t)p, sizeof *cnt);
        int64_t **bin = (int64_t **)malloc((size_t)p * sizeof *bin);
        int64_t *cap = (int64_t *)malloc((size_t)p * sizeof *cap);
        for (int j = 0; j < p; j++) { cap[j] = 16; bin[j] = malloc(16 * sizeof **bin); }

        for (int64_t t = 0; t < n; t++) {           /* O(np); p is small */
            int best = 0;
            for (int j = 1; j < p; j++) if (load[j] < load[best]) best = j;
            if (cnt[best] == cap[best]) {
                cap[best] *= 2;
                bin[best] = realloc(bin[best], (size_t)cap[best] * sizeof **bin);
            }
            bin[best][cnt[best]++] = v[t].i;
            load[best] += v[t].c;
        }
        int64_t w = 0;
        for (int j = 0; j < p; j++) {
            pt->offs[j] = w;
            memcpy(pt->order + w, bin[j], (size_t)cnt[j] * sizeof **bin);
            w += cnt[j];
            free(bin[j]);
        }
        pt->offs[p] = w;
        free(bin); free(cap); free(load); free(cnt); free(v);
        return 0;
    }

    case PART_NONE:
        pt->contiguous = 1; pt__linspace_cuts(n, p, pt->offs); return 0;
    }
    return -1;
}

void partition_free(partition_t *pt)
{
    free(pt->offs); free(pt->order);
    memset(pt, 0, sizeof *pt);
}

void partition_loads(const partition_t *pt, const double *cost, double *loads)
{
    for (int j = 0; j < pt->p; j++) {
        double s = 0.0;
        if (pt->contiguous)
            for (int64_t i = pt->offs[j]; i < pt->offs[j + 1]; i++) s += cost[i];
        else
            for (int64_t t = pt->offs[j]; t < pt->offs[j + 1]; t++) s += cost[pt->order[t]];
        loads[j] = s;
    }
}

double partition_efficiency(const partition_t *pt, const double *cost)
{
    double *l = (double *)malloc((size_t)pt->p * sizeof *l);
    partition_loads(pt, cost, l);
    double mx = 0.0, sum = 0.0;
    for (int j = 0; j < pt->p; j++) { sum += l[j]; if (l[j] > mx) mx = l[j]; }
    free(l);
    return mx > 0.0 ? (sum / pt->p) / mx : 0.0;
}

#endif /* PARTITION_IMPLEMENTATION */
