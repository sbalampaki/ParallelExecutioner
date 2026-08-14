#define _POSIX_C_SOURCE 200809L

#include "kernel_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void kc_die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}


lookup_t load_lookup(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); kc_die("open lookup"); }

    lookup_t L;
    memset(&L, 0, sizeof L);
    L.slot    = calloc(MAX_DENSE, sizeof *L.slot);
    L.vscale  = calloc(MAX_DENSE, sizeof *L.vscale);
    L.voffset = calloc(MAX_DENSE, sizeof *L.voffset);
    L.lo      = calloc(MAX_DENSE, sizeof *L.lo);
    L.hi      = calloc(MAX_DENSE, sizeof *L.hi);
    if (!L.slot || !L.vscale || !L.voffset || !L.lo || !L.hi)
        kc_die("calloc lookup");
    for (int i = 0; i < MAX_DENSE; i++) L.slot[i] = -1;

    char line[4096];
    if (!fgets(line, sizeof line, f)) kc_die("empty lookup");   /* header */

    int32_t max_id = -1, max_slot = -1;
    long rows = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        double fld[6];
        for (int j = 0; j < 6; j++) {
            char *end;
            fld[j] = strtod(p, &end);
            if (end == p) kc_die("malformed lookup row");
            p = end;
            if (*p == ',') p++;
        }
        int32_t id = (int32_t)fld[0];
        if (id < 0 || id >= MAX_DENSE) kc_die("dense_id out of range");
        L.slot[id]    = (int16_t)fld[1];
        L.vscale[id]  = (float)fld[2];
        L.voffset[id] = (float)fld[3];
        L.lo[id]      = (float)fld[4];
        L.hi[id]      = (float)fld[5];
        if (id > max_id) max_id = id;
        if (L.slot[id] > max_slot) max_slot = L.slot[id];
        rows++;
    }
    fclose(f);

    L.k = max_id + 1;
    L.n_slots = max_slot + 1;
    if (rows != L.k)
        fprintf(stderr, "WARN: %ld rows but max dense_id+1 = %d "
                        "(gaps in the dense id space?)\n", rows, L.k);
    return L;
}


void kernel_stay(const int64_t *restrict offsets,
                 const int32_t *restrict itemid,
                 const float   *restrict t,
                 const float   *restrict value,
                 int64_t                 i,
                 const lookup_t *restrict L,
                 acc_t          *restrict acc,
                 double         *restrict row)
{
    const int32_t S = L->n_slots;

    for (int32_t s = 0; s < S; s++) {
        acc[s].n = 0;
        acc[s].kv = acc[s].kt = 0.0;
        acc[s].sv = acc[s].svv = 0.0;
        acc[s].st = acc[s].stt = 0.0;
        acc[s].stv = 0.0;
        acc[s].vmin = acc[s].vmax = acc[s].vlast = 0.0;
    }

    const int64_t r0 = offsets[i], r1 = offsets[i + 1];
    for (int64_t r = r0; r < r1; r++) {
        const int32_t id = itemid[r];
        const int16_t s  = L->slot[id];
        if (s < 0) continue;               /* not whitelisted */

      
        double v = (double)value[r] * (double)L->vscale[id]
                                    + (double)L->voffset[id];
        const double clo = (double)L->lo[id];
        const double chi = (double)L->hi[id];
        if (v < clo) v = clo;
        else if (v > chi) v = chi;

        const double tt = (double)t[r];
        acc_t *a = &acc[s];

        if (a->n == 0) {
            a->kv = v;              
            a->kt = tt;
            a->vmin = a->vmax = v;
        }
        a->vlast = v;
        if (v < a->vmin) a->vmin = v;
        if (v > a->vmax) a->vmax = v;

 
        const double dv = v  - a->kv;
        const double dt = tt - a->kt;
        a->n++;
        a->sv  += dv;
        a->svv += dv * dv;
        a->st  += dt;
        a->stt += dt * dt;
        a->stv += dt * dv;
    }

    for (int32_t s = 0; s < S; s++) {
        const acc_t *a = &acc[s];
        double *o = row + (int64_t)s * N_AGG;
        if (a->n == 0) {
            o[0] = 0.0;
            o[1] = o[2] = o[3] = o[4] = o[5] = o[6] = o[7] = NAN;
        } else {
        
            const double n   = (double)a->n;
            const double inv = 1.0 / n;
            double var = (a->svv - a->sv * a->sv * inv) * inv;
            if (var < 0.0) var = 0.0;      /* rounding, n==1 */
            const double t_ss = a->stt - a->st * a->st * inv;
            const double cov  = a->stv - a->st * a->sv * inv;

            o[0] = n;
            o[1] = a->kv + a->sv * inv;
            o[2] = a->vmin;
            o[3] = a->vmax;
            o[4] = sqrt(var);                     /* population */
            o[5] = a->kv;                         /* first       */
            o[6] = a->vlast;
       
            o[7] = (a->n >= 2 && t_ss > 0.0) ? cov / t_ss : NAN;
        }
    }
}
