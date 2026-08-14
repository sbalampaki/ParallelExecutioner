
#ifndef KERNEL_CORE_H
#define KERNEL_CORE_H

#include <stdint.h>

#define N_AGG     8
#define MAX_DENSE 65536


typedef struct {
    int32_t  k;                 /* number of dense ids            */
    int32_t  n_slots;           /* distinct output variables      */
    int16_t *slot;              /* -1 = drop                      */
    float   *vscale, *voffset;  /* unit harmonisation             */
    float   *lo, *hi;           /* clip bounds, harmonised units  */
} lookup_t;


typedef struct {
    int64_t n;
    double  kv, kt;         /* shift constants = first (v, t) for the slot */
    double  sv, svv;        /* sum(v-kv),  sum((v-kv)^2)                   */
    double  st, stt;        /* sum(t-kt),  sum((t-kt)^2)                   */
    double  stv;            /* sum((t-kt)(v-kv))                           */
    double  vmin, vmax;
    double  vlast;          
} acc_t;

lookup_t load_lookup(const char *path);

void kernel_stay(const int64_t *restrict offsets,
                 const int32_t *restrict itemid,
                 const float   *restrict t,
                 const float   *restrict value,
                 int64_t                 i,
                 const lookup_t *restrict L,
                 acc_t          *restrict acc,
                 double         *restrict row);

#endif /* KERNEL_CORE_H */
