

#define _GNU_SOURCE

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NUMA_MAP_IMPLEMENTATION
#include "numa_map.h"
#define PARTITION_IMPLEMENTATION
#include "partition.h"
#define TIMING_CSV_IMPLEMENTATION
#include "timing_csv.h"
#define PAPI_WRAP_IMPLEMENTATION
#include "papi_wrap.h"
#include "kernel_core.h"

#define RHO_REF 5.31
#define MAXT 128

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int read_topo(int cpu, const char *what) {
    char p[160];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
    FILE *f = fopen(p, "r");
    if (!f) return -1;
    int v, ok = (fscanf(f, "%d", &v) == 1);
    fclose(f);
    return ok ? v : -1;
}

static double read_ghz_cpu(int cpu) {
    char p[160];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(p, "r");
    if (!f) return -1.0;
    long khz; int ok = (fscanf(f, "%ld", &khz) == 1);
    fclose(f);
    return ok ? (double)khz / 1e6 : -1.0;
}


typedef struct {
    pthread_mutex_t lock;
    int64_t next, end;
    char pad[128 - sizeof(pthread_mutex_t) - 2 * sizeof(int64_t)];
} range_t;

typedef struct {
    int tid, cpu, nthreads, stealing;
    const csr_t   *csr;
    const lookup_t *L;
    const int64_t *walk;
    double *out, *tiles;
    acc_t  *accs;
    int64_t C, F, K, chunk;
    range_t *ranges;
    const int64_t *base_lo, *base_hi;
    pthread_barrier_t *bar;
    /* results */
    double  busy, ghz;
    int64_t nrec, nsta, nsteal;
    int     cpu_end;
    long    papi[PW_NEVENTS];
    unsigned seed;
} targ_t;

static int pop_own(targ_t *a, int64_t *lo, int64_t *hi) {
    range_t *r = &a->ranges[a->tid];
    pthread_mutex_lock(&r->lock);
    if (r->next >= r->end) { pthread_mutex_unlock(&r->lock); return 0; }
    *lo = r->next;
    *hi = r->next + a->chunk;
    if (*hi > r->end) *hi = r->end;
    r->next = *hi;
    pthread_mutex_unlock(&r->lock);
    return 1;
}


static int try_steal(targ_t *a, int64_t *lo, int64_t *hi) {
    for (int att = 0; att < 2 * a->nthreads; att++) {
        int v = (int)(rand_r(&a->seed) % (unsigned)a->nthreads);
        if (v == a->tid) continue;
        range_t *r = &a->ranges[v];
        pthread_mutex_lock(&r->lock);
        int64_t rem = r->end - r->next;
        if (rem >= 2 * a->chunk) {
            int64_t take = rem / 2;
            *hi = r->end;
            *lo = r->end - take;
            r->end = *lo;
            pthread_mutex_unlock(&r->lock);
            return 1;
        }
        pthread_mutex_unlock(&r->lock);
    }
    return 0;
}

static void *worker(void *p) {
    targ_t *a = (targ_t *)p;
    const csr_t *c = a->csr;
    const int64_t F = a->F, C = a->C;

    if (a->cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0)
            fprintf(stderr, "warn: thread %d could not bind to cpu %d\n",
                    a->tid, a->cpu);
    }

    acc_t  *acc  = a->accs  + (size_t)a->tid * a->L->n_slots;
    double *tile = a->tiles + (size_t)a->tid * (C > 0 ? C : 1) * F;
    double  busy = 0.0, ghz_sum = 0.0;
    int     ghz_n = 0;
    int64_t nrec = 0, nsta = 0, nsteal = 0;

    pw_thread_start();

    for (int64_t it = 0; it < a->K; it++) {
        /* thread 0 resets every range; the barriers make the reset visible
         * to all threads before any of them starts claiming */
        if (a->tid == 0)
            for (int j = 0; j < a->nthreads; j++) {
                a->ranges[j].next = a->base_lo[j];
                a->ranges[j].end  = a->base_hi[j];
            }
        pthread_barrier_wait(a->bar);

        const double t0 = now_s();
        for (;;) {
            int64_t lo, hi;
            int stolen = 0;
            if (!pop_own(a, &lo, &hi)) {
                if (!a->stealing || !try_steal(a, &lo, &hi)) break;
                stolen = 1;
            }
            for (int64_t b = lo; b < hi; b += (C > 0 ? C : (hi - lo))) {
                const int64_t e = (C > 0 && b + C < hi) ? b + C : hi;
                for (int64_t t = b; t < e; t++) {
                    const int64_t s = a->walk[t];
                    if (C == 0) {
                        kernel_stay(c->offsets, c->itemid, c->t, c->value,
                                    s, a->L, acc, a->out + s * F);
                    } else {
                        kernel_stay(c->offsets, c->itemid, c->t, c->value,
                                    s, a->L, acc, tile + (t - b) * F);
                    }
                }
                if (C > 0)
                    for (int64_t t = b; t < e; t++)
                        memcpy(a->out + a->walk[t] * F, tile + (t - b) * F,
                               (size_t)F * sizeof(double));
            }
            if (it == 0) {
                nsta += hi - lo;
                for (int64_t t = lo; t < hi; t++)
                    nrec += c->offsets[a->walk[t] + 1] - c->offsets[a->walk[t]];
                nsteal += stolen;
            }
        }
        busy += now_s() - t0;

        a->cpu_end = sched_getcpu();
        { double g = read_ghz_cpu(a->cpu_end); if (g > 0) { ghz_sum += g; ghz_n++; } }
        pthread_barrier_wait(a->bar);
    }

    pw_thread_stop(a->papi);
    a->busy = busy;
    a->ghz  = ghz_n ? ghz_sum / ghz_n : -1.0;
    a->nrec = nrec; a->nsta = nsta; a->nsteal = nsteal;
    return NULL;
}


typedef struct { double c; int64_t i; } pair_t;
static int pair_desc(const void *x, const void *y) {
    const pair_t *a = x, *b = y;
    if (a->c > b->c) return -1;
    if (a->c < b->c) return  1;
    return (a->i < b->i) ? -1 : (a->i > b->i);
}
static int pair_asc(const void *x, const void *y) { return -pair_desc(x, y); }

int main(int argc, char **argv)
{
    const char *csr_dir = NULL, *lookup_path = NULL, *out_path = NULL;
    const char *variant = NULL, *window = "", *ref_path = NULL;
    const char *timings_path = "timings.csv", *threads_path = "threads.csv";
    const char *git_sha = "unknown", *row_order = "canonical";
    const char *whitelist = "natural", *platform = "broadwell";
    const char *cpulist = NULL;
    char part_arg[32] = "block";
    int64_t C = 0, K = 1, chunk = 64;
    int nthreads = 0, reps = 6, interleave = 1, stealing = 1, use_papi = 1;
    double warmup_s = 2.0;
    long job_id = -1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(n) (!strcmp(a, n) && i + 1 < argc)
        if      (ARG("--csr"))         csr_dir = argv[++i];
        else if (ARG("--lookup"))      lookup_path = argv[++i];
        else if (ARG("--variant"))     variant = argv[++i];
        else if (ARG("--window"))      window = argv[++i];
        else if (ARG("--out"))         out_path = argv[++i];
        else if (ARG("--reference"))   ref_path = argv[++i];
        else if (ARG("--timings"))     timings_path = argv[++i];
        else if (ARG("--threads-csv")) threads_path = argv[++i];
        else if (ARG("--git-sha"))     git_sha = argv[++i];
        else if (ARG("--job-id"))      job_id = atol(argv[++i]);
        else if (ARG("--platform"))    platform = argv[++i];
        else if (ARG("--row-order"))   row_order = argv[++i];
        else if (ARG("--whitelist"))   whitelist = argv[++i];
        else if (ARG("--partitioner")) snprintf(part_arg, sizeof part_arg, "%s", argv[++i]);
        else if (ARG("--cpus"))        cpulist = argv[++i];
        else if (ARG("--chunk"))       chunk = atoll(argv[++i]);
        else if (ARG("--tile"))        C = atoll(argv[++i]);
        else if (ARG("--repeat"))      K = atoll(argv[++i]);
        else if (ARG("--reps"))        reps = atoi(argv[++i]);
        else if (ARG("--threads"))     nthreads = atoi(argv[++i]);
        else if (ARG("--warmup-s"))    warmup_s = atof(argv[++i]);
        else if (!strcmp(a, "--no-steal"))     stealing = 0;
        else if (!strcmp(a, "--no-interleave")) interleave = 0;
        else if (!strcmp(a, "--no-papi"))      use_papi = 0;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
        #undef ARG
    }
    if (!csr_dir || !lookup_path) die("--csr DIR --lookup FILE required");
    if (!variant) { const char *b = strrchr(csr_dir, '/'); variant = b ? b + 1 : csr_dir; }


    int cpus[MAXT];
    int ncpu = 0;
    if (cpulist) {
        char buf[1024]; snprintf(buf, sizeof buf, "%s", cpulist);
        for (char *tok = strtok(buf, ","); tok && ncpu < MAXT; tok = strtok(NULL, ","))
            cpus[ncpu++] = atoi(tok);
        if (nthreads <= 0) nthreads = ncpu;
        if (nthreads != ncpu) die("--threads must match the length of --cpus");
    } else {
        if (nthreads <= 0) nthreads = 1;
        for (int i = 0; i < nthreads; i++) cpus[i] = -1;
        fprintf(stderr, "warn: no --cpus given; threads are unbound and "
                        "first-touch locality is not controlled\n");
    }
    if (nthreads > MAXT) die("too many threads");
    if (chunk < 1) chunk = 1;

    csr_t csr;
    if (csr_open(&csr, csr_dir, interleave) != 0) return 1;
    const char *stage_mode = (interleave && csr.mbind_ok) ? "mbind" : "serial";

    lookup_t L = load_lookup(lookup_path);
    if (!strcmp(whitelist, "none")) {
        for (int i = 0; i < MAX_DENSE; i++) L.slot[i] = -1;
    } else if (!strcmp(whitelist, "all")) {
        for (int i = 0; i < MAX_DENSE; i++)
            L.slot[i] = (int16_t)(i % (L.n_slots > 0 ? L.n_slots : 1));
    } else if (strcmp(whitelist, "natural")) die("--whitelist natural|none|all");

    const int64_t n_stays = csr.n_stays;
    const int64_t F = (int64_t)L.n_slots * N_AGG;

    double *n_rec = malloc((size_t)n_stays * sizeof *n_rec);
    double *work  = malloc((size_t)n_stays * sizeof *work);
    if (!n_rec || !work) die("malloc cost");
    int64_t tot_k = 0;
    for (int64_t i = 0; i < n_stays; i++) {
        int64_t r0 = csr.offsets[i], r1 = csr.offsets[i + 1], kk = 0;
        for (int64_t r = r0; r < r1; r++) if (L.slot[csr.itemid[r]] >= 0) kk++;
        n_rec[i] = (double)(r1 - r0);
        work[i]  = (double)(r1 - r0) + RHO_REF * (double)kk;
        tot_k += kk;
    }
    const double hit_rate = csr.n_records ? (double)tot_k / (double)csr.n_records : 0.0;

    int64_t *stay_order = malloc((size_t)n_stays * sizeof *stay_order);
    if (!stay_order) die("malloc stay_order");
    if (!strcmp(row_order, "canonical")) {
        for (int64_t i = 0; i < n_stays; i++) stay_order[i] = i;
    } else if (!strcmp(row_order, "sorted_desc") || !strcmp(row_order, "sorted_asc")) {
        pair_t *v = malloc((size_t)n_stays * sizeof *v);
        if (!v) die("malloc sort");
        for (int64_t i = 0; i < n_stays; i++) { v[i].c = n_rec[i]; v[i].i = i; }
        qsort(v, (size_t)n_stays, sizeof *v,
              !strcmp(row_order, "sorted_desc") ? pair_desc : pair_asc);
        for (int64_t i = 0; i < n_stays; i++) stay_order[i] = v[i].i;
        free(v);
    } else die("--row-order canonical|sorted_desc|sorted_asc");

    double *cost_pos = malloc((size_t)n_stays * sizeof *cost_pos);
    double *nrec_pos = malloc((size_t)n_stays * sizeof *nrec_pos);
    if (!cost_pos || !nrec_pos) die("malloc cost_pos");
    for (int64_t t = 0; t < n_stays; t++) {
        cost_pos[t] = work[stay_order[t]];
        nrec_pos[t] = n_rec[stay_order[t]];
    }

    part_kind_t pk;
    if (part_parse(part_arg, &pk)) die("bad --partitioner");
    partition_t pt;
    const double *key = (pk == PART_NZBALANCED) ? nrec_pos : cost_pos;
    if (partition_build(&pt, pk, nthreads, key, n_stays, chunk) != 0)
        die("partition_build");
    const double pred_eff = partition_efficiency(&pt, cost_pos);

    int64_t *walk = malloc((size_t)n_stays * sizeof *walk);
    if (!walk) die("malloc walk");
    for (int64_t t = 0; t < n_stays; t++)
        walk[t] = stay_order[pt.order ? pt.order[t] : t];

    if (C < 0) C = 0;
    if (C > n_stays) C = n_stays;
    const int64_t Cbuf = C > 0 ? C : 1;

    double *out   = aligned_alloc(64, (size_t)n_stays * F * sizeof(double));
    double *tiles = aligned_alloc(64, (size_t)nthreads * Cbuf * F * sizeof(double));
    acc_t  *accs  = aligned_alloc(64, (size_t)nthreads * L.n_slots * sizeof(acc_t));
    range_t *ranges = aligned_alloc(128, (size_t)nthreads * sizeof(range_t));
    if (!out || !tiles || !accs || !ranges) die("aligned_alloc");

    int64_t base_lo[MAXT], base_hi[MAXT];
    for (int j = 0; j < nthreads; j++) {
        base_lo[j] = pt.offs[j];
        base_hi[j] = pt.offs[j + 1];
        pthread_mutex_init(&ranges[j].lock, NULL);
        ranges[j].next = base_lo[j];
        ranges[j].end  = base_hi[j];
    }

    double *ref = NULL;
    if (ref_path && !strcmp(whitelist, "natural")) {
        FILE *fr = fopen(ref_path, "rb");
        if (!fr) die("open --reference");
        ref = malloc((size_t)n_stays * F * sizeof(double));
        if (!ref) die("malloc reference");
        if (fread(ref, sizeof(double), (size_t)n_stays * F, fr)
            != (size_t)(n_stays * F)) die("short read on --reference");
        fclose(fr);
    }

    int papi_ok = use_papi ? pw_init() : -1;
    fprintf(stderr, "%s%s\n", pw_status(), papi_ok == 0 ? "" : "  -- counters -1");

    fprintf(stderr,
        "csr %s  variant %s  n_stays %" PRId64 "  n_records %" PRId64 "\n"
        "slots %d  F %" PRId64 "  hit %.4f (%s)\n"
        "threads %d  partitioner %s  schedule %s  chunk %" PRId64 "\n"
        "row_order %s  tile %" PRId64 "  repeat %" PRId64 "  reps %d\n"
        "mempolicy %s  predicted work efficiency %.4f\n\n",
        csr_dir, variant, n_stays, csr.n_records, L.n_slots, F, hit_rate,
        whitelist, nthreads, part_name(pk),
        stealing ? "stealing" : "precomputed", chunk,
        row_order, C, K, reps, csr.mempolicy, pred_eff);

    FILE *ft = timing_csv_open(timings_path);
    FILE *fh = thread_csv_open(threads_path);
    if (!ft || !fh) die("open output CSV");

    pthread_barrier_t bar;
    targ_t A[MAXT];
    pthread_t th[MAXT];

 
    memset(out, 0, (size_t)n_stays * F * sizeof(double));
    memset(tiles, 0, (size_t)nthreads * Cbuf * F * sizeof(double));
    memset(accs, 0, (size_t)nthreads * L.n_slots * sizeof(acc_t));

    #define LAUNCH(KK)                                                        \
        do {                                                                  \
            pthread_barrier_init(&bar, NULL, (unsigned)nthreads);              \
            for (int j = 0; j < nthreads; j++) {                               \
                memset(&A[j], 0, sizeof A[j]);                                 \
                A[j].tid = j; A[j].cpu = cpus[j]; A[j].nthreads = nthreads;    \
                A[j].stealing = stealing; A[j].csr = &csr; A[j].L = &L;        \
                A[j].walk = walk; A[j].out = out; A[j].tiles = tiles;          \
                A[j].accs = accs; A[j].C = C; A[j].F = F; A[j].K = (KK);       \
                A[j].chunk = chunk; A[j].ranges = ranges;                      \
                A[j].base_lo = base_lo; A[j].base_hi = base_hi;                \
                A[j].bar = &bar; A[j].seed = (unsigned)(j * 2654435761u + 1);  \
                pthread_create(&th[j], NULL, worker, &A[j]);                   \
            }                                                                  \
            for (int j = 0; j < nthreads; j++) pthread_join(th[j], NULL);      \
            pthread_barrier_destroy(&bar);                                     \
        } while (0)

    if (warmup_s > 0.0) {
        double w0 = now_s(); int passes = 0;
        while (now_s() - w0 < warmup_s) { LAUNCH(1); passes++; }
        fprintf(stderr, "warm-up: %d untimed passes in %.2f s\n\n",
                passes, now_s() - w0);
    }

    for (int rep = 0; rep < reps; rep++) {
        const double t0 = now_s();
        LAUNCH(K);
        const double wall = now_s() - t0;

        double max_t = 0, sum_t = 0, ghz_s = 0; int ghz_n = 0;
        int64_t max_r = 0, sum_r = 0, steals = 0;
        long tot[PW_NEVENTS];
        for (int e = 0; e < PW_NEVENTS; e++) tot[e] = -1;
        for (int j = 0; j < nthreads; j++) {
            if (A[j].busy > max_t) max_t = A[j].busy;
            sum_t += A[j].busy;
            if (A[j].nrec > max_r) max_r = A[j].nrec;
            sum_r += A[j].nrec; steals += A[j].nsteal;
            if (A[j].ghz > 0) { ghz_s += A[j].ghz; ghz_n++; }
            for (int e = 0; e < PW_NEVENTS; e++)
                if (A[j].papi[e] >= 0)
                    tot[e] = (tot[e] < 0 ? 0 : tot[e]) + A[j].papi[e];
        }
        const double mean_r = (double)sum_r / nthreads;

        int n_sock[2] = {0, 0}, smt_used = 0, topo_ok = 1, seen[2][256];
        memset(seen, 0, sizeof seen);
        for (int j = 0; j < nthreads; j++) {
            int sk = read_topo(A[j].cpu_end, "physical_package_id");
            int cr = read_topo(A[j].cpu_end, "core_id");
            if (sk < 0 || sk > 1 || cr < 0 || cr >= 256) { topo_ok = 0; break; }
            n_sock[sk]++;
            if (seen[sk][cr]) smt_used = 1;
            seen[sk][cr] = 1;
        }
        const char *placement = !topo_ok ? "unknown" : smt_used ? "smt"
                              : n_sock[1] == 0 ? "socket0"
                              : n_sock[0] == 0 ? "socket1" : "spread";

        const char *bitid = "skip";
        if (ref) bitid = memcmp(ref, out, (size_t)n_stays * F * sizeof(double))
                         ? "fail" : "pass";

        char rid[64];
        timing_make_run_id(rid, sizeof rid, job_id, 0, rep);
        timing_row_t r; timing_row_init(&r);
        r.run_id = rid; r.job_id = job_id; r.git_sha = git_sha;
        { static char host[128]; gethostname(host, sizeof host); r.node = host; }
        { static char ts[32]; time_t nw = time(NULL);
          strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", localtime(&nw)); r.timestamp = ts; }
        r.csr_variant = variant; r.window_W = window;
        r.n_stays = n_stays; r.n_records = csr.n_records;
        r.paradigm = "pthreads"; r.platform = platform;
        r.partitioner = part_name(pk);
        r.schedule = stealing ? "stealing" : "precomputed";
        r.chunk_size = stealing ? chunk : -1;
        r.row_order = row_order;
        r.n_threads = nthreads; r.smt = (topo_ok && smt_used);
        r.thread_placement = placement;
        r.stage_mode = stage_mode; r.mempolicy = csr.mempolicy;
        r.whitelist_frac = !strcmp(whitelist, "none") ? 0.0
                         : !strcmp(whitelist, "all")  ? 1.0 : hit_rate;
        r.k_iterations = K; r.timed_region = "kernel";
        r.repetition = rep; r.warmup = (rep == 0);
        r.wall_time_s = wall;
        r.throughput_rec_s = (double)csr.n_records * (double)K / wall;
        r.work_imbalance = mean_r > 0 ? (double)max_r / mean_r : 0.0;
        r.time_imbalance = sum_t > 0 ? max_t / (sum_t / nthreads) : 0.0;
        r.n_records_max_thread = max_r;
        r.n_records_mean_thread = mean_r;
        r.bitidentical = bitid;
        r.l3_tcm = tot[PW_L3_TCM]; r.l3_tca = tot[PW_L3_TCA];
        r.tot_cyc = tot[PW_TOT_CYC]; r.tot_ins = tot[PW_TOT_INS];
        r.achieved_ghz = ghz_n ? ghz_s / ghz_n : -1.0;
        { static char note[128];
          snprintf(note, sizeof note,
                   "tile=%" PRId64 ";pred_eff=%.4f;steals=%" PRId64,
                   C, pred_eff, steals);
          r.notes = note; }
        if (!timing_csv_write(ft, &r)) die("invalid timing row");

        for (int j = 0; j < nthreads; j++) {
            int sk = read_topo(A[j].cpu_end, "physical_package_id");
            thread_row_t tr = { TIMING_SCHEMA_VERSION, rid, j, A[j].cpu_end,
                                sk >= 0 ? sk : 0, A[j].nsta, A[j].nrec, A[j].busy };
            thread_csv_write(fh, &tr);
        }

        fprintf(stderr, "rep %d%s  wall %.6f s  %.1f M rec/s  work_imb %.4f  "
                        "time_imb %.4f  %.2f GHz  %-7s  steals %" PRId64
                        "  bitid %s\n",
                rep, rep ? "" : " (warmup)", wall, r.throughput_rec_s / 1e6,
                r.work_imbalance, r.time_imbalance, r.achieved_ghz,
                placement, steals, bitid);
    }

    fclose(ft); fclose(fh);
    if (out_path) {
        FILE *fo = fopen(out_path, "wb");
        if (!fo) die("open --out");
        fwrite(out, sizeof(double), (size_t)n_stays * F, fo);
        fclose(fo);
    }
    for (int j = 0; j < nthreads; j++) pthread_mutex_destroy(&ranges[j].lock);
    free(n_rec); free(work); free(cost_pos); free(nrec_pos); free(stay_order);
    free(walk); free(out); free(tiles); free(accs); free(ranges); free(ref);
    partition_free(&pt);
    csr_close(&csr);
    return 0;
}
