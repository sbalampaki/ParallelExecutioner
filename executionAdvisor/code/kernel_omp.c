

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUMA_MAP_IMPLEMENTATION
#include "numa_map.h"
#define PARTITION_IMPLEMENTATION
#include "partition.h"
#define TIMING_CSV_IMPLEMENTATION
#include "timing_csv.h"
#define PAPI_WRAP_IMPLEMENTATION
#include "papi_wrap.h"
#include "kernel_core.h"

/* Declared rather than pulled in via _GNU_SOURCE, for the same reason
 * numa_map.h declares syscall(): feature-test-macro ordering across four
 * drivers is a silent trap. */
extern int sched_getcpu(void);

#define RHO_REF 5.31    /* D34: b/a from D29's 915 / 145 M rec/s fixtures */

typedef enum { SCH_PRECOMPUTED, SCH_STATIC, SCH_STATIC_CHUNKED,
               SCH_DYNAMIC, SCH_GUIDED } sched_kind_t;

static const char *SCHED_NAMES[] = { "precomputed", "static", "static_chunked",
                                     "dynamic", "guided" };

static int sched_parse(const char *s, sched_kind_t *out) {
    for (int i = 0; i < 5; i++)
        if (!strcmp(s, SCHED_NAMES[i])) { *out = (sched_kind_t)i; return 0; }
    return -1;
}

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
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
    char p[128];
    snprintf(p, sizeof p,
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(p, "r");
    if (!f) return -1.0;
    long khz;
    int ok = (fscanf(f, "%ld", &khz) == 1);
    fclose(f);
    return ok ? (double)khz / 1e6 : -1.0;
}


typedef struct { double c; int64_t i; } pair_t;
static int pair_desc(const void *a, const void *b) {
    const pair_t *x = a, *y = b;
    if (x->c > y->c) return -1;
    if (x->c < y->c) return  1;
    return (x->i < y->i) ? -1 : (x->i > y->i);
}
static int pair_asc(const void *a, const void *b) { return -pair_desc(a, b); }

int main(int argc, char **argv)
{
    const char *csr_dir = NULL, *lookup_path = NULL, *out_path = NULL;
    const char *variant = NULL, *window = "", *ref_path = NULL;
    const char *timings_path = "timings.csv", *threads_path = "threads.csv";
    const char *git_sha = "unknown", *row_order = "canonical";
    const char *whitelist = "natural", *stage_mode = "interleave";
    const char *platform = "broadwell", *timed_region = "kernel";
    const char *user_note = NULL;
    char part_arg[32] = "block";
    sched_kind_t skind = SCH_PRECOMPUTED;
    int64_t C = 4096, K = 1, chunk = -1;
    double warmup_s = 2.0;
    int nthreads = 0, reps = 6, interleave = 1, smt = 0, use_papi = 1;
    long job_id = -1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(name) (!strcmp(a, name) && i + 1 < argc)
        if      (ARG("--csr"))         csr_dir = argv[++i];
        else if (ARG("--lookup"))      lookup_path = argv[++i];
        else if (ARG("--variant"))     variant = argv[++i];
        else if (ARG("--window"))      window = argv[++i];
        else if (ARG("--out"))         out_path = argv[++i];
        else if (ARG("--reference"))   ref_path = argv[++i];
        else if (ARG("--timings"))     timings_path = argv[++i];
        else if (ARG("--threads-csv")) threads_path = argv[++i];
        else if (ARG("--git-sha"))     git_sha = argv[++i];
        else if (ARG("--note"))        user_note = argv[++i];
        else if (ARG("--job-id"))      job_id = atol(argv[++i]);
        else if (ARG("--platform"))    platform = argv[++i];
        else if (ARG("--row-order"))   row_order = argv[++i];
        else if (ARG("--whitelist"))   whitelist = argv[++i];
        else if (ARG("--stage-mode"))  stage_mode = argv[++i];
        else if (ARG("--timed-region"))timed_region = argv[++i];
        else if (ARG("--partitioner")) snprintf(part_arg, sizeof part_arg, "%s", argv[++i]);
        else if (ARG("--schedule"))  { if (sched_parse(argv[++i], &skind)) die("bad --schedule"); }
        else if (ARG("--chunk"))       chunk = atoll(argv[++i]);
        else if (ARG("--tile"))        C = atoll(argv[++i]);
        else if (ARG("--repeat"))      K = atoll(argv[++i]);
        else if (ARG("--reps"))        reps = atoi(argv[++i]);
        else if (ARG("--warmup-s"))    warmup_s = atof(argv[++i]);
        else if (ARG("--threads"))     nthreads = atoi(argv[++i]);
        else if (!strcmp(a, "--no-interleave")) { interleave = 0; stage_mode = "serial"; }
        else if (!strcmp(a, "--smt"))  smt = 1;
        else if (!strcmp(a, "--no-papi")) use_papi = 0;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
        #undef ARG
    }
    if (!csr_dir || !lookup_path) die("--csr DIR --lookup FILE required");
    if (!variant) { const char *b = strrchr(csr_dir, '/'); variant = b ? b + 1 : csr_dir; }
    if (nthreads <= 0) nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);

    csr_t csr;
    if (csr_open(&csr, csr_dir, interleave) != 0) return 1;
    if (interleave && !csr.mbind_ok) stage_mode = "serial";
    else if (interleave) stage_mode = "mbind";

    const int64_t n_stays = csr.n_stays;
    lookup_t L = load_lookup(lookup_path);

    if (!strcmp(whitelist, "none")) {
        for (int i = 0; i < MAX_DENSE; i++) L.slot[i] = -1;
    } else if (!strcmp(whitelist, "all")) {
        for (int i = 0; i < MAX_DENSE; i++)
            L.slot[i] = (int16_t)(i % (L.n_slots > 0 ? L.n_slots : 1));
    } else if (strcmp(whitelist, "natural")) {
        die("--whitelist must be natural|none|all");
    }
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
        if (!v) die("malloc sort scratch");
        for (int64_t i = 0; i < n_stays; i++) { v[i].c = n_rec[i]; v[i].i = i; }
        qsort(v, (size_t)n_stays, sizeof *v,
              !strcmp(row_order, "sorted_desc") ? pair_desc : pair_asc);
        for (int64_t i = 0; i < n_stays; i++) stay_order[i] = v[i].i;
        free(v);
    } else die("--row-order must be canonical|sorted_desc|sorted_asc");

    /* cost in POSITION space, which is what the partitioner cuts */
    double *cost_pos = malloc((size_t)n_stays * sizeof *cost_pos);
    if (!cost_pos) die("malloc cost_pos");
    for (int64_t t = 0; t < n_stays; t++) cost_pos[t] = work[stay_order[t]];

    part_kind_t pk;
    const char *rec_part;
    switch (skind) {
    case SCH_PRECOMPUTED:
        if (part_parse(part_arg, &pk)) die("bad --partitioner");
        if (pk == PART_NONE) die("schedule precomputed requires a partitioner");
        rec_part = part_name(pk); break;
    case SCH_STATIC:         pk = PART_BLOCK;        rec_part = "block";        break;
    case SCH_STATIC_CHUNKED: pk = PART_BLOCK_CYCLIC; rec_part = "block_cyclic"; break;
    default:                 pk = PART_NONE;         rec_part = "none";         break;
    }
    if (skind == SCH_STATIC_CHUNKED || skind == SCH_DYNAMIC || skind == SCH_GUIDED) {
        if (chunk < 1) chunk = 64;
    } else chunk = -1;

    partition_t pt;
    const double *cut_key = cost_pos;
    double *nrec_pos = NULL;
    if (pk == PART_NZBALANCED) {
        nrec_pos = malloc((size_t)n_stays * sizeof *nrec_pos);
        if (!nrec_pos) die("malloc nrec_pos");
        for (int64_t t = 0; t < n_stays; t++) nrec_pos[t] = n_rec[stay_order[t]];
        cut_key = nrec_pos;
    }
    if (partition_build(&pt, pk, nthreads, cut_key, n_stays,
                        chunk > 0 ? chunk : 1) != 0) die("partition_build");
    const double pred_eff = partition_efficiency(&pt, cost_pos);

    /* walk[t] = stay index processed at position t; woff = per-thread range */
    int64_t *walk = malloc((size_t)n_stays * sizeof *walk);
    if (!walk) die("malloc walk");
    for (int64_t t = 0; t < n_stays; t++)
        walk[t] = stay_order[pt.order ? pt.order[t] : t];

   
    if (C < 0) C = 0;
    if (C > n_stays) C = n_stays;
    const int64_t Cbuf = (C > 0) ? C : 1;

    /*  buffers  */
    double *out   = aligned_alloc(64, (size_t)n_stays * F * sizeof(double));
    double *tiles = aligned_alloc(64, (size_t)nthreads * Cbuf * F * sizeof(double));
    acc_t  *accs  = aligned_alloc(64, (size_t)nthreads * L.n_slots * sizeof(acc_t));
    if (!out || !tiles || !accs) die("aligned_alloc");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        memset(tiles + (size_t)tid * Cbuf * F, 0, (size_t)Cbuf * F * sizeof(double));
        memset(accs  + (size_t)tid * L.n_slots, 0, (size_t)L.n_slots * sizeof(acc_t));
        for (int64_t t = pt.offs[tid]; t < pt.offs[tid + 1]; t++)
            memset(out + walk[t] * F, 0, (size_t)F * sizeof(double));
    }

    double *ref = NULL;
    if (ref_path && !strcmp(whitelist, "natural")) {
        FILE *fr = fopen(ref_path, "rb");
        if (!fr) { perror(ref_path); die("open --reference"); }
        ref = malloc((size_t)n_stays * F * sizeof(double));
        if (!ref) die("malloc reference");
        if (fread(ref, sizeof(double), (size_t)n_stays * F, fr)
            != (size_t)(n_stays * F)) die("short read on --reference");
        fclose(fr);
    }

    int papi_ok = use_papi ? pw_init() : -1;
    fprintf(stderr, "%s%s\n", pw_status(),
            (papi_ok == 0) ? "" : "  -- counters will be -1");
    if (papi_ok == 0) fprintf(stderr, "PAPI events: %s\n", pw_events());
    long (*th_papi)[PW_NEVENTS] = calloc((size_t)nthreads, sizeof *th_papi);

    double  *th_busy = calloc((size_t)nthreads, sizeof *th_busy);
    double  *th_ghz  = calloc((size_t)nthreads, sizeof *th_ghz);
    int     *th_cpu0 = calloc((size_t)nthreads, sizeof *th_cpu0);
    int64_t *th_rec  = calloc((size_t)nthreads, sizeof *th_rec);
    int64_t *th_sta  = calloc((size_t)nthreads, sizeof *th_sta);
    int     *th_cpu  = calloc((size_t)nthreads, sizeof *th_cpu);

    fprintf(stderr,
        "csr %s  variant %s  n_stays %" PRId64 "  n_records %" PRId64 "\n"
        "slots %d  F %" PRId64 "  hit_rate %.4f (%s)\n"
        "threads %d  partitioner %s  schedule %s  chunk %" PRId64 "\n"
        "row_order %s  tile %" PRId64 "%s  repeat %" PRId64 "  reps %d\n"
        "mempolicy %s (mbind_ok %d)  predicted work efficiency %.4f\n\n",
        csr_dir, variant, n_stays, csr.n_records, L.n_slots, F, hit_rate,
        whitelist, nthreads, rec_part, SCHED_NAMES[skind], chunk,
        row_order, C, C ? "" : " (direct write)", K, reps, csr.mempolicy, csr.mbind_ok, pred_eff);

    if (skind == SCH_STATIC)         omp_set_schedule(omp_sched_static,  0);
    else if (skind == SCH_STATIC_CHUNKED) omp_set_schedule(omp_sched_static,  (int)chunk);
    else if (skind == SCH_DYNAMIC)   omp_set_schedule(omp_sched_dynamic, (int)chunk);
    else if (skind == SCH_GUIDED)    omp_set_schedule(omp_sched_guided,  (int)chunk);

    FILE *ft = timing_csv_open(timings_path);
    FILE *fh = thread_csv_open(threads_path);
    if (!ft || !fh) die("open output CSV");

    if (!strcmp(timed_region, "end_to_end")) K = 1;


    if (warmup_s > 0.0) {
        double w0 = now_s();
        int passes = 0;
        while (now_s() - w0 < warmup_s) {
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                acc_t  *acc  = accs  + (size_t)tid * L.n_slots;
                double *tile = tiles + (size_t)tid * Cbuf * F;
                #pragma omp for schedule(static)
                for (int64_t t = 0; t < n_stays; t++) {
                    kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value,
                                walk[t], &L, acc, tile);
                    memcpy(out + walk[t] * F, tile, (size_t)F * sizeof(double));
                }
            }
            passes++;
        }
        fprintf(stderr, "warm-up: %d untimed passes in %.2f s\n",
                passes, now_s() - w0);
    }

    /*  repetitions  */
    for (int rep = 0; rep < reps; rep++) {
        for (int j = 0; j < nthreads; j++) {
            th_busy[j] = 0.0; th_rec[j] = th_sta[j] = 0; th_ghz[j] = -1.0;
        }
        double t0 = now_s();

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            acc_t  *acc  = accs  + (size_t)tid * L.n_slots;
            double *tile = tiles + (size_t)tid * Cbuf * F;
            double busy = 0.0, ghz_sum_t = 0.0;
            int ghz_n_t = 0;
            int64_t nrec = 0, nsta = 0;
            th_cpu0[tid] = sched_getcpu();
            for (int e = 0; e < PW_NEVENTS; e++) th_papi[tid][e] = -1;
            if (papi_ok == 0) pw_thread_start();

            #pragma omp barrier
            for (int64_t it = 0; it < K; it++) {
                const double a = omp_get_wtime();

                if (skind == SCH_PRECOMPUTED || skind == SCH_STATIC
                    || skind == SCH_STATIC_CHUNKED) {
              
                    const int64_t lo = pt.offs[tid], hi = pt.offs[tid + 1];
                    if (C == 0) {
                        for (int64_t t = lo; t < hi; t++)
                            kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value,
                                        walk[t], &L, acc, out + walk[t] * F);
                    } else {
                        for (int64_t b = lo; b < hi; b += C) {
                            const int64_t e = (b + C < hi) ? b + C : hi;
                            for (int64_t t = b; t < e; t++)
                                kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value,
                                            walk[t], &L, acc, tile + (t - b) * F);
                            for (int64_t t = b; t < e; t++)
                                memcpy(out + walk[t] * F, tile + (t - b) * F,
                                       (size_t)F * sizeof(double));
                        }
                    }
                    if (it == 0) {
                        nsta = hi - lo;
                        for (int64_t t = lo; t < hi; t++)
                            nrec += csr.offsets[walk[t] + 1] - csr.offsets[walk[t]];
                    }
                } else {
             
                    #pragma omp for schedule(runtime) nowait
                    for (int64_t t = 0; t < n_stays; t++) {
                        if (C == 0) {
                            kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value,
                                        walk[t], &L, acc, out + walk[t] * F);
                        } else {
                            kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value,
                                        walk[t], &L, acc, tile);
                            memcpy(out + walk[t] * F, tile,
                                   (size_t)F * sizeof(double));
                        }
                        nsta++;
                        nrec += csr.offsets[walk[t] + 1] - csr.offsets[walk[t]];
                    }
                }

                busy += omp_get_wtime() - a;
          
                th_cpu[tid] = sched_getcpu();
                { double g = read_ghz_cpu(th_cpu[tid]);
                  if (g > 0) { ghz_sum_t += g; ghz_n_t++; } }
    
                #pragma omp barrier
            }

            if (papi_ok == 0) pw_thread_stop(th_papi[tid]);
            th_busy[tid] = busy;
            th_ghz[tid]  = ghz_n_t ? ghz_sum_t / ghz_n_t : -1.0;
            th_rec[tid]  = (skind == SCH_PRECOMPUTED || skind == SCH_STATIC
                            || skind == SCH_STATIC_CHUNKED) ? nrec : nrec / (K ? K : 1);
            th_sta[tid]  = (skind == SCH_PRECOMPUTED || skind == SCH_STATIC
                            || skind == SCH_STATIC_CHUNKED) ? nsta : nsta / (K ? K : 1);
        }

        const double wall = now_s() - t0;
        double ghz_sum = 0.0; int ghz_n = 0, migrated = 0;
        for (int j = 0; j < nthreads; j++) {
            if (th_ghz[j] > 0) { ghz_sum += th_ghz[j]; ghz_n++; }
            if (th_cpu[j] != th_cpu0[j]) migrated++;
        }
        const double ghz = ghz_n ? ghz_sum / ghz_n : -1.0;

        double max_t = 0, sum_t = 0, mean_r = 0;
        int64_t max_r = 0, sum_r = 0;
        for (int j = 0; j < nthreads; j++) {
            if (th_busy[j] > max_t) max_t = th_busy[j];
            sum_t += th_busy[j];
            if (th_rec[j] > max_r) max_r = th_rec[j];
            sum_r += th_rec[j];
        }
        mean_r = (double)sum_r / nthreads;

        int n_sock[2] = {0, 0}, smt_used = 0, topo_ok = 1, seen[2][256];
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 256; b++) seen[a][b] = 0;
        for (int j = 0; j < nthreads; j++) {
            int sk = read_topo(th_cpu[j], "physical_package_id");
            int cr = read_topo(th_cpu[j], "core_id");
            if (sk < 0 || sk > 1 || cr < 0 || cr >= 256) { topo_ok = 0; break; }
            n_sock[sk]++;
            if (seen[sk][cr]) smt_used = 1;
            seen[sk][cr] = 1;
        }

        const char *placement =
              !topo_ok    ? "unknown"
            : smt_used    ? "smt"
            : n_sock[1]==0 ? "socket0"
            : n_sock[0]==0 ? "socket1"
            :                "spread";

        const char *bitid = "skip";
        if (ref) bitid = memcmp(ref, out, (size_t)n_stays * F * sizeof(double))
                         ? "fail" : "pass";

        char rid[64];
        timing_make_run_id(rid, sizeof rid, job_id, 0, rep);

        timing_row_t r;
        timing_row_init(&r);
        r.run_id = rid; r.job_id = job_id; r.git_sha = git_sha;
        { static char host[128]; gethostname(host, sizeof host); r.node = host; }
        { static char ts[32]; time_t now = time(NULL);
          strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", localtime(&now)); r.timestamp = ts; }
        r.csr_variant = variant; r.window_W = window;
        r.n_stays = n_stays; r.n_records = csr.n_records;
        r.paradigm = "openmp"; r.platform = platform;
        r.partitioner = rec_part; r.schedule = SCHED_NAMES[skind];
        r.chunk_size = chunk; r.row_order = row_order;
        r.n_threads = nthreads; r.smt = (topo_ok && smt_used) ? 1 : smt;
        r.thread_placement = placement;
        r.stage_mode = stage_mode; r.mempolicy = csr.mempolicy;
        r.numa_balancing = -1;
        r.whitelist_frac = !strcmp(whitelist, "none") ? 0.0
                         : !strcmp(whitelist, "all")  ? 1.0 : hit_rate;
        r.k_iterations = K; r.timed_region = timed_region;
        r.repetition = rep; r.warmup = (rep == 0);
        r.wall_time_s = wall;
        r.throughput_rec_s = (double)csr.n_records * (double)K / wall;
        r.work_imbalance = mean_r > 0 ? (double)max_r / mean_r : 0.0;
        r.time_imbalance = sum_t > 0 ? max_t / (sum_t / nthreads) : 0.0;
        r.n_records_max_thread = max_r;
        r.n_records_mean_thread = mean_r;
        r.bitidentical = bitid;
        /* Sum over threads; -1 iff no thread produced a value, so a
         * partially-armed event is not silently reported as zero. */
        {
            long tot[PW_NEVENTS];
            for (int e = 0; e < PW_NEVENTS; e++) {
                long acc = 0; int any = 0;
                for (int j = 0; j < nthreads; j++)
                    if (th_papi[j][e] >= 0) { acc += th_papi[j][e]; any = 1; }
                tot[e] = any ? acc : -1;
            }
            r.l3_tcm  = tot[PW_L3_TCM];
            r.l3_tca  = tot[PW_L3_TCA];
            r.tot_cyc = tot[PW_TOT_CYC];
            r.tot_ins = tot[PW_TOT_INS];
        }
        r.achieved_ghz = ghz;
       { static char note[256];
          snprintf(note, sizeof note,
                   "tile=%" PRId64 ";pred_eff=%.4f;migrated=%d%s%s",
                   C, pred_eff, migrated,
                   user_note ? ";" : "", user_note ? user_note : "");
          r.notes = note; }
        if (!timing_csv_write(ft, &r)) die("invalid timing row");

        for (int j = 0; j < nthreads; j++) {
            int sk = read_topo(th_cpu[j], "physical_package_id");
            thread_row_t tr = { TIMING_SCHEMA_VERSION, rid, j, th_cpu[j],
                                sk >= 0 ? sk : (th_cpu[j] & 1),
                                th_sta[j], th_rec[j], th_busy[j] };
            thread_csv_write(fh, &tr);
        }

        fprintf(stderr, "rep %d%s  wall %.6f s  %.1f M rec/s  "
                        "work_imb %.4f  time_imb %.4f  %.2f GHz  %-7s  mig %d  "
                        "bitid %s\n",
                rep, rep ? "" : " (warmup)", wall,
                r.throughput_rec_s / 1e6, r.work_imbalance, r.time_imbalance,
                ghz, placement, migrated, bitid);
        if (r.tot_cyc > 0 && r.tot_ins > 0)
            fprintf(stderr, "        IPC %.3f", (double)r.tot_ins / r.tot_cyc);
        if (r.l3_tca > 0 && r.l3_tcm >= 0)
            fprintf(stderr, "   L3 miss %.1f%% (%ld/%ld)",
                    100.0 * r.l3_tcm / r.l3_tca, r.l3_tcm, r.l3_tca);
        if (r.tot_cyc > 0 || r.l3_tca > 0) fprintf(stderr, "\n");
        if (migrated)
            fprintf(stderr, "  ** %d thread(s) migrated cpu during this rep -- "
                            "first-touch locality is broken. Bind with "
                            "OMP_PROC_BIND / OMP_PLACES.\n", migrated);
        if (!strcmp(bitid, "fail"))
            fprintf(stderr, "  ** BIT-IDENTITY VIOLATED -- a race, or a build that "
                            "did not link the shared kernel_core.o\n");
    }

    fclose(ft); fclose(fh);

    if (out_path) {
        FILE *fo = fopen(out_path, "wb");
        if (!fo) { perror(out_path); die("open --out"); }
        if (fwrite(out, sizeof(double), (size_t)n_stays * F, fo)
            != (size_t)(n_stays * F)) die("short write");
        fclose(fo);
    }

    free(n_rec); free(work); free(cost_pos); free(nrec_pos); free(stay_order);
    free(walk); free(out); free(tiles); free(accs); free(ref);
    free(th_busy); free(th_rec); free(th_sta); free(th_cpu);
    free(th_ghz); free(th_cpu0); free(th_papi);
    partition_free(&pt);
    csr_close(&csr);
    return 0;
}
