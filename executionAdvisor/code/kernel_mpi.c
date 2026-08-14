

#define _GNU_SOURCE

#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
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
#include "kernel_core.h"

#define RHO_REF 5.31
#define MAXT 128

static int RANK = 0, NRANK = 1;

static void die(const char *m) {
    fprintf(stderr, "[rank %d] FATAL: %s\n", RANK, m);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

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
    int prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NRANK);
    if (prov < MPI_THREAD_FUNNELED && RANK == 0)
        fprintf(stderr, "warn: MPI thread level %d < FUNNELED\n", prov);

    const char *csr_dir = NULL, *lookup_path = NULL, *ref_path = NULL;
    const char *variant = NULL, *window = "", *out_path = NULL;
    const char *timings_path = "timings.csv", *threads_path = "threads.csv";
    const char *git_sha = "unknown", *row_order = "canonical";
    const char *whitelist = "natural", *platform = "broadwell";
    const char *rank_cpus = NULL, *csr_suffix = NULL, *user_note = NULL;
    char part_arg[32] = "block";
    int64_t C = 0, K = 1;
    int nthreads = 0, reps = 6, interleave = 1, n_nodes = 1;
    double warmup_s = 2.0;
    long job_id = -1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(n) (!strcmp(a, n) && i + 1 < argc)
        if      (ARG("--csr"))          csr_dir = argv[++i];

        else if (ARG("--csr-per-rank")) csr_suffix = argv[++i];
        else if (ARG("--note"))         user_note = argv[++i];
        else if (ARG("--lookup"))       lookup_path = argv[++i];
        else if (ARG("--variant"))      variant = argv[++i];
        else if (ARG("--window"))       window = argv[++i];
        else if (ARG("--reference"))    ref_path = argv[++i];
        else if (ARG("--out"))          out_path = argv[++i];
        else if (ARG("--timings"))      timings_path = argv[++i];
        else if (ARG("--threads-csv"))  threads_path = argv[++i];
        else if (ARG("--git-sha"))      git_sha = argv[++i];
        else if (ARG("--job-id"))       job_id = atol(argv[++i]);
        else if (ARG("--platform"))     platform = argv[++i];
        else if (ARG("--row-order"))    row_order = argv[++i];
        else if (ARG("--whitelist"))    whitelist = argv[++i];
        else if (ARG("--partitioner"))  snprintf(part_arg, sizeof part_arg, "%s", argv[++i]);
        else if (ARG("--rank-cpus"))    rank_cpus = argv[++i];
        else if (ARG("--tile"))         C = atoll(argv[++i]);
        else if (ARG("--repeat"))       K = atoll(argv[++i]);
        else if (ARG("--reps"))         reps = atoi(argv[++i]);
        else if (ARG("--threads"))      nthreads = atoi(argv[++i]);
        else if (ARG("--nodes"))        n_nodes = atoi(argv[++i]);
        else if (ARG("--warmup-s"))     warmup_s = atof(argv[++i]);
        else if (!strcmp(a, "--no-interleave")) interleave = 0;
        else { if (RANK == 0) fprintf(stderr, "unknown arg: %s\n", a); MPI_Finalize(); return 2; }
        #undef ARG
    }
    if (!csr_dir || !lookup_path) die("--csr DIR --lookup FILE required");
    if (!variant) { const char *b = strrchr(csr_dir, '/'); variant = b ? b + 1 : csr_dir; }

    int cpus[MAXT], ncpu = 0;
    if (rank_cpus) {
        char buf[4096]; snprintf(buf, sizeof buf, "%s", rank_cpus);
        char *save = NULL, *grp = strtok_r(buf, ";", &save);
        for (int r = 0; grp && r <= RANK; r++) {
            if (r == RANK) {
                char g[1024]; snprintf(g, sizeof g, "%s", grp);
                for (char *tk = strtok(g, ","); tk && ncpu < MAXT; tk = strtok(NULL, ","))
                    cpus[ncpu++] = atoi(tk);
            }
            grp = strtok_r(NULL, ";", &save);
        }
        if (ncpu == 0) die("--rank-cpus has no group for this rank");
        if (nthreads <= 0) nthreads = ncpu;
        if (nthreads != ncpu) die("--threads must match this rank's cpu group");
    } else {
        if (nthreads <= 0) nthreads = omp_get_max_threads();
        for (int i = 0; i < nthreads; i++) cpus[i] = -1;
        if (RANK == 0)
            fprintf(stderr, "warn: no --rank-cpus; threads unbound and thread "
                            "ORDER is uncontrolled (D44: worth 21%% per cycle)\n");
    }
    omp_set_num_threads(nthreads);

    /* ---- CSR  */
    char mydir[4096];
    if (csr_suffix) snprintf(mydir, sizeof mydir, "%s%s%d", csr_dir, csr_suffix, RANK);
    else            snprintf(mydir, sizeof mydir, "%s", csr_dir);
    csr_t csr;
    if (csr_open(&csr, mydir, interleave) != 0) die("csr_open");

    lookup_t L = load_lookup(lookup_path);
    if (!strcmp(whitelist, "none")) {
        for (int i = 0; i < MAX_DENSE; i++) L.slot[i] = -1;
    } else if (!strcmp(whitelist, "all")) {
        for (int i = 0; i < MAX_DENSE; i++)
            L.slot[i] = (int16_t)(i % (L.n_slots > 0 ? L.n_slots : 1));
    } else if (strcmp(whitelist, "natural")) die("--whitelist natural|none|all");

    const int64_t n_stays = csr.n_stays;
    const int64_t F = (int64_t)L.n_slots * N_AGG;

    /*  cost vectors and row order (identical to the OpenMP driver)  */
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

    /* ---- two-level partition: ranks first, then threads within a rank  */
    part_kind_t pk;
    if (part_parse(part_arg, &pk)) die("bad --partitioner");
    partition_t rp;
    const double *rkey = (pk == PART_NZBALANCED) ? nrec_pos : cost_pos;
    if (partition_build(&rp, pk, NRANK, rkey, n_stays, 1) != 0) die("rank partition");
    const int64_t my_lo = rp.offs[RANK], my_hi = rp.offs[RANK + 1];
    const int64_t my_n  = my_hi - my_lo;

    partition_t tp;
    if (partition_build(&tp, pk, nthreads, rkey + my_lo, my_n, 1) != 0)
        die("thread partition");
    const double pred_eff = partition_efficiency(&tp, cost_pos + my_lo);

    int64_t *walk = malloc((size_t)n_stays * sizeof *walk);
    if (!walk) die("malloc walk");
    for (int64_t t = 0; t < n_stays; t++)
        walk[t] = stay_order[rp.order ? rp.order[t] : t];

    if (C < 0) C = 0;
    const int64_t Cbuf = C > 0 ? C : 1;

   
    double *myout = aligned_alloc(64, (size_t)my_n * F * sizeof(double));
    double *tiles = aligned_alloc(64, (size_t)nthreads * Cbuf * F * sizeof(double));
    acc_t  *accs  = aligned_alloc(64, (size_t)nthreads * L.n_slots * sizeof(acc_t));
    if (!myout || !tiles || !accs) die("aligned_alloc");

    double  *th_busy = calloc((size_t)nthreads, sizeof *th_busy);
    double  *th_ghz  = calloc((size_t)nthreads, sizeof *th_ghz);
    int64_t *th_rec  = calloc((size_t)nthreads, sizeof *th_rec);
    int64_t *th_sta  = calloc((size_t)nthreads, sizeof *th_sta);
    int     *th_cpu  = calloc((size_t)nthreads, sizeof *th_cpu);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (cpus[tid] >= 0) {
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[tid], &s);
            if (sched_setaffinity(0, sizeof s, &s) != 0)
                fprintf(stderr, "[rank %d] warn: thread %d bind to cpu %d failed\n",
                        RANK, tid, cpus[tid]);
        }
        memset(tiles + (size_t)tid * Cbuf * F, 0, (size_t)Cbuf * F * sizeof(double));
        memset(accs  + (size_t)tid * L.n_slots, 0, (size_t)L.n_slots * sizeof(acc_t));
        for (int64_t t = tp.offs[tid]; t < tp.offs[tid + 1]; t++)
            memset(myout + t * F, 0, (size_t)F * sizeof(double));
    }

    /* ---- gather layout  */
    int *rcount = NULL, *rdispl = NULL;
    double *full = NULL, *ref = NULL;
    if (RANK == 0) {
        rcount = malloc((size_t)NRANK * sizeof *rcount);
        rdispl = malloc((size_t)NRANK * sizeof *rdispl);
        for (int r = 0; r < NRANK; r++) {
            rcount[r] = (int)((rp.offs[r + 1] - rp.offs[r]) * F);
            rdispl[r] = (int)(rp.offs[r] * F);
        }
        full = aligned_alloc(64, (size_t)n_stays * F * sizeof(double));
        if (!full) die("alloc gather buffer");
        if (ref_path && !strcmp(whitelist, "natural")) {
            FILE *fr = fopen(ref_path, "rb");
            if (!fr) die("open --reference");
            ref = malloc((size_t)n_stays * F * sizeof(double));
            if (!ref) die("malloc reference");
            if (fread(ref, sizeof(double), (size_t)n_stays * F, fr)
                != (size_t)(n_stays * F)) die("short read on --reference");
            fclose(fr);
        }
    }

    if (RANK == 0)
        fprintf(stderr,
            "csr %s  variant %s  n_stays %" PRId64 "  n_records %" PRId64 "\n"
            "ranks %d  threads/rank %d  nodes %d  partitioner %s\n"
            "slots %d  F %" PRId64 "  hit %.4f (%s)  tile %" PRId64 "\n"
            "row_order %s  repeat %" PRId64 "  reps %d  mempolicy %s\n\n",
            mydir, variant, n_stays, csr.n_records, NRANK, nthreads, n_nodes,
            part_name(pk), L.n_slots, F, hit_rate, whitelist, C,
            row_order, K, reps, csr.mempolicy);

    FILE *ft = NULL, *fh = NULL;
    if (RANK == 0) {
        ft = timing_csv_open(timings_path);
        fh = thread_csv_open(threads_path);
        if (!ft || !fh) die("open output CSV");
    }

    /* ---- one pass over this rank's slice  */
    #define PASS()                                                             \
        do {                                                                   \
            const int tid = omp_get_thread_num();                              \
            acc_t  *acc  = accs  + (size_t)tid * L.n_slots;                    \
            double *tile = tiles + (size_t)tid * Cbuf * F;                     \
            const int64_t lo = tp.offs[tid], hi = tp.offs[tid + 1];            \
            if (C == 0) {                                                      \
                for (int64_t t = lo; t < hi; t++)                              \
                    kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value,     \
                                walk[my_lo + t], &L, acc, myout + t * F);      \
            } else {                                                           \
                for (int64_t b = lo; b < hi; b += C) {                         \
                    const int64_t e = (b + C < hi) ? b + C : hi;               \
                    for (int64_t t = b; t < e; t++)                            \
                        kernel_stay(csr.offsets, csr.itemid, csr.t, csr.value, \
                                    walk[my_lo + t], &L, acc,                  \
                                    tile + (t - b) * F);                       \
                    for (int64_t t = b; t < e; t++)                            \
                        memcpy(myout + t * F, tile + (t - b) * F,              \
                               (size_t)F * sizeof(double));                    \
                }                                                              \
            }                                                                  \
        } while (0)

    if (warmup_s > 0.0) {
        double w0 = now_s(); int passes = 0;
        while (now_s() - w0 < warmup_s) {
            #pragma omp parallel
            { PASS(); }
            passes++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (RANK == 0)
            fprintf(stderr, "warm-up: >=%d untimed passes in %.2f s\n\n",
                    passes, now_s() - w0);
    }

    for (int rep = 0; rep < reps; rep++) {
        for (int j = 0; j < nthreads; j++) { th_busy[j] = 0; th_rec[j] = th_sta[j] = 0; th_ghz[j] = -1; }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = now_s();

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double busy = 0.0, gsum = 0.0; int gn = 0;
            int64_t nrec = 0, nsta = 0;
            #pragma omp barrier
            for (int64_t it = 0; it < K; it++) {
                const double a = omp_get_wtime();
                PASS();
                busy += omp_get_wtime() - a;
                th_cpu[tid] = sched_getcpu();
                { double g = read_ghz_cpu(th_cpu[tid]); if (g > 0) { gsum += g; gn++; } }
                #pragma omp barrier
            }
            if (1) {
                nsta = tp.offs[tid + 1] - tp.offs[tid];
                for (int64_t t = tp.offs[tid]; t < tp.offs[tid + 1]; t++) {
                    const int64_t s = walk[my_lo + t];
                    nrec += csr.offsets[s + 1] - csr.offsets[s];
                }
            }
            th_busy[tid] = busy; th_rec[tid] = nrec; th_sta[tid] = nsta;
            th_ghz[tid] = gn ? gsum / gn : -1.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double wall_kernel = now_s() - t0;

   
        const double g0 = now_s();
        MPI_Gatherv(myout, (int)(my_n * F), MPI_DOUBLE,
                    full, rcount, rdispl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        const double wall_e2e = wall_kernel + (now_s() - g0);

    
        double loc_max = 0, loc_sum = 0, loc_ghz = 0; int gn = 0;
        int64_t loc_maxr = 0, loc_sumr = 0;
        for (int j = 0; j < nthreads; j++) {
            if (th_busy[j] > loc_max) loc_max = th_busy[j];
            loc_sum += th_busy[j];
            if (th_rec[j] > loc_maxr) loc_maxr = th_rec[j];
            loc_sumr += th_rec[j];
            if (th_ghz[j] > 0) { loc_ghz += th_ghz[j]; gn++; }
        }
        double gmax = 0, gsum = 0, gghz = 0;
        long long gmaxr = 0, gsumr = 0; int ggn = 0;
        MPI_Reduce(&loc_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc_sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&loc_ghz, &gghz, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&gn, &ggn, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        { long long a = loc_maxr, b = loc_sumr;
          MPI_Reduce(&a, &gmaxr, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
          MPI_Reduce(&b, &gsumr, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); }

        if (RANK == 0) {
            const int totthr = NRANK * nthreads;
            const double mean_t = gsum / totthr;
            const double mean_r = (double)gsumr / totthr;
            const char *bitid = "skip";
            if (ref) bitid = memcmp(ref, full, (size_t)n_stays * F * sizeof(double))
                             ? "fail" : "pass";

            int n_sock[2] = {0,0}, smt = 0, topo_ok = 1, seen[2][256];
            memset(seen, 0, sizeof seen);
            for (int j = 0; j < nthreads; j++) {
                int sk = read_topo(th_cpu[j], "physical_package_id");
                int cr = read_topo(th_cpu[j], "core_id");
                if (sk < 0 || sk > 1 || cr < 0 || cr >= 256) { topo_ok = 0; break; }
                n_sock[sk]++; if (seen[sk][cr]) smt = 1; seen[sk][cr] = 1;
            }
            const char *placement = !topo_ok ? "unknown" : smt ? "smt"
                                  : n_sock[1] == 0 ? "socket0"
                                  : n_sock[0] == 0 ? "socket1" : "spread";

            for (int which = 0; which < 2; which++) {
                const double wall = which ? wall_e2e : wall_kernel;
                char rid[64];
                timing_make_run_id(rid, sizeof rid, job_id, which, rep);
                timing_row_t r; timing_row_init(&r);
                r.run_id = rid; r.job_id = job_id; r.git_sha = git_sha;
                { static char h[128]; gethostname(h, sizeof h); r.node = h; }
                { static char ts[32]; time_t nw = time(NULL);
                  strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", localtime(&nw));
                  r.timestamp = ts; }
                r.csr_variant = variant; r.window_W = window;
                r.n_stays = n_stays; r.n_records = csr.n_records;
                r.paradigm = (nthreads > 1) ? "hybrid" : "mpi";
                r.platform = platform;
                r.partitioner = part_name(pk); r.schedule = "precomputed";
                r.chunk_size = -1; r.row_order = row_order;
                r.n_ranks = NRANK; r.n_threads = nthreads;
                r.smt = (topo_ok && smt); r.thread_placement = placement;
                r.n_nodes = n_nodes;
                r.stage_mode = (interleave && csr.mbind_ok) ? "mbind" : "serial";
                r.mempolicy = csr.mempolicy;
                r.whitelist_frac = !strcmp(whitelist, "none") ? 0.0
                                 : !strcmp(whitelist, "all")  ? 1.0 : hit_rate;
                r.k_iterations = K;
                r.timed_region = which ? "end_to_end" : "kernel";
                r.repetition = rep; r.warmup = (rep == 0);
                r.wall_time_s = wall;
                r.throughput_rec_s = (double)csr.n_records * (double)K / wall;
                r.work_imbalance = mean_r > 0 ? (double)gmaxr / mean_r : 0.0;
                r.time_imbalance = mean_t > 0 ? gmax / mean_t : 0.0;
                r.n_records_max_thread = (long)gmaxr;
                r.n_records_mean_thread = mean_r;
                r.bitidentical = which ? bitid : "skip";
                r.achieved_ghz = ggn ? gghz / ggn : -1.0;
                { static char note[256];
                  snprintf(note, sizeof note,
                           "tile=%" PRId64 ";pred_eff=%.4f;per_rank_csr=%d%s%s",
                           C, pred_eff, csr_suffix ? 1 : 0,
                           user_note ? ";" : "", user_note ? user_note : "");
                  r.notes = note; }
                if (!timing_csv_write(ft, &r)) die("invalid timing row");
                if (!which)
                    for (int j = 0; j < nthreads; j++) {
                        int sk = read_topo(th_cpu[j], "physical_package_id");
                        thread_row_t tr = { TIMING_SCHEMA_VERSION, rid, j,
                                            th_cpu[j], sk >= 0 ? sk : 0,
                                            th_sta[j], th_rec[j], th_busy[j] };
                        thread_csv_write(fh, &tr);
                    }
            }
            fprintf(stderr, "rep %d%s  kernel %.6f s  e2e %.6f s  %.1f M rec/s  "
                            "work_imb %.4f  time_imb %.4f  %.2f GHz  %-7s  bitid %s\n",
                    rep, rep ? "" : " (warmup)", wall_kernel, wall_e2e,
                    (double)csr.n_records * K / wall_kernel / 1e6,
                    mean_r > 0 ? (double)gmaxr / mean_r : 0.0,
                    mean_t > 0 ? gmax / mean_t : 0.0,
                    ggn ? gghz / ggn : -1.0, placement, bitid);
        }
    }
    #undef PASS

    if (RANK == 0) {
        if (ft) fclose(ft);
        if (fh) fclose(fh);
        if (out_path) {
            FILE *fo = fopen(out_path, "wb");
            if (!fo) die("open --out");
            fwrite(full, sizeof(double), (size_t)n_stays * F, fo);
            fclose(fo);
        }
    }

    free(n_rec); free(work); free(cost_pos); free(nrec_pos); free(stay_order);
    free(walk); free(myout); free(tiles); free(accs);
    free(th_busy); free(th_ghz); free(th_rec); free(th_sta); free(th_cpu);
    free(rcount); free(rdispl); free(full); free(ref);
    partition_free(&rp); partition_free(&tp);
    csr_close(&csr);
    MPI_Finalize();
    return 0;
}
