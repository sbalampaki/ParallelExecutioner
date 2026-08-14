
#define _GNU_SOURCE
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define F_FEATURES 160          /* D20: 20 variables x 8 aggregates */

static void *map_ro(const char *path, size_t *len_out)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); exit(1); }
    size_t len = (size_t)st.st_size;
    void *p = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);
    *len_out = len;
    return p;
}

int main(int argc, char **argv)
{
    const char *csr_dir  = NULL;
    const char *tag      = "run";
    int         touch_all = 1;
    int         reps     = 6;
    int         write_out = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csr")   && i + 1 < argc) csr_dir = argv[++i];
        else if (!strcmp(argv[i], "--tag")   && i + 1 < argc) tag = argv[++i];
        else if (!strcmp(argv[i], "--reps")  && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--touch") && i + 1 < argc) touch_all = !strcmp(argv[++i], "all");
        else if (!strcmp(argv[i], "--write-out")) write_out = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!csr_dir) { fprintf(stderr, "--csr DIR required\n"); return 2; }

    char p_off[4096], p_item[4096], p_t[4096], p_val[4096];
    snprintf(p_off,  sizeof p_off,  "%s/offsets.i64", csr_dir);
    snprintf(p_item, sizeof p_item, "%s/itemid.i32",  csr_dir);
    snprintf(p_t,    sizeof p_t,    "%s/t.f32",       csr_dir);
    snprintf(p_val,  sizeof p_val,  "%s/value.f32",   csr_dir);

    size_t l_off, l_item, l_t, l_val;
    const int64_t  *offsets = map_ro(p_off,  &l_off);
    const uint32_t *itemid  = map_ro(p_item, &l_item);
  
    const uint32_t *tbits   = map_ro(p_t,    &l_t);
    const uint32_t *vbits   = map_ro(p_val,  &l_val);

    int64_t n_stays   = (int64_t)(l_off / 8) - 1;
    int64_t n_records = offsets[n_stays];

    if ((size_t)n_records * 4 != l_item || l_item != l_t || l_t != l_val) {
        fprintf(stderr,
                "CSR size mismatch: n_records=%lld itemid=%zu t=%zu value=%zu\n",
                (long long)n_records, l_item, l_t, l_val);
        return 1;
    }

    int n_threads = omp_get_max_threads();

    double  *out = NULL;
    if (write_out) {
        out = aligned_alloc(64, (size_t)n_stays * F_FEATURES * sizeof(double));
        if (!out) { fprintf(stderr, "OOM on output matrix\n"); return 1; }
     
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_stays; i++)
            memset(out + (size_t)i * F_FEATURES, 0, F_FEATURES * sizeof(double));
    }

    double  *th_secs = calloc((size_t)n_threads, sizeof(double));
    int64_t *th_recs = calloc((size_t)n_threads, sizeof(int64_t));
    int64_t *th_stay = calloc((size_t)n_threads, sizeof(int64_t));
    int     *th_cpu  = calloc((size_t)n_threads, sizeof(int));

    const double bytes_per_rec = touch_all ? 12.0 : 4.0;
    const double bytes_total   = bytes_per_rec * (double)n_records
                               + 8.0 * (double)(n_stays + 1)
                               + (write_out ? 8.0 * F_FEATURES * (double)n_stays : 0.0);

    fprintf(stderr,
            "csr=%s n_stays=%lld n_records=%lld threads=%d touch=%s write_out=%d\n",
            csr_dir, (long long)n_stays, (long long)n_records, n_threads,
            touch_all ? "all" : "itemid", write_out);

    printf("RUN,tag,threads,touch,write_out,rep,warmup,wall_s,records,"
           "records_per_s,gb_s,work_imb,time_imb\n");
    printf("THR,tag,threads,rep,tid,cpu,socket,stays,records,secs,gb_s\n");

    uint64_t global_sink = 0;

    for (int rep = 0; rep < reps; rep++) {
        double t0 = omp_get_wtime();

        #pragma omp parallel reduction(+:global_sink)
        {
            int tid = omp_get_thread_num();
            uint32_t a0 = 0, a1 = 0, a2 = 0;
            int64_t  nrec = 0, nstay = 0;

            th_cpu[tid] = sched_getcpu();

            #pragma omp barrier            /* align thread start times */
            double s = omp_get_wtime();

            if (touch_all) {
                #pragma omp for schedule(static) nowait
                for (int64_t i = 0; i < n_stays; i++) {
                    int64_t lo = offsets[i], hi = offsets[i + 1];
                    for (int64_t j = lo; j < hi; j++) {
                        a0 += itemid[j];
                        a1 ^= tbits[j];
                        a2 ^= vbits[j];
                    }
                    nrec += hi - lo; nstay++;
                    if (out)
                        for (int f = 0; f < F_FEATURES; f++)
                            out[(size_t)i * F_FEATURES + f] = (double)(a0 + (uint32_t)f);
                }
            } else {
                #pragma omp for schedule(static) nowait
                for (int64_t i = 0; i < n_stays; i++) {
                    int64_t lo = offsets[i], hi = offsets[i + 1];
                    for (int64_t j = lo; j < hi; j++)
                        a0 += itemid[j];
                    nrec += hi - lo; nstay++;
                    if (out)
                        for (int f = 0; f < F_FEATURES; f++)
                            out[(size_t)i * F_FEATURES + f] = (double)(a0 + (uint32_t)f);
                }
            }

            double e = omp_get_wtime();
            th_secs[tid] = e - s;
            th_recs[tid] = nrec;
            th_stay[tid] = nstay;
            global_sink += (uint64_t)a0 + (uint64_t)a1 + (uint64_t)a2;
        }

        double wall = omp_get_wtime() - t0;

        double max_s = 0.0, sum_s = 0.0;
        int64_t max_r = 0, sum_r = 0;
        for (int i = 0; i < n_threads; i++) {
            if (th_secs[i] > max_s) max_s = th_secs[i];
            sum_s += th_secs[i];
            if (th_recs[i] > max_r) max_r = th_recs[i];
            sum_r += th_recs[i];
        }
        double mean_s = sum_s / n_threads;
        double mean_r = (double)sum_r / n_threads;
        double time_imb = mean_s > 0 ? max_s / mean_s : 0.0;
        double work_imb = mean_r > 0 ? (double)max_r / mean_r : 0.0;

        printf("RUN,%s,%d,%s,%d,%d,%d,%.6f,%lld,%.3e,%.3f,%.4f,%.4f\n",
               tag, n_threads, touch_all ? "all" : "itemid", write_out, rep,
               rep == 0 ? 1 : 0, wall, (long long)n_records,
               (double)n_records / wall, bytes_total / wall / 1e9,
               work_imb, time_imb);

        for (int i = 0; i < n_threads; i++) {
            double thr_bytes = bytes_per_rec * (double)th_recs[i]
                             + (write_out ? 8.0 * F_FEATURES * (double)th_stay[i] : 0.0);
            printf("THR,%s,%d,%d,%d,%d,%d,%lld,%lld,%.6f,%.3f\n",
                   tag, n_threads, rep, i, th_cpu[i], th_cpu[i] & 1,
                   (long long)th_stay[i], (long long)th_recs[i], th_secs[i],
                   th_secs[i] > 0 ? thr_bytes / th_secs[i] / 1e9 : 0.0);
        }
        fflush(stdout);
    }

    fprintf(stderr, "sink=%llu\n", (unsigned long long)global_sink);
    free(th_secs); free(th_recs); free(th_stay); free(th_cpu); free(out);
    return 0;
}
