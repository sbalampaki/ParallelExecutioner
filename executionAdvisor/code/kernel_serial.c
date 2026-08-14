

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "kernel_core.h"


static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void *map_file(const char *path, size_t *len_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); die("open"); }
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat");
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) die("mmap");
    close(fd);
    *len_out = (size_t)st.st_size;
    return p;
}


static void run_kernel(const int64_t *restrict offsets,
                       const int32_t *restrict itemid,
                       const float   *restrict t,
                       const float   *restrict value,
                       int64_t n_stays,
                       const lookup_t *L,
                       int64_t C,
                       double *restrict out,
                       double *restrict tile,
                       acc_t  *restrict acc)
{
    const int64_t F = (int64_t)L->n_slots * N_AGG;

    for (int64_t base = 0; base < n_stays; base += C) {
        const int64_t hi_stay = (base + C < n_stays) ? base + C : n_stays;

        for (int64_t i = base; i < hi_stay; i++) {
            kernel_stay(offsets, itemid, t, value, i, L, acc,
                        tile + (i - base) * F);
        }

        memcpy(out + base * F, tile, (size_t)(hi_stay - base) * F * sizeof(double));
    }
}

/* NaN-aware: under IEEE 754 NaN != NaN, so a plain memcmp-by-value would
 * report every missing cell as a mismatch (D23). */
static int64_t diff_count(const double *a, const double *b, int64_t n) {
    int64_t d = 0;
    for (int64_t i = 0; i < n; i++) {
        if (isnan(a[i]) && isnan(b[i])) continue;
        if (a[i] != b[i]) d++;
    }
    return d;
}

static double checksum(const double *a, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) if (!isnan(a[i])) s += a[i];
    return s;
}


int main(int argc, char **argv) {
    const char *csr = NULL, *lookup_path = NULL, *out_path = NULL;
    int64_t C = 4096, K = 1;
    int verify_tiles = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--csr")    && i + 1 < argc) csr = argv[++i];
        else if (!strcmp(argv[i], "--lookup") && i + 1 < argc) lookup_path = argv[++i];
        else if (!strcmp(argv[i], "--out")    && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--tile")   && i + 1 < argc) C = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) K = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--verify-tiles")) verify_tiles = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!csr || !lookup_path)
        die("usage: --csr DIR --lookup FILE [--out F] [--tile C] "
            "[--repeat K] [--verify-tiles]");

    char p[4096];
    size_t len_off, len_itm, len_t, len_val;
    snprintf(p, sizeof p, "%s/offsets.i64", csr);
    const int64_t *offsets = map_file(p, &len_off);
    snprintf(p, sizeof p, "%s/itemid.i32", csr);
    const int32_t *itemid  = map_file(p, &len_itm);
    snprintf(p, sizeof p, "%s/t.f32", csr);
    const float   *t       = map_file(p, &len_t);
    snprintf(p, sizeof p, "%s/value.f32", csr);
    const float   *value   = map_file(p, &len_val);

    const int64_t n_stays   = (int64_t)(len_off / sizeof(int64_t)) - 1;
    const int64_t n_records = offsets[n_stays];

    if (len_itm != (size_t)n_records * 4 ||
        len_t   != (size_t)n_records * 4 ||
        len_val != (size_t)n_records * 4)
        die("CSR binary sizes disagree with offsets[n_stays]");

    lookup_t L = load_lookup(lookup_path);
    const int64_t F = (int64_t)L.n_slots * N_AGG;

    if (C < 1) C = 1;
    if (C > n_stays) C = n_stays;

    fprintf(stderr,
        "csr        : %s\n"
        "n_stays    : %" PRId64 "\n"
        "n_records  : %" PRId64 "\n"
        "K (dense)  : %d\n"
        "slots      : %d   -> F = %" PRId64 "\n"
        "tile C     : %" PRId64 "   (tile buffer %.2f MB)\n"
        "repeat K   : %" PRId64 "\n",
        csr, n_stays, n_records, L.k, L.n_slots, F, C,
        (double)C * F * 8 / 1e6, K);

    double *out  = aligned_alloc(64, (size_t)n_stays * F * sizeof(double));
    double *tile = aligned_alloc(64, (size_t)C * F * sizeof(double));
    acc_t  *acc  = aligned_alloc(64, (size_t)L.n_slots * sizeof(acc_t));
    if (!out || !tile || !acc) die("aligned_alloc");


    if (verify_tiles) {
        const int64_t tiles[3] = { 1, 4096, n_stays };
        double *ref = malloc((size_t)n_stays * F * sizeof(double));
        if (!ref) die("malloc ref");
        int bad = 0;
        for (int v = 0; v < 3; v++) {
            int64_t c = tiles[v] > n_stays ? n_stays : tiles[v];
            double *tl = aligned_alloc(64, (size_t)c * F * sizeof(double));
            if (!tl) die("aligned_alloc verify tile");
            run_kernel(offsets, itemid, t, value, n_stays, &L, c, out, tl, acc);
            if (v == 0) {
                memcpy(ref, out, (size_t)n_stays * F * sizeof(double));
                fprintf(stderr, "verify: tile=%-8" PRId64 " reference\n", c);
            } else {
                int64_t d = diff_count(ref, out, n_stays * F);
                fprintf(stderr, "verify: tile=%-8" PRId64 " diffs=%" PRId64 " %s\n",
                        c, d, d ? "FAIL" : "OK");
                bad |= (d != 0);
            }
            free(tl);
        }
        free(ref);
        fprintf(stderr, "verify: %s\n", bad ? "TILE INVARIANCE VIOLATED" : "all tile sizes bit-identical");
        if (bad) return 1;
    }


    volatile double sink = 0.0;
    double t0 = now_s();
    for (int64_t k = 0; k < K; k++) {
        run_kernel(offsets, itemid, t, value, n_stays, &L, C, out, tile, acc);
        sink += out[0];             
    }
    double el = now_s() - t0;
    (void)sink;

    fprintf(stderr,
        "\nelapsed    : %.6f s  (%" PRId64 " iterations)\n"
        "per-iter   : %.6f s\n"
        "throughput : %.3f M records/s\n"
        "checksum   : %.10e\n",
        el, K, el / (double)K,
        (double)n_records * (double)K / el / 1e6,
        checksum(out, n_stays * F));

  
    printf("serial,none,%" PRId64 ",1,1,0,1,broadwell,none,scalar,fp64,%s,1,"
           "%.6f,1.0,,,,,tile=%" PRId64 ";K=%" PRId64 "\n",
           C, csr, el / (double)K, C, K);

    if (out_path) {
        FILE *fo = fopen(out_path, "wb");
        if (!fo) { perror(out_path); die("open --out"); }
        if (fwrite(out, sizeof(double), (size_t)n_stays * F, fo)
            != (size_t)(n_stays * F)) die("short write");
        fclose(fo);
        fprintf(stderr, "wrote      : %s  (%" PRId64 " x %" PRId64 " float64, %.1f MB)\n",
                out_path, n_stays, F, (double)n_stays * F * 8 / 1e6);
    }
    return 0;
}
