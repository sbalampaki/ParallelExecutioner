
#ifndef TIMING_CSV_H
#define TIMING_CSV_H

#include <stdio.h>

#define TIMING_SCHEMA_VERSION 2

#define TIMING_CSV_HEADER \
"schema_version,run_id,job_id,node,git_sha,timestamp," \
"csr_variant,window_W,n_stays,n_records," \
"paradigm,platform,gpu,kernel_variant,precision," \
"partitioner,schedule,chunk_size,row_order," \
"n_ranks,n_threads,smt,thread_placement,n_nodes," \
"stage_mode,numa_balancing,mempolicy,whitelist_frac," \
"k_iterations,timed_region,repetition,warmup," \
"wall_time_s,throughput_rec_s," \
"work_imbalance,time_imbalance,n_records_max_thread,n_records_mean_thread," \
"bitidentical,l3_tcm,l3_tca,tot_cyc,tot_ins,achieved_ghz,notes"

#define THREAD_CSV_HEADER \
"schema_version,run_id,tid,cpu,socket,n_stays,n_records,secs"

typedef struct {
    int         schema_version;      /* set by timing_row_init */
    const char *run_id;
    long        job_id;
    const char *node;
    const char *git_sha;
    const char *timestamp;

    const char *csr_variant;         /* Wfull, sizematched_W24, ...  KEY */
    const char *window_W;            /* 6,12,24,48,full — derived, not a key */
    long        n_stays;
    long        n_records;

    const char *paradigm;            /* serial|openmp|pthreads|mpi|hybrid|cuda */
    const char *platform;            
    const char *gpu;                
    const char *kernel_variant;      /* shifted|welford|int16|... */
    const char *precision;           /* fp32|fp64|tf32|bf16 */

    const char *partitioner;        
    const char *schedule;
    long        chunk_size;          /* -1 when the schedule has no chunk */
    const char *row_order;           /* canonical|sorted_desc|sorted_asc */

    int n_ranks, n_threads, smt, n_nodes;

    const char *thread_placement;    /* socket0|socket1|spread|smt|unknown */

    const char *stage_mode;         
    int         numa_balancing;      
    const char *mempolicy;          
    double      whitelist_frac;      

    long        k_iterations;       
    const char *timed_region;     
    int         repetition;
    int         warmup;              /* 1 iff repetition == 0 */

    double      wall_time_s;
    double      throughput_rec_s;

    double      work_imbalance;      
    double      time_imbalance;      
    long        n_records_max_thread;
    double      n_records_mean_thread;

    const char *bitidentical;        

    long        l3_tcm, l3_tca, tot_cyc, tot_ins;   /* -1 if not collected */
    double      achieved_ghz;
    const char *notes;
} timing_row_t;

typedef struct {
    int    schema_version;
    const char *run_id;
    int    tid, cpu, socket;
    long   n_stays, n_records;
    double secs;
} thread_row_t;

void  timing_row_init(timing_row_t *r);
int   timing_row_validate(const timing_row_t *r, char *err, size_t errlen);
FILE *timing_csv_open(const char *path);   
FILE *thread_csv_open(const char *path);
int   timing_csv_write(FILE *f, const timing_row_t *r);  
int   thread_csv_write(FILE *f, const thread_row_t *r);


void  timing_make_run_id(char *buf, size_t n, long job_id, int rank, int seq);

#endif /* TIMING_CSV_H */

#ifdef TIMING_CSV_IMPLEMENTATION

#include <string.h>
#include <time.h>

static int tcsv__in(const char *s, const char *const *set)
{
    if (!s) return 0;
    for (int i = 0; set[i]; i++)
        if (!strcmp(s, set[i])) return 1;
    return 0;
}

static const char *const TCSV_PARTITIONER[] = {
    "none", "block", "cyclic", "block_cyclic", "nzbalanced",
    "wbalanced", "contig_opt", "greedy", NULL };
static const char *const TCSV_SCHEDULE[] = {
    "precomputed", "static", "static_chunked", "dynamic", "guided",
    "stealing", NULL };
static const char *const TCSV_ROW_ORDER[] = {
    "canonical", "sorted_desc", "sorted_asc", NULL };
static const char *const TCSV_STAGE[] = {
    "serial", "interleave", "mbind", NULL };
static const char *const TCSV_MEMPOL[] = {
    "default", "interleave", "bind", NULL };
static const char *const TCSV_REGION[] = { "kernel", "end_to_end", NULL };
static const char *const TCSV_BITID[]  = { "pass", "fail", "skip", NULL };
static const char *const TCSV_PLACE[]  = { "socket0", "socket1", "spread",
                                           "smt", "unknown", NULL };

void timing_row_init(timing_row_t *r)
{
    memset(r, 0, sizeof *r);
    r->schema_version = TIMING_SCHEMA_VERSION;
    r->run_id = r->node = r->git_sha = r->timestamp = "";
    r->csr_variant = r->window_W = "";
    r->paradigm = "serial";  r->platform = "broadwell";
    r->gpu = "none";         r->kernel_variant = "shifted";
    r->precision = "fp64";
    r->partitioner = "block"; r->schedule = "static";
    r->chunk_size = -1;       r->row_order = "canonical";
    r->n_ranks = r->n_threads = r->n_nodes = 1;
    r->thread_placement = "unknown";
    r->stage_mode = "interleave";   /*  the default is now interleaved */
    r->numa_balancing = -1;
    r->mempolicy = "default";
    r->whitelist_frac = 1.0;
    r->k_iterations = 1;
    r->timed_region = "kernel";
    r->bitidentical = "skip";
    r->l3_tcm = r->l3_tca = r->tot_cyc = r->tot_ins = -1;
    r->achieved_ghz = -1.0;
    r->notes = "";
}

int timing_row_validate(const timing_row_t *r, char *err, size_t errlen)
{
#define FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); return 0; } while (0)
    if (r->schema_version != TIMING_SCHEMA_VERSION)
        FAIL("schema_version %d != %d", r->schema_version, TIMING_SCHEMA_VERSION);
    if (!r->csr_variant || !*r->csr_variant)
        FAIL("csr_variant is required (window_W is not a key)");
    if (!tcsv__in(r->partitioner, TCSV_PARTITIONER))
        FAIL("bad partitioner '%s'", r->partitioner ? r->partitioner : "(null)");
    if (!tcsv__in(r->schedule, TCSV_SCHEDULE))
        FAIL("bad schedule '%s'", r->schedule ? r->schedule : "(null)");
    if (!tcsv__in(r->row_order, TCSV_ROW_ORDER)) FAIL("bad row_order");
    if (!tcsv__in(r->stage_mode, TCSV_STAGE))    FAIL("bad stage_mode");
    if (!tcsv__in(r->mempolicy, TCSV_MEMPOL))    FAIL("bad mempolicy");
    if (!tcsv__in(r->timed_region, TCSV_REGION)) FAIL("bad timed_region");
    if (!tcsv__in(r->bitidentical, TCSV_BITID))  FAIL("bad bitidentical");
    if (!tcsv__in(r->thread_placement, TCSV_PLACE))
        FAIL("bad thread_placement '%s'",
             r->thread_placement ? r->thread_placement : "(null)");

  
    int chunked = !strcmp(r->schedule, "static_chunked")
               || !strcmp(r->schedule, "dynamic")
               || !strcmp(r->schedule, "guided")
               || !strcmp(r->schedule, "stealing");
    if (chunked && r->chunk_size < 1)
        FAIL("schedule '%s' requires chunk_size >= 1", r->schedule);
    if (!chunked && r->chunk_size != -1)
        FAIL("schedule '%s' must have chunk_size = -1", r->schedule);

    /* The identification rule from schema.md §1.1: a row must say whether
     * its assignment was computable a priori. 'none' + 'static' is just
     * 'block' and would silently pollute the rows Day 12 regresses on. */
    if (!strcmp(r->partitioner, "none")) {
        if (strcmp(r->schedule, "dynamic") && strcmp(r->schedule, "guided")
            && strcmp(r->schedule, "stealing"))
            FAIL("partitioner 'none' requires a runtime schedule; "
                 "'none'+'%s' should be recorded as an explicit partitioner",
                 r->schedule);
    } else if (!strcmp(r->schedule, "precomputed")) {
        /* fine */
    }
    if (!strcmp(r->schedule, "precomputed") && !strcmp(r->partitioner, "none"))
        FAIL("schedule 'precomputed' requires a partitioner");

    if (r->whitelist_frac < 0.0 || r->whitelist_frac > 1.0)
        FAIL("whitelist_frac %g outside [0,1]", r->whitelist_frac);
    if (r->repetition < 0) FAIL("repetition < 0");
    if ((r->repetition == 0) != (r->warmup == 1))
        FAIL("warmup must be 1 iff repetition == 0");
    if (r->n_threads < 1) FAIL("n_threads < 1");
    if (r->n_threads > 16 && !r->smt && !strcmp(r->platform, "broadwell"))
        FAIL("n_threads=%d on broadwell requires smt=1", r->n_threads);
    if (r->k_iterations < 1) FAIL("k_iterations < 1");
    if (r->wall_time_s <= 0.0) FAIL("wall_time_s <= 0");
    return 1;
#undef FAIL
}


static void tcsv__str(FILE *f, const char *s, int last)
{
    if (!s) s = "";
    for (const char *p = s; *p; ++p) {
        char c = *p;
        if (c == ',') c = ';';
        else if (c == '\n' || c == '\r' || c == '"') c = ' ';
        fputc(c, f);
    }
    fputc(last ? '\n' : ',', f);
}

static FILE *tcsv__open(const char *path, const char *header)
{
    FILE *f = fopen(path, "a+");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) fprintf(f, "%s\n", header);
    return f;
}

FILE *timing_csv_open(const char *path) { return tcsv__open(path, TIMING_CSV_HEADER); }
FILE *thread_csv_open(const char *path) { return tcsv__open(path, THREAD_CSV_HEADER); }

int timing_csv_write(FILE *f, const timing_row_t *r)
{
    char err[256];
    if (!timing_row_validate(r, err, sizeof err)) {
        fprintf(stderr, "timing_csv_write: INVALID ROW: %s\n", err);
        return 0;
    }
    fprintf(f, "%d,", r->schema_version);
    tcsv__str(f, r->run_id, 0);
    fprintf(f, "%ld,", r->job_id);
    tcsv__str(f, r->node, 0);
    tcsv__str(f, r->git_sha, 0);
    tcsv__str(f, r->timestamp, 0);
    tcsv__str(f, r->csr_variant, 0);
    tcsv__str(f, r->window_W, 0);
    fprintf(f, "%ld,%ld,", r->n_stays, r->n_records);
    tcsv__str(f, r->paradigm, 0);
    tcsv__str(f, r->platform, 0);
    tcsv__str(f, r->gpu, 0);
    tcsv__str(f, r->kernel_variant, 0);
    tcsv__str(f, r->precision, 0);
    tcsv__str(f, r->partitioner, 0);
    tcsv__str(f, r->schedule, 0);
    fprintf(f, "%ld,", r->chunk_size);
    tcsv__str(f, r->row_order, 0);
    fprintf(f, "%d,%d,%d,", r->n_ranks, r->n_threads, r->smt);
    tcsv__str(f, r->thread_placement, 0);
    fprintf(f, "%d,", r->n_nodes);
    tcsv__str(f, r->stage_mode, 0);
    fprintf(f, "%d,", r->numa_balancing);
    tcsv__str(f, r->mempolicy, 0);
    fprintf(f, "%.4f,%ld,", r->whitelist_frac, r->k_iterations);
    tcsv__str(f, r->timed_region, 0);
    fprintf(f, "%d,%d,%.9f,%.6e,%.6f,%.6f,%ld,%.4f,",
            r->repetition, r->warmup, r->wall_time_s, r->throughput_rec_s,
            r->work_imbalance, r->time_imbalance,
            r->n_records_max_thread, r->n_records_mean_thread);
    tcsv__str(f, r->bitidentical, 0);
    fprintf(f, "%ld,%ld,%ld,%ld,%.4f,",
            r->l3_tcm, r->l3_tca, r->tot_cyc, r->tot_ins, r->achieved_ghz);
    tcsv__str(f, r->notes, 1);
    return 1;
}

int thread_csv_write(FILE *f, const thread_row_t *r)
{
    fprintf(f, "%d,", TIMING_SCHEMA_VERSION);
    tcsv__str(f, r->run_id, 0);
    fprintf(f, "%d,%d,%d,%ld,%ld,%.9f\n",
            r->tid, r->cpu, r->socket, r->n_stays, r->n_records, r->secs);
    return 1;
}

void timing_make_run_id(char *buf, size_t n, long job_id, int rank, int seq)
{
    snprintf(buf, n, "%ld-%d-%d-%06d", job_id, rank, (int)getpid(), seq);
}

#endif /* TIMING_CSV_IMPLEMENTATION */
