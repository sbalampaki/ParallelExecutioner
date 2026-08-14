
#ifndef PAPI_WRAP_H
#define PAPI_WRAP_H

/* index order matches the timing CSV column order */
enum { PW_L3_TCM = 0, PW_L3_TCA, PW_TOT_CYC, PW_TOT_INS, PW_NEVENTS };

/* Call once from the master thread. 0 = counters available. */
int         pw_init(void);
/* Per thread, immediately before the timed region. */
int         pw_thread_start(void);

int         pw_thread_stop(long *out);
const char *pw_status(void);
const char *pw_events(void);

#endif /* PAPI_WRAP_H */

#ifdef PAPI_WRAP_IMPLEMENTATION

#include <stdio.h>
#include <string.h>

static char pw__status[256] = "PAPI: not compiled in (-DUSE_PAPI)";
static char pw__events[256] = "";

#ifdef USE_PAPI

#include <papi.h>
#include <pthread.h>

static int pw__code[PW_NEVENTS];      /* PAPI event codes */
static int pw__have[PW_NEVENTS];      /* 1 if this event armed globally */
static int pw__ok = 0;

static __thread int pw__es = PAPI_NULL;
static __thread int pw__slot[PW_NEVENTS];   /* event -> index in read buffer */
static __thread int pw__n = 0;

static unsigned long pw__tid(void) { return (unsigned long)pthread_self(); }

int pw_init(void)
{
    static const char *names[PW_NEVENTS] =
        { "PAPI_L3_TCM", "PAPI_L3_TCA", "PAPI_TOT_CYC", "PAPI_TOT_INS" };

    int v = PAPI_library_init(PAPI_VER_CURRENT);
    if (v != PAPI_VER_CURRENT) {
        snprintf(pw__status, sizeof pw__status,
                 "PAPI: library_init failed (%d)", v);
        return -1;
    }
    if (PAPI_thread_init(pw__tid) != PAPI_OK) {
        snprintf(pw__status, sizeof pw__status, "PAPI: thread_init failed");
        return -1;
    }

    size_t off = 0;
    int n_have = 0;
    for (int i = 0; i < PW_NEVENTS; i++) {
        pw__have[i] = 0;
        if (PAPI_event_name_to_code((char *)names[i], &pw__code[i]) != PAPI_OK)
            continue;
        if (PAPI_query_event(pw__code[i]) != PAPI_OK)
            continue;
        pw__have[i] = 1;
        n_have++;
        off += (size_t)snprintf(pw__events + off, sizeof pw__events - off,
                                "%s%s", off ? "," : "", names[i]);
    }
    pw__ok = (n_have > 0);
    snprintf(pw__status, sizeof pw__status,
             "PAPI: %d/%d events available", n_have, PW_NEVENTS);
    return pw__ok ? 0 : -1;
}

int pw_thread_start(void)
{
    if (!pw__ok) return -1;
    PAPI_register_thread();
    pw__es = PAPI_NULL;
    pw__n = 0;
    for (int i = 0; i < PW_NEVENTS; i++) pw__slot[i] = -1;
    if (PAPI_create_eventset(&pw__es) != PAPI_OK) { pw__es = PAPI_NULL; return -1; }
    for (int i = 0; i < PW_NEVENTS; i++) {
        if (!pw__have[i]) continue;
        /* Added one at a time: a counter-allocation conflict then costs
         * that single event rather than the whole set. */
        if (PAPI_add_event(pw__es, pw__code[i]) == PAPI_OK)
            pw__slot[i] = pw__n++;
    }
    if (pw__n == 0) { PAPI_destroy_eventset(&pw__es); pw__es = PAPI_NULL; return -1; }
    return PAPI_start(pw__es) == PAPI_OK ? 0 : -1;
}

int pw_thread_stop(long *out)
{
    for (int i = 0; i < PW_NEVENTS; i++) out[i] = -1;
    if (pw__es == PAPI_NULL) return -1;
    long long buf[PW_NEVENTS];
    int rc = PAPI_stop(pw__es, buf);
    if (rc == PAPI_OK)
        for (int i = 0; i < PW_NEVENTS; i++)
            if (pw__slot[i] >= 0) out[i] = (long)buf[pw__slot[i]];
    PAPI_cleanup_eventset(pw__es);
    PAPI_destroy_eventset(&pw__es);
    pw__es = PAPI_NULL;
    return rc == PAPI_OK ? 0 : -1;
}

#else

int pw_init(void) { return -1; }
int pw_thread_start(void) { return -1; }
int pw_thread_stop(long *out)
{
    for (int i = 0; i < PW_NEVENTS; i++) out[i] = -1;
    return -1;
}

#endif

const char *pw_status(void) { return pw__status; }
const char *pw_events(void) { return pw__events[0] ? pw__events : "(none)"; }

#endif /* PAPI_WRAP_IMPLEMENTATION */
