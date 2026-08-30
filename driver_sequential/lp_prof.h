/* lp_prof.h -- opt-in, zero-cost-when-off instrumentation for level3_seq.c.
 *
 * Compiled in only with -DLP_PROF. Every counter is per-thread and padded to
 * a full cache line, so accumulating them inside the parallel regions does
 * not itself perturb what is being measured (an unpadded shared array would
 * false-share and manufacture exactly the kind of scaling loss we are trying
 * to attribute).
 *
 * Timebase is CLOCK_MONOTONIC via clock_gettime (vDSO, no syscall).
 */
#ifndef LP_PROF_H
#define LP_PROF_H

#ifdef LP_PROF

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define LP_PROF_MAXT 128

typedef struct {
    double t_icopy;
    double t_ocopy;
    double t_kernel;
    double t_region;   /* wall time this thread spent inside parallel regions */
    long   n_icopy;
    long   n_ocopy;
    long   n_kernel;
    char   pad[64 - ((4*sizeof(double) + 3*sizeof(long)) % 64)];
} lp_prof_thread_t;

extern lp_prof_thread_t lp_prof_t[LP_PROF_MAXT];
extern double lp_prof_wall;      /* wall time of the whole gemm_tiling_seq call */
extern double lp_prof_alloc;     /* master-thread time in malloc/free */
extern double lp_prof_serial;    /* master-thread time in explicitly sequential compute */
extern long   lp_prof_regions;   /* number of fork/join parallel regions entered */
extern int    lp_prof_threads;   /* omp_get_max_threads() seen by the call */
extern const char *lp_prof_stage;

static inline double lp_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define LP_T() (lp_prof_t[omp_get_thread_num()])

#define LP_RESET() do { \
    memset(lp_prof_t, 0, sizeof(lp_prof_t)); \
    lp_prof_wall = lp_prof_alloc = lp_prof_serial = 0.0; \
    lp_prof_regions = 0; \
} while (0)

#define LP_TIC(v)      double v = lp_now()
#define LP_ADD(fld, v) do { LP_T().fld += lp_now() - (v); LP_T().n_##fld++; } while (0)
#define LP_ADDT(fld,v) do { LP_T().fld += lp_now() - (v); } while (0)
#define LP_GADD(g, v)  do { (g) += lp_now() - (v); } while (0)
#define LP_REGION()    do { lp_prof_regions++; } while (0)

void lp_prof_report(void);

#else /* !LP_PROF */

#define LP_RESET()          do {} while (0)
#define LP_TIC(v)           do {} while (0)
#define LP_ADD(fld, v)      do {} while (0)
#define LP_ADDT(fld, v)     do {} while (0)
#define LP_GADD(g, v)       do {} while (0)
#define LP_REGION()         do {} while (0)

#endif /* LP_PROF */

#endif /* LP_PROF_H */
