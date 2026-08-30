#include "interface.h"
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#define COMPSIZE 1

// ── Constrained-case parallelism for the is-loop (M-blocking) ──────────────
// Mirrors driver_sequential/level3_seq.c's compute_m_blocks -- see the
// comment there for the full rationale. Duplicated here (not shared via a
// header) because level3.c and level3_seq.c are independent translation
// units; MAX_M_BLOCKS is a local macro, no risk of ODR conflict.
#define MAX_M_BLOCKS 4096

static int compute_m_blocks(long m_from, long m_to, int gemm_p_new,
                             long *offsets, long *sizes) {
    int nb = 0;
    long is = m_from;
    while (is < m_to) {
        long min_i = m_to - is;
        if (min_i >= gemm_p_new * 2) {
            min_i = gemm_p_new;
        } else if (min_i > gemm_p_new) {
            min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
        }
        assert(nb < MAX_M_BLOCKS && "increase MAX_M_BLOCKS for this workload's M");
        offsets[nb] = is;
        sizes[nb]   = min_i;
        nb++;
        is += min_i;
    }
    return nb;
}

// See the identical helper in level3_seq.c for why panels must be
// independent allocations, not offsets into the shared sa/sb block: this
// path's kernel reads `sb` (packed by block 0) as its B input, and an
// offset-based panel let one thread's overrun corrupt `sb` while other
// threads were still reading it. Confirmed by experiment on the
// level3_seq.c CblasPre path (same structure); not independently
// re-confirmed as broken here, but the risk is structurally identical, so
// fixed the same way rather than left as a latent, unverified risk.
static float **alloc_thread_panels(int nthreads, long panel_floats) {
    float **panels = malloc((size_t)nthreads * sizeof(float *));
    for (int t = 0; t < nthreads; t++)
        panels[t] = malloc((size_t)panel_floats * sizeof(float));
    return panels;
}
static void free_thread_panels(float **panels, int nthreads) {
    for (int t = 0; t < nthreads; t++) free(panels[t]);
    free(panels);
}

#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) \
	GEMM_BETA((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
    BETA, NULL, 0, NULL, 0, \
    (float *)(C) + ((M_FROM) + (N_FROM) * (LDC)), LDC)

#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
	KERNEL_FUNC(M, N, K, ALPHA, SA, SB, (float *)(C) + ((X) + (Y) * LDC), LDC)

#define A	args -> a
#define LDA	args -> lda
#define B	args -> b
#define LDB	args -> ldb
#define C	args -> c
#define LDC	args -> ldc
#define M	args -> m
#define N	args -> n
#define K	args -> k

int gemm_tiling(arg_t *args, long *range_m, long *range_n, float *sa, float *sb, 
           copy_op_func_t ICOPY_OPERATION,
           copy_op_func_t OCOPY_OPERATION){
    long k, lda, ldb, ldc;
    float alpha, beta;
    float *a, *b;
    float *c;
    long m_from, m_to, n_from, n_to;

    long ls, is, js;
    long min_l, min_i, min_j;
    long jjs, min_jj;

    long l1stride, gemm_p, l2size;

    k = K;

    a = (float *)A;
    b = (float *)B;
    c = (float *)C;

    lda = LDA;
    ldb = LDB;
    ldc = LDC;

    alpha = args -> alpha;
    beta  = args -> beta;

    m_from = 0;
    m_to   = M;

    if (range_m) {
        m_from = *(((long *)range_m) + 0);
        m_to   = *(((long *)range_m) + 1);
    }

    n_from = 0;
    n_to   = N;

    if (range_n) {
        n_from = *(((long *)range_n) + 0);
        n_to   = *(((long *)range_n) + 1);
    }

    if (beta != 1) {
        BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc);
    }

    if ((k == 0) || (alpha == 0)) return 0;

    l2size = GEMM_P * GEMM_Q;

    long m_blk_off[MAX_M_BLOCKS], m_blk_sz[MAX_M_BLOCKS];
    int ref_nthreads = omp_get_max_threads();
    long ref_panel_floats = (long)GEMM_P * GEMM_Q;
    float **ref_thread_panels = (ref_nthreads > 1)
        ? alloc_thread_panels(ref_nthreads, ref_panel_floats) : NULL;

    for(js = n_from; js < n_to; js += GEMM_R){
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for(ls = 0; ls < k; ls += min_l){

            min_l = k - ls;

            if (min_l >= GEMM_Q * 2) {
                // gemm_p = GEMM_P;
                min_l  = GEMM_Q;
            } else {
                if (min_l > GEMM_Q) {
                    min_l = ((min_l / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }
                gemm_p = ((l2size / min_l + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                while (gemm_p * min_l > l2size) gemm_p -= GEMM_UNROLL_M;
            }

            long pad_min_l = min_l;

            /* First, we have to move data A to L2 cache */
            min_i = m_to - m_from;
            l1stride = 1;

            if (min_i >= GEMM_P * 2) {
                min_i = GEMM_P;
            } else {
                if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                } else {
                    l1stride = 0;
                }
            }

      icopy_s = clock();
      ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
      icopy_e = clock();
      icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;

            for(jjs = js; jjs < js + min_j; jjs += min_jj){
                min_jj = min_j + js - jjs;
                if (min_jj >= 3*GEMM_UNROLL_N) min_jj = 3*GEMM_UNROLL_N;
                else
                /*
                    if (min_jj >= 2*GEMM_UNROLL_N) min_jj = 2*GEMM_UNROLL_N;
                        else
                */
                        if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;


  ocopy_s = clock();
  OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs,
    sb + pad_min_l * (jjs - js) * COMPSIZE * l1stride);
  ocopy_e = clock();
  ocopy += (double)(ocopy_e - ocopy_s) / CLOCKS_PER_SEC * 1000;


  kernel_s = clock();
  KERNEL_OPERATION(min_i, min_jj, min_l, alpha,
     sa, sb + pad_min_l * (jjs - js)  * COMPSIZE * l1stride, c, ldc, m_from, jjs);
  kernel_e = clock();
  kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;

            }

  // Block-0 above packed the full B panel into `sb`; every remaining
  // is-block only READS sb (read-only from here on) and writes its own
  // private `sa` panel + disjoint slice of `c`, so this is safe to run in
  // parallel with static partitioning over precomputed block boundaries --
  // as long as each thread's panel is a genuinely independent allocation
  // (see alloc_thread_panels() above). See level3_seq.c's CblasMid branch
  // for the full rationale -- identical pattern, applied here to the plain
  // (non-propagating) GEMM path so the "oblas" reference column is a fair
  // parallel baseline, not a single-threaded one, when comparing against
  // parallel LP-GEMM.

  if (ref_nthreads <= 1) {
            for(is = m_from + min_i; is < m_to; is += min_i){
                min_i = m_to - is;

                if (min_i >= GEMM_P * 2) {
                    min_i = GEMM_P;
                } else
                    if (min_i > GEMM_P) {
                        min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

  icopy_s = clock();
  ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
  icopy_e = clock();
  icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;
  // **** Native implementation **** //

  // **** Native implementation **** //
  kernel_s = clock();
  KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is, js);
  kernel_e = clock();
  kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;

            } /* end of is */
  } else {
      // CALLER CONTRACT: same thread count as every other stage being
      // compared/chained -- see level3_seq.c for the full note.
      int n_mblocks_ref = compute_m_blocks(m_from + min_i, m_to, GEMM_P, m_blk_off, m_blk_sz);
      #pragma omp parallel for schedule(static)
      for (int bi = 0; bi < n_mblocks_ref; bi++) {
          int tid = omp_get_thread_num();
          float *sa_t = ref_thread_panels[tid];
          long is_b = m_blk_off[bi];
          long mi_b = m_blk_sz[bi];
          ICOPY_OPERATION(min_l, mi_b, a, lda, ls, is_b, sa_t);
          KERNEL_OPERATION(mi_b, min_j, min_l, alpha, sa_t, sb, c, ldc, is_b, js);
      }
  }
        } /* end of js */
    } /* end of ls */
    if (ref_thread_panels) free_thread_panels(ref_thread_panels, ref_nthreads);
    return 0;
}
