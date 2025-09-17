#include "interface.h"
#include <stdlib.h>

#define COMPSIZE 1

#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) \
	GEMM_BETA((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
    BETA, NULL, 0, NULL, 0, \
    (float *)(C) + ((N_FROM) + (M_FROM) * (LDC)), LDC)

#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) GEMM_ITCOPY(M, N, (float *)(A) + ((Y) + (X) * (LDA)), LDA, BUFFER);

#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) GEMM_ONCOPY(M, N, (float *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER);

#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
	KERNEL_FUNC_PRE(M, N, K, ALPHA, SA, SB, (float *)(C) + ((X) + (Y) * LDC), LDC)

#define A	args -> a
#define LDA	args -> lda
#define B	args -> b
#define LDB	args -> ldb
#define C	args -> c
#define LDC	args -> ldc
#define M	args -> m
#define N	args -> n
#define K	args -> k

int gemm_tiling_pre(arg_t *args, long *range_m, long *range_n, float *sa, float *sb){
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

    if (beta) {
        if (beta != 1) {
            BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc);
        }
    }

    if ((k == 0) || (alpha == 0)) return 0;

    l2size = GEMM_P * GEMM_Q;

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

            // **** Native implementation **** //
            // icopy_start();
            icopy_s = clock();
            ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
            icopy_e = clock();
            icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;
            // icopy_stop();
            // **** Native implementation **** //

            for(jjs = js; jjs < js + min_j; jjs += min_jj){
                min_jj = min_j + js - jjs;
                if (min_jj >= 3*GEMM_UNROLL_N) min_jj = 3*GEMM_UNROLL_N;
                else
                /*
                    if (min_jj >= 2*GEMM_UNROLL_N) min_jj = 2*GEMM_UNROLL_N;
                        else
                */
                        if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;


                // **** Native implementation **** //
                // ocopy_start();
                ocopy_s = clock();
                OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, sb + pad_min_l * (jjs - js) * COMPSIZE * l1stride);
                ocopy_e = clock();
                ocopy += (double)(ocopy_e - ocopy_s) / CLOCKS_PER_SEC * 1000;
                // ocopy_stop();
                // **** Native implementation **** //

                // **** Native implementation **** //
                // kernel_start();
                kernel_s = clock();
                KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa, sb + pad_min_l * (jjs - js)  * COMPSIZE * l1stride, c, min_i, m_from, js/GEMM_R * m_to + jjs);
                kernel_e = clock();
                kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;
                // kernel_stop();
                // **** Native implementation **** //

            }

            for(is = m_from + min_i; is < m_to; is += min_i){
                min_i = m_to - is;

                if (min_i >= GEMM_P * 2) {
                    min_i = GEMM_P;
                } else
                    if (min_i > GEMM_P) {
                        min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                // **** Native implementation **** //
                // icopy_start();
                icopy_s = clock();
                ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
                icopy_e = clock();
                icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;
                // icopy_stop();
                // **** Native implementation **** //

                // **** Native implementation **** //
                // kernel_start();
                kernel_s = clock();
                KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c, min_i, is*min_j, js/GEMM_R * m_to);
                kernel_e = clock();
                kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;
                // kernel_stop();
                // **** Native implementation **** //

            } /* end of is */
        } /* end of js */
    } /* end of ls */
    return 0;
}
