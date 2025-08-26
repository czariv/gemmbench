#define EXTRACT_FLOAT(v) RISCV_RVV(vfmv_f_s_f32m1_f32)(v)
#ifndef TYPE
    #define TYPE float
#endif

#ifndef ZERO
    #define ZERO 0
#endif

#ifndef GEMM_UNROLL_M
    #define GEMM_UNROLL_M  16
#endif

#ifndef GEMM_UNROLL_N
    #define GEMM_UNROLL_N  8
#endif

#ifndef GEMM_P
    #define GEMM_P 128
#endif

#ifndef GEMM_Q
    #define GEMM_Q 128
#endif

#ifndef GEMM_R
    #define GEMM_R 16384
#endif

#ifndef BUFFER_SIZE
    #define BUFFER_SIZE (32 << 20)
#endif

#ifndef SIZE
    #define SIZE sizeof(TYPE)
#endif

#ifndef EVAL_THRESHOLD
    #define EVAL_THRESHOLD 0.001
#endif

#define COLMAJOR 1
#define GEMM_ITCOPY_POS gemm_icopy_pos


#include "interface_pre.h"

void gemm_pos(int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC);

int gemm_tiling_pos(arg_t *args, 
           long *range_m, long *range_n, 
           float *sa, float *sb);

int gemm_icopy_pos(long m, long n, float *a, long lda, float *b);
