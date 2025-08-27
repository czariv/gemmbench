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

#ifndef GEMM_R_PRE
    #define GEMM_R_PRE 128
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
#define KERNEL_FUNC_PRE gemm_kernel_pre


#include "interface.h"

void gemm_pre(int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC);

int gemm_tiling_pre(arg_t *args, 
           long *range_m, long *range_n, 
           float *sa, float *sb);

int gemm_kernel_pre(long M, long N, long K, float alpha, float* A, float* B, float* C, long ldc);
