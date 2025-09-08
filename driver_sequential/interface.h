#ifndef __INTERFACE_H__
#define __INTERFACE_H__

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

#define KERNEL_FUNC gemm_kernel
#define GEMM_ITCOPY gemm_icopy
#define GEMM_ONCOPY gemm_ocopy
#define GEMM_BETA beta_operation

#include <time.h>
#include <stdio.h>
extern clock_t icopy_s, icopy_e, ocopy_s, ocopy_e, kernel_s, kernel_e, calling_s, calling_e;
extern double icopy, ocopy, kernel, calling;
void print_metrics();

typedef struct {
    void *a, *b, *c, *d;
    int m, n, k, lda, ldb, ldc, ldd;
    float alpha, beta;
} arg_t;

void gemm(int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC);

int gemm_tiling(arg_t *args, 
           long *range_m, long *range_n, 
           float *sa, float *sb);

int gemm_icopy(long m, long n, float *a, long lda, float *b);
int gemm_ocopy(long m, long n, float *a, long lda, float *b);
int gemm_kernel(long M, long N, long K, float alpha, float* A, float* B, float* C, long ldc);
int beta_operation(long m, long n, long dummy1, float beta, float *dummy2, long dummy3, float *dummy4, long dummy5, float *c, long ldc);
#endif