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
#define KERNEL_FUNC_PRE gemm_kernel_pre
#define GEMM_INCOPY generic_icopy
#define GEMM_ITCOPY gemm_icopy
#define GEMM_ONCOPY gemm_ocopy
#define GEMM_OTCOPY generic_ocopy
#define GEMM_BETA beta_operation

#include <time.h>
#include <stdio.h>
extern clock_t icopy_s, icopy_e, ocopy_s, ocopy_e, kernel_s, kernel_e, calling_s, calling_e;
extern double icopy, ocopy, kernel, calling;
void print_metrics(char* name);
void reset_var();

typedef struct {
    void *a, *b, *c, *d;
    int m, n, k, lda, ldb, ldc, ldd;
    float alpha, beta;
} arg_t;

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_SEQ      {CblasPre=151, CblasMid=152, CblasPos=153} CBLAS_SEQ;

typedef int (*copy_op_func_t)(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER);
typedef int (*kernel_op_func_t)(long M, long N, long K, float ALPHA, 
                                float* SA, float* SB, float* C, long LDC, 
                                int X, int Y);

int icopy_trans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER);
int icopy_notrans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER);
int ocopy_notrans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER);
int ocopy_trans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER);
int kernel_op_pre(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y);
int kernel_op(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y);

void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
           int M, int N, int K,
           float alpha,
           const float *a, int ldA,
           const float *b, int ldB,
           float beta,
           float *c, int ldC);

int gemm_tiling(arg_t *args, 
           long *range_m, long *range_n, 
           float *sa, float *sb,
           copy_op_func_t copy_a_func,
           copy_op_func_t copy_b_func);

int gemm_icopy(long m, long n, float *a, long lda, float *b);
int gemm_ocopy(long m, long n, float *a, long lda, float *b);
int generic_icopy(long m, long n, float *a, long lda, float *b);
int generic_ocopy(long m, long n, float *a, long lda, float *b);
int gemm_kernel(long M, long N, long K, float alpha, float* A, float* B, float* C, long ldc);
int beta_operation(long m, long n, long dummy1, float beta, float *dummy2, long dummy3, float *dummy4, long dummy5, float *c, long ldc);


void gemm_seq(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const enum CBLAS_SEQ Seq,
           int M, int N, int K,
           float alpha,
           const float *a, int ldA,
           const float *b, int ldB,
           float beta,
           float *c, int ldC);

int gemm_tiling_seq(arg_t *args, 
           long *range_m, long *range_n, 
           float *sa, float *sb,
           copy_op_func_t copy_a_func,
           copy_op_func_t copy_b_func,
           kernel_op_func_t kernel_func,
           const enum CBLAS_SEQ Seq);

int gemm_kernel_pre(long M, long N, long K, float alpha, float* A, float* B, float* C, long ldc);

#endif