#define EXTRACT_FLOAT(v) RISCV_RVV(vfmv_f_s_f32m1_f32)(v)

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
