#define EXTRACT_FLOAT(v) RISCV_RVV(vfmv_f_s_f32m1_f32)(v)


#include "interface_pre.h"

void gemm_pos(int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC);

int gemm_tiling_pos(arg_t *args, 
           long *range_m, long *range_n, 
           float *sa);

int gemm_icopy_pos(long m, long n, float *a, long lda, float *b);
