#include <stdlib.h>
#include "interface.h"

int kernel_op_pre(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y) {
    float *c_panel_start = (float *)C + ((long)X + (long)Y * (long)LDC);
    return KERNEL_FUNC_PRE(M, N, K, ALPHA, SA, SB, c_panel_start, LDC);
}
int kernel_op(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y) {
    float *c_panel_start = (float *)C + ((long)X + (long)Y * (long)LDC);
    return KERNEL_FUNC(M, N, K, ALPHA, SA, SB, c_panel_start, LDC);
}

void gemm_seq(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const enum CBLAS_SEQ Seq,
           int M, int N, int K,
           float alpha,
           const float *a, int ldA,
           const float *b, int ldB,
           float beta,
           float *c, int ldC){
    arg_t args;
    float *buffer;
    float *sa, *sb;
    copy_op_func_t copy_a;
    copy_op_func_t copy_b;
    kernel_op_func_t kernel;

    if (Order != CblasRowMajor) {
        args.m = M;
        args.n = N;
        args.k = K;

        args.a = (void *)a;
        args.b = (void *)b;
        args.c = (void *)c;

        args.lda = ldA;
        args.ldb = ldB;
        args.ldc = ldC;

        if (TransA == CblasNoTrans) copy_a = icopy_notrans_op;
        if (TransA == CblasTrans)   copy_a = icopy_trans_op;

        if (TransB == CblasNoTrans) copy_b = ocopy_notrans_op;
        if (TransB == CblasNoTrans) copy_b = ocopy_trans_op;
    }
    else if (Order == CblasRowMajor) {
        args.m = N;
        args.n = M;
        args.k = K;

        args.a = (void *)b;
        args.b = (void *)a;
        args.c = (void *)c;

        args.lda = ldB;
        args.ldb = ldA;
        args.ldc = ldC;
        if (TransB == CblasNoTrans) copy_a = icopy_notrans_op;
        if (TransB == CblasTrans)   copy_a = icopy_trans_op;

        if (TransA == CblasNoTrans) copy_b = ocopy_notrans_op;
        if (TransA == CblasTrans)   copy_b = ocopy_trans_op;
    }

    if ( (Seq == CblasPre) || (Seq == CblasMid)) {
        kernel = kernel_op_pre;
    } else if (Seq == CblasPos) {
        kernel = kernel_op;
    }

    args.alpha = alpha;
    args.beta  = beta;

    if ((args.m == 0) || (args.n == 0)) return;

    buffer = (float *) malloc( BUFFER_SIZE );
    
    sa = (float *)( buffer );
    sb = (float *)( (long) sa + (BUFFER_SIZE/SIZE)/2 );

    gemm_tiling_seq(&args, NULL, NULL, sa, sb, copy_a, copy_b, kernel, Seq);

    free(buffer);

    return;
}
