#include <stdlib.h>
#include "interface.h"


int kernel_op_pre_stride(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y) {
    float *c_panel_start = (float *)C + ((long)X + (long)Y * (long)LDC);
    return KERNEL_FUNC_PRE_STRIDE(M, N, K, ALPHA, SA, SB, c_panel_start, LDC);
}

int kernel_op_pre(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y) {
    float *c_panel_start = (float *)C + ((long)X + (long)Y * (long)LDC);
    return KERNEL_FUNC_PRE(M, N, K, ALPHA, SA, SB, c_panel_start, LDC);
}
int kernel_op(long M, long N, long K, float ALPHA, float* SA, float* SB, float* C, long LDC, int X, int Y) {
    float *c_panel_start = (float *)C + ((long)X + (long)Y * (long)LDC);
    return KERNEL_FUNC(M, N, K, ALPHA, SA, SB, c_panel_start, LDC);
}

// gemm_seq_buf: identical to gemm_seq() but takes caller-owned sa/sb scratch
// buffers instead of malloc/free-ing BUFFER_SIZE bytes on every call. Lets
// benchmark harnesses hoist the allocation outside the timed region; sa and
// sb must be GEMM_SB_OFFSET_BYTES and (BUFFER_SIZE - GEMM_SB_OFFSET_BYTES)
// bytes respectively, laid out contiguously exactly like gemm_seq()'s own
// split below (see GEMM_SB_OFFSET_BYTES in interface.h).
void gemm_seq_buf(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const enum CBLAS_SEQ Seq,
           int M, int N, int K,
           float alpha,
           const float *a, int ldA,
           const float *b, int ldB,
           float beta,
           float *c, int ldC,
           int TileN, int StrideN,
           float *sa, float *sb){
    arg_t args;
    copy_op_func_t copy_a;
    copy_op_func_t copy_b;
    kernel_op_func_t kernel;

    if (Order == CblasColMajor) {
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
        if (TransB == CblasTrans)   copy_b = ocopy_trans_op;
    } else if (Order == CblasRowMajor) {
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
        // Always the _stride kernel for Pre/Mid. It is a strict
        // generalisation of kernel_op_pre: gemm_kernel_pre hardcodes the
        // inter-column-group advance as `LDC*(ndim-3) - M`, which is only
        // correct when LDC == M, while gemm_kernel_pre_stride uses
        // `LDC*ndim - M*GEMM_UNROLL_N`; both reduce to M*(ndim-4) when
        // LDC == M, so unsliced calls are unchanged. level3_seq.c relies on
        // the general form to partition M across threads at GEMM_UNROLL_M
        // granularity INSIDE a layout block (LDC stays the block height while
        // M is the slice height) -- see the pick_m_slice comment there.
        kernel = kernel_op_pre_stride;
    } else if (Seq == CblasPos) {
        kernel = kernel_op;
    }

    args.alpha = alpha;
    args.beta  = beta;

    if ((args.m == 0) || (args.n == 0)) return;

    gemm_tiling_seq(&args, NULL, NULL, sa, sb, copy_a, copy_b, kernel, Seq, TileN, StrideN);

    return;
}

void gemm_seq(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const enum CBLAS_SEQ Seq,
           int M, int N, int K,
           float alpha,
           const float *a, int ldA,
           const float *b, int ldB,
           float beta,
           float *c, int ldC,
           int TileN, int StrideN){
    arg_t args;
    float *buffer;
    float *sa, *sb;
    copy_op_func_t copy_a;
    copy_op_func_t copy_b;
    kernel_op_func_t kernel;

    if (Order == CblasColMajor) {
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
        if (TransB == CblasTrans)   copy_b = ocopy_trans_op;
    } else if (Order == CblasRowMajor) {
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
        // Always the _stride kernel for Pre/Mid. It is a strict
        // generalisation of kernel_op_pre: gemm_kernel_pre hardcodes the
        // inter-column-group advance as `LDC*(ndim-3) - M`, which is only
        // correct when LDC == M, while gemm_kernel_pre_stride uses
        // `LDC*ndim - M*GEMM_UNROLL_N`; both reduce to M*(ndim-4) when
        // LDC == M, so unsliced calls are unchanged. level3_seq.c relies on
        // the general form to partition M across threads at GEMM_UNROLL_M
        // granularity INSIDE a layout block (LDC stays the block height while
        // M is the slice height) -- see the pick_m_slice comment there.
        kernel = kernel_op_pre_stride;
    } else if (Seq == CblasPos) {
        kernel = kernel_op;
    }

    args.alpha = alpha;
    args.beta  = beta;

    if ((args.m == 0) || (args.n == 0)) return;

    buffer = (float *) malloc( BUFFER_SIZE );
    
    sa = (float *)( buffer );
    sb = (float *)( (char *) sa + GEMM_SB_OFFSET_BYTES );

    gemm_tiling_seq(&args, NULL, NULL, sa, sb, copy_a, copy_b, kernel, Seq, TileN, StrideN);

    free(buffer);

    return;
}
