#include <stdlib.h>
#include "interface.h"

clock_t icopy_s = 0, icopy_e = 0, ocopy_s = 0, ocopy_e = 0, kernel_s = 0, kernel_e = 0, calling_s = 0, calling_e = 0;
double icopy, ocopy, kernel, calling;

void print_metrics(char* name){
    printf("\n%s\n", name);
    printf("Input Copy Time  : %.2f microseconds\n", icopy);
    printf("Output Copy Time : %.2f microseconds\n", ocopy);
    printf("Kernel Time      : %.2f microseconds\n", kernel);
    printf("Overhead Time    : %.2f microseconds\n", calling - (kernel+icopy+ocopy));
    printf("Calling Time     : %.2f microseconds\n", calling);
}

void reset_var(){
    icopy_s = icopy_e = ocopy_s = ocopy_e = 0;
    kernel_s = kernel_e = calling_s = calling_e = 0;
    icopy = ocopy = kernel = calling = 0.0;
}

int icopy_trans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER) {
    float *panel_start = (float *)A + ((long)Y + (long)X * (long)LDA);
    return GEMM_ITCOPY(M, N, panel_start, LDA, BUFFER);
}
int icopy_notrans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER) {
    float *panel_start = (float *)A + ((long)X + (long)Y * (long)LDA);
    return GEMM_INCOPY(M, N, panel_start, LDA, BUFFER);
}
int ocopy_notrans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER) {
    float *panel_start = (float *)A + ((long)X + (long)Y * (long)LDA);
    return GEMM_ONCOPY(M, N, panel_start, LDA, BUFFER); 
}
int ocopy_trans_op(long M, long N, float *A, long LDA, int X, int Y, float *BUFFER) {
    float *panel_start = (float *)A + ((long)Y + (long)X * (long)LDA);
    return GEMM_OTCOPY(M, N, panel_start, LDA, BUFFER);
}

void gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
           int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC){
    arg_t args;
    float *buffer;
    float *sa, *sb;
    copy_op_func_t copy_a;
    copy_op_func_t copy_b;

    if (TransA == CblasNoTrans) {
        copy_a = icopy_notrans_op;
    } else {
        copy_a = icopy_trans_op;
    }

    if (TransB == CblasNoTrans) {
        copy_b = ocopy_notrans_op;
    } else {
        copy_b = ocopy_trans_op;
    }

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
    }

    args.alpha = alpha;
    args.beta  = beta;

    if ((args.m == 0) || (args.n == 0)) return;

    buffer = (float *) malloc( BUFFER_SIZE );
    
    sa = (float *)( buffer );
    sb = (float *)( (long) sa + (BUFFER_SIZE/SIZE)/2 );

    gemm_tiling(&args, NULL, NULL, sa, sb, copy_a, copy_b);

    free(buffer);

    return;
}
