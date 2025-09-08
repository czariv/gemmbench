#include <stdlib.h>
#include "interface.h"

clock_t icopy_s = 0, icopy_e = 0, ocopy_s = 0, ocopy_e = 0, kernel_s = 0, kernel_e = 0, calling_s = 0, calling_e = 0;
double icopy, ocopy, kernel, calling;

void print_metrics(){
    printf("\n--- Execution Time Metrics ---\n");
    printf("Input Copy Time  : %.2f microseconds\n", icopy);
    printf("Output Copy Time : %.2f microseconds\n", ocopy);
    printf("Kernel Time      : %.2f microseconds\n", kernel);
    printf("Overhead Time    : %.2f microseconds\n", calling - (kernel+icopy+ocopy));
    printf("Calling Time     : %.2f microseconds\n", calling);
}

#ifndef COLMAJOR
void gemm(int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC){
#else
void gemm(int N, int M, int K,
           float alpha,
           float *b, int ldB,
           float *a, int ldA,
           float beta,
           float *c, int ldC){
#endif
    arg_t args;
    float *buffer;
    float *sa, *sb;

    args.m = M;
    args.n = N;
    args.k = K;

    args.a = (void *)a;
    args.b = (void *)b;
    args.c = (void *)c;

    args.lda = ldA;
    args.ldb = ldB;
    args.ldc = ldC;

    args.alpha = alpha;
    args.beta  = beta;

    if ((args.m == 0) || (args.n == 0)) return;

    buffer = (float *) malloc( BUFFER_SIZE );
    
    sa = (float *)( buffer );
    sb = (float *)( (long) sa + (BUFFER_SIZE/SIZE)/2 );

    gemm_tiling(&args, NULL, NULL, sa, sb);

    free(buffer);

    return;
}
