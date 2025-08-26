#include <stdlib.h>
#include "interface_pos.h"

#ifndef COLMAJOR
void gemm_pos(int M, int N, int K,
           float alpha,
           float *a, int ldA,
           float *b, int ldB,
           float beta,
           float *c, int ldC){
#else
void gemm_pos(int N, int M, int K,
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

    gemm_tiling_pos(&args, NULL, NULL, sa, sb);

    free(buffer);

    return;
}
