#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "interface.h"
#include <stdint.h>
#include <math.h>

#define MAX 64
#define MAXELE 1024

#define rand_min -5
#define rand_max 5

int main(int argc, char* argv[]){

    if (argc != 6) {
        fprintf(stderr, "Invalid number of arguments. M N K Alpha Beta expected.\n");
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    float alpha = (float) atof(argv[4]);
    float beta = (float) atof(argv[5]);

    //printf("%d %d %d %f %f\n", M, N, K, alpha, beta);
    //srand(time(NULL));

    //gemm usage
    // \alpha * A_{mxk} * B_{kxn} + \beta * C_{mxn}
    // \alpha * C{mxn} * D{nxk} + \beta * E{mxk}
    float counter = 0;
    float *A = (float *) malloc(M*K*sizeof(float));
    for(int i=0; i<M; i++){
        for (int j=0; j<K; j++){
            A[K*i+j] = (float) counter;//((float) rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
	        counter = counter + 1;
	    }
    }

    counter = 0;
    float *B = (float *) malloc(K*N*sizeof(float));
    for(int i=0; i<K; i++){
        for (int j=0; j<N; j++){
            B[N*i+j] = (float) counter;//((float)rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
            counter = counter + 1;
        }
    }

    float* C_native = (float *) malloc(M*N*sizeof(float));
    float* C_naive = (float *) malloc(M*N*sizeof(float));

    counter = 0;
    float *D = (float *) malloc(N*K*sizeof(float));
    for(int i=0; i<N; i++){
        for (int j=0; j<K; j++){
            D[K*i+j] = (float) counter;//((float) rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
	        counter = counter + 1;
	    }
    }

    float* E_native = (float *) malloc(M*K*sizeof(float));
    float* E_naive = (float *) malloc(M*K*sizeof(float));
    clock_t tic = clock();

    // **** Native implementation **** //
    // native_start();
    gemm(M, N, K,
        alpha,
        A, /*lda*/ K,
        B, /*ldb*/ N,
        beta,
        C_native, /*ldc*/ N);
    // native_stop();
    // **** Native implementation **** //

    // **** Native implementation **** //
    // native_start();
    gemm(M, K, N,
        alpha,
        C_native, /*lda*/ N,
        D, /*ldb*/ K,
        beta,
        E_native, /*ldc*/ K);
    // native_stop();
    // **** Native implementation **** //

    
    clock_t toc = clock();

    printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    // **** Naive implementation **** //
    // naive_start();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            C_naive[i*N+j] *= beta;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++)
                C_naive[i*N+j] += alpha * A[i*K+k] * B[k*N+j];
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            E_naive[i*K+j] *= beta;
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < K; j++)
                E_naive[i*K+j] += alpha * C_naive[i*N+k] * D[k*K+j];
        }
    }
    // naive_stop();
    // **** Naive implementation **** //

    float rel_diff;
    for(int i=0; i<M; i++){
        for(int j = 0; j<K; j++){
            rel_diff = fabsf(E_native[i*K+j] - E_naive[i*K+j]) / (fabsf(E_naive[i*K+j]) + 1e-8f);
            if (rel_diff > 1e-4)
                return 1;
        }
    }

    return 0;
}
