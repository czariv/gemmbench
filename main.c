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
    clock_t start, end;

    //printf("%d %d %d %f %f\n", M, N, K, alpha, beta);
    //srand(time(NULL));

    //gemm usage
    // \alpha * A_{mxk} * B_{kxn} + \beta * C_{mxn}
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

    float* clean_cache = malloc((1<<20) * sizeof(float));;

    for(int i=0; i < 1<<20; i++) clean_cache[i] = 0;


    calling_s = clock();
    //native implementation (teiu)
    gemm(M, N, K,
         alpha,
         A, /*lda*/ K,
         B, /*ldb*/ N,
         beta,
         C_native, /*ldc*/ N);
    calling_e = clock();
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    printf("Time taken (native): %f seconds\n", calling);

    start = clock();
    //naive implementation
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++)
          C_naive[i*N+j] *= beta;
      for (int k = 0; k < K; k++) {
         for (int j = 0; j < N; j++)
            C_naive[i*N+j] += alpha * A[i*K+k] * B[k*N+j];
      }
    }
    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time taken (naive): %f seconds\n", cpu_time_used);

    clean_cache[0] = 10;
    print_metrics();
     float rel_diff;
     for(int i=0; i<M; i++){
         for(int j = 0; j<N; j++){
             rel_diff = fabsf(C_native[i*N+j] - C_naive[i*N+j]) / (fabsf(C_naive[i*N+j]) + 1e-8f);
             if (rel_diff > 1e-4)
                 return 1;
         }
     }

    return 0;
}
