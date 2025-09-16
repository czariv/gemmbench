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
    float counter = 0;
    float *A = (float *) malloc(M*K*sizeof(float));
    for(int i=0; i<K; i++){
        for (int j=0; j<M; j++){
            A[M*i+j] = (float) counter;//((float) rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
        counter = counter + 1;
	    }
    }

    counter = 1;
    float *B = (float *) malloc(K*N*sizeof(float));
    for(int i=0; i<N; i++){
        for (int j=0; j<K; j++){
                B[K*i+j] = (float) counter++;//((float)rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
	}
    }
    counter = 1;
    float *B_t = (float *) malloc(K*N*sizeof(float));
    for(int i=0; i<N; i++){
        for (int j=0; j<K; j++){
                B_t[i+j*N] = (float) counter++;//((float)rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
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
        A, /*lda*/ M,
        B_t, /*ldb*/ N,
        beta,
        C_native, /*ldc*/ M);
    calling_e = clock();
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;

    //naive implementation
    for (int i = 0; i < N; i++){
        for (int j = 0; j < M; j++){
            C_naive[i*M+j] *= beta;
            for (int k = 0; k < K; k++)
                C_naive[i*M+j] += alpha * A[k*M+j] * B[i*K+k];
        }
    }

    clean_cache[0] = 10;
    print_metrics();
    #ifdef DEBUG
        printf("======================= DEBUG =======================\n");
        if((N<50)&&(M<50)&&(K<50)){
        printf("A =\n");
        for(int i=0; i<K; i++){
            for (int j=0; j<M; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)A[M*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("B =\n");
        for(int i=0; i<N; i++){
            for (int j=0; j<K; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)B[K*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("C native =\n");
        for(int i=0; i<N; i++){
            for (int j=0; j<M; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)C_native[M*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("C naive =\n");
        for(int i=0; i<N; i++){
            for (int j=0; j<M; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)C_naive[M*i + j]);
            }
            printf("\n");
        }
    } else{
        FILE *file_nat_c = fopen("c_native.csv","w");
        printf("C native = c_native.csv\n");
        for(int i=0; i<N; i++){
            for (int j=0; j<M; j++){
                if(j>0)
                    fprintf(file_nat_c, ",");
                fprintf(file_nat_c, "%.2f", C_native[i*M + j]);
            }
            fprintf(file_nat_c, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nai_c = fopen("c_naive.csv","w");
        printf("C naive = c_naive.csv\n");
        for(int i=0; i<N; i++){
            for (int j=0; j<M; j++){
                if(j>0)
                    fprintf(file_nai_c, ",");
                fprintf(file_nai_c, "%.2f", C_naive[i*M + j]);
            }
            fprintf(file_nai_c, "\n");
        }
    }
    #endif
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