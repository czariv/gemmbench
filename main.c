#include <linux/perf_event.h>
#include <math.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "interface_pos.h"


static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct counter {
    int fd;
    long long value;
    uint32_t type;
    uint64_t config;
};

static void setup_counter(struct counter *c, uint32_t type, uint64_t config, int group_fd) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 0;
    pe.exclude_hv = 0;

    c->fd = perf_event_open(&pe, 0, -1, group_fd, 0);
    if (c->fd == -1) {
        perror("perf_event_open");
        exit(1);
    }
    c->type = type;
    c->config = config;
}

static void region_start(struct counter *cs, int n, struct timespec *ts) {
    for (int i=0; i<n; i++) ioctl(cs[i].fd, PERF_EVENT_IOC_RESET, 0);
    for (int i=0; i<n; i++) ioctl(cs[i].fd, PERF_EVENT_IOC_ENABLE, 0);
    clock_gettime(CLOCK_MONOTONIC, ts);
}

static void region_stop(struct counter *cs, int n, struct timespec *ts, const char *label) {
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    for (int i=0; i<n; i++) {
        ioctl(cs[i].fd, PERF_EVENT_IOC_DISABLE, 0);
        read(cs[i].fd, &cs[i].value, sizeof(long long));
    }

    double elapsed = (ts_end.tv_sec - ts->tv_sec) +
                     (ts_end.tv_nsec - ts->tv_nsec)/1e9;

    printf("== %s ==\n", label);
    printf("Time: %f sec\n", elapsed);
    printf("Instructions: %lld\n", cs[0].value);
    printf("L1D loads: %lld  misses: %lld  hits: %lld miss rate: %.2f%%\n",
        cs[1].value, cs[2].value, cs[1].value - cs[2].value,
        100.0 * (double)cs[2].value / (double)(cs[1].value ? cs[1].value : 1));
    printf("Branches: %lld  misses: %lld  miss rate: %.2f%%\n",
        cs[3].value, cs[4].value,
        100.0 * (double)cs[4].value / (double)(cs[3].value ? cs[3].value : 1));
    printf("-----------------------------\n");
}

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
    for(int i=0; i<K; i++){
        for (int j=0; j<M; j++){
            A[M*i+j] = (float) counter;//((float) rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
	    counter = counter + 1;
	}
   }

    counter = 0;
    float *B = (float *) malloc(K*N*sizeof(float));
    for(int i=0; i<N; i++){
        for (int j=0; j<K; j++){
            B[K*i+j] = (float) counter;//((float)rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
	    counter = counter + 1;
	}
    }

    float* C_native = (float *) malloc(M*N*sizeof(float));
    float* C_native_og = (float *) malloc(M*N*sizeof(float));
    float* C_naive = (float *) malloc(M*N*sizeof(float));

    counter = 0;
    float *D = (float *) malloc(N*K*sizeof(float));
    for(int i=0; i<K; i++){
        for (int j=0; j<N; j++){
            D[N*i+j] = (float) counter;//((float) rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
	        counter = counter + 1;
	    }
    }

    float* E_native = (float *) malloc(M*K*sizeof(float));
    float* E_native_og = (float *) malloc(M*K*sizeof(float));
    float* E_naive = (float *) malloc(M*K*sizeof(float));

    float* clean_cache = malloc((1<<20) * sizeof(float));;

    for(int i=0; i < 1<<20; i++) clean_cache[i] = 0;

    struct counter ctrs[5];
    setup_counter(&ctrs[0], PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, -1);
    setup_counter(&ctrs[1], PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16), ctrs[0].fd);
    setup_counter(&ctrs[2], PERF_TYPE_HW_CACHE,
        (PERF_COUNT_HW_CACHE_L1D) |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16), ctrs[0].fd);
    setup_counter(&ctrs[3], PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
        ctrs[0].fd);
    setup_counter(&ctrs[4], PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_BRANCH_MISSES,
        ctrs[0].fd);

    struct timespec ts;
    
    region_start(ctrs, 5, &ts);
    // **** Native implementation **** //
    // native_start();
    gemm_pre(M, N, K,
        alpha,
        A, /*lda*/ K,
        B, /*ldb*/ N,
        beta,
        C_native, /*ldc*/ 128);
    // native_stop();
    // **** Native implementation **** //
    // counter = 0;
    // for(int i=0; i<(M*N); i++){
    //     C_native[i] = counter;
    //     counter++;
    // }
    region_stop(ctrs, 5, &ts, "gemm_pre");

    region_start(ctrs, 5, &ts);
    // **** Native implementation **** //
    // native_start();
    gemm_pos(M, K, N,
        alpha,
        C_native, /*lda*/ 128,
        D, /*ldb*/ K,
        beta,
        E_native, /*ldc*/ K);
    // native_stop();
    // **** Native implementation **** //
    region_stop(ctrs, 5, &ts, "gemm_pos");

    region_start(ctrs, 5, &ts);
    // **** Native implementation **** //
    // native_start();
    gemm(M, N, K,
        alpha,
        A, /*lda*/ K,
        B, /*ldb*/ N,
        beta,
        C_native_og, /*ldc*/ N);
    // native_stop();
    // **** Native implementation **** //
    // counter = 0;
    // for(int i=0; i<(M*N); i++){
    //     C_native_og[i] = counter;
    //     counter++;
    // }
    region_stop(ctrs, 5, &ts, "gemm1_native");

    region_start(ctrs, 5, &ts);
    // **** Native implementation **** //
    // native_start();
    gemm(M, K, N,
        alpha,
        C_native_og, /*lda*/ N,
        D, /*ldb*/ K,
        beta,
        E_native_og, /*ldc*/ K);
    // native_stop();
    // **** Native implementation **** //
    region_stop(ctrs, 5, &ts, "gemm2_native");

    region_start(ctrs, 5, &ts);
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
    region_stop(ctrs, 5, &ts, "gemm_naive");

    #ifdef DEBUG
    printf("======================= DEBUG =======================\n");
    if((N<50)&&(M<50)&&(K<50)){
        printf("A =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<K; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)A[K*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("B =\n");
        for(int i=0; i<K; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)B[N*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("C native =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)C_native[N*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("C naive =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)C_native_og[N*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("E native =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)E_native[N*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("E naive =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)E_native_og[N*i + j]);
            }
            printf("\n");
        }
    }
    else if ((N<1000)&&(M<1000)&&(K<1000)) {
        FILE *file_nat_c = fopen("c_native.csv","w");
        printf("C native = c_native.csv\n");
        for(int i=0; i<(M*N)/128; i++){
            for (int j=0; j<128; j++){
                if(j>0)
                    fprintf(file_nat_c, ",");
                fprintf(file_nat_c, "%.2f", C_native[128*i + j]);
            }
            fprintf(file_nat_c, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nai_c = fopen("c_naive.csv","w");
        printf("C naive = c_naive.csv\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    fprintf(file_nai_c, ",");
                fprintf(file_nai_c, "%.2f", C_native_og[N*i + j]);
            }
            fprintf(file_nai_c, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nat = fopen("e_native.csv","w");
        printf("E native = e_native.csv\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<K; j++){
                if(j>0)
                    fprintf(file_nat, ",");
                fprintf(file_nat, "%.2f", E_native[K*i + j]);
            }
            fprintf(file_nat, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nai = fopen("e_naive.csv","w");
        printf("E naive = e_naive.csv\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<K; j++){
                if(j>0)
                    fprintf(file_nai, ",");
                fprintf(file_nai, "%.2f", E_native_og[K*i + j]);
            }
            fprintf(file_nai, "\n");
        }
    }
    // printf("================== EXECUTION TIMES ==================\n");
    // printf("Elapsed Native Changed First: %f seconds\n", (double)(native_first) / CLOCKS_PER_SEC);
    // printf("Elapsed Native Original First: %f seconds\n", (double)(native_og_first) / CLOCKS_PER_SEC);

    // printf("Elapsed Native Changed Second: %f seconds\n", (double)(native_second) / CLOCKS_PER_SEC);
    // printf("Elapsed Native Original Second: %f seconds\n", (double)(native_og_second) / CLOCKS_PER_SEC);

    // printf("Elapsed Native Changed: %f seconds\n", (double)(native) / CLOCKS_PER_SEC);

    // printf("Elapsed Native Original: %f seconds\n", (double)(native_og) / CLOCKS_PER_SEC);

    // printf("Elapsed Naive: %f seconds\n", (double)(naive) / CLOCKS_PER_SEC);
    #endif

    float rel_diff;
    for(int i=0; i<M; i++){
        for(int j = 0; j<K; j++){
            rel_diff = fabsf(E_native[i*K+j] - E_native_og[i*K+j]) / (fabsf(E_native_og[i*K+j]) + 1e-8f);
            if (rel_diff > 1e-4)
                return 1;
        }
    }

    return 0;
}