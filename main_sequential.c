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
#include "interface.h"

#ifdef PERF
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
#endif

#define MAX 64
#define MAXELE 1024

#define rand_min -5
#define rand_max 5

int main(int argc, char* argv[]){

    if (argc != 8) {
        fprintf(stderr, "Invalid number of arguments. M K N1 N2 N3 Alpha Beta expected.\n");
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N1 = atoi(argv[3]);
    int N2 = atoi(argv[4]);
    int N3 = atoi(argv[5]);
    float alpha = (float) atof(argv[6]);
    float beta = (float) atof(argv[7]);
    double start, cost;

    //printf("%d %d %d %f %f\n", M, N, K, alpha, beta);
    //srand(time(NULL));

    //gemm usage
    // \alpha * A_{mxk} * B_{kxn} + \beta * C_{mxn}
    // \alpha * C{mxn} * D{nxk} + \beta * E{mxk}
    float counter = 0;
    float *A_t = (float *) malloc(M*K*sizeof(float));
    for(int i=0; i<K; i++){
        for (int j=0; j<M; j++){
            A_t[i+j*K] = (float) counter++;//((float) rand() / (float)RAND_MAX) * (rand_max-rand_min) + rand_min;
        }
    }

    counter = 0;
    float *B_t = (float *) malloc(K*N1*sizeof(float));
    for(int i=0; i<N1; i++){
        for (int j=0; j<K; j++){
            B_t[i+j*N1] = (float) counter++;
        }
    }

    float* C_native = (float *) malloc(M*N1*sizeof(float));
    float* C_native_og = (float *) malloc(M*N1*sizeof(float));

    counter = 0;
    float *D_t = (float *) malloc(N1*N2*sizeof(float));
    for(int i=0; i<N2; i++){
        for (int j=0; j<N1; j++){
             D_t[i+j*N2] = (float) counter++;
        }
    }


    float* E_native = (float *) malloc(M*N2*sizeof(float));
    float* E_native_og = (float *) malloc(M*N2*sizeof(float));

    float* F_native = (float *) malloc(M*N3*sizeof(float));
    float* F_native_og = (float *) malloc(M*N3*sizeof(float));

    counter = 0;
    float *G_t = (float *) malloc(N2*N3*sizeof(float));
    for(int i=0; i<N3; i++)
        for(int j=0; j<N2; j++)
            G_t[i+j*N3] = (float) counter++;

    float* clean_cache = malloc((1<<20) * sizeof(float));;

    for(int i=0; i < 1<<20; i++) clean_cache[i] = 0;
    
    #ifdef PERF
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
    #endif
    reset_var();
    calling_s = clock();
    // **** Native implementation **** //
    // native_start();
    gemm_pre(N1, M, K,
        alpha,
        B_t, /*ldb*/ N1,
        A_t, /*lda*/ K,
        beta,
        C_native, /*ldc*/ N1);
    // native_stop();
    // **** Native implementation **** //
    calling_e = clock();
    #ifdef PERF
    region_stop(ctrs, 5, &ts, "gemm_pre");
    #endif
    #ifdef TIME
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    print_metrics("gemm_pre");
    reset_var();
    #endif
    
    #ifdef PERF
    region_start(ctrs, 5, &ts);
    #endif
    calling_s = clock();
    // **** Native implementation **** //
    // native_start();
    gemm_mid(N2, M, N1,
        alpha,
        D_t, /*ldb*/ N2,
        C_native, /*lda*/ N1,
        beta,
        E_native, /*ldc*/ N2);
    // native_stop();
    // **** Native implementation **** //
    calling_e = clock();
    #ifdef PERF
    region_stop(ctrs, 5, &ts, "gemm_mid");
    #endif
    #ifdef TIME
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    print_metrics("gemm_mid");
    reset_var();
    #endif

    #ifdef PERF
    region_start(ctrs, 5, &ts);
    #endif
    calling_s = clock();
    // **** Native implementation **** //
    // native_start();
    gemm_pos(N3, M, N2,
        alpha,
        G_t, /*ldb*/ N3,
        E_native, /*lda*/ N2,
        beta,
        F_native, /*ldc*/ N3);
    // native_stop();
    // **** Native implementation **** //
    calling_e = clock();
    #ifdef PERF
    region_stop(ctrs, 5, &ts, "gemm_pos");
    #endif
    #ifdef TIME
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    print_metrics("gemm_pos");
    reset_var();
    #endif

    // calling_e = clock();
    // calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    // printf("Time taken (native): %f seconds\n", calling);
    free(clean_cache);
    clean_cache = malloc((1<<20) * sizeof(float));;
    
    #ifdef PERF
    region_start(ctrs, 5, &ts);
    #endif
    calling_s = clock();
    // **** Native implementation **** //
    // native_start();
    gemm(N1, M, K,
        alpha,
        B_t, /*lda*/ N1,
        A_t, /*ldb*/ K,
        beta,
        C_native_og, /*ldc*/ N1);
    // native_stop();
    // **** Native implementation **** //
    calling_e = clock();
    #ifdef PERF
    region_stop(ctrs, 5, &ts, "gemm1_native");
    #endif
    #ifdef TIME
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    print_metrics("native_pre");
    reset_var();
    #endif
    
    #ifdef PERF
    region_start(ctrs, 5, &ts);
    #endif
    // **** Native implementation **** //
    // native_start();
    calling_s = clock();
    gemm(N2, M, N1,
        alpha,
        D_t, /*ldb*/ N2,
        C_native_og, /*lda*/ N1,
        beta,
        E_native_og, /*ldc*/ N2);
    // native_stop();
    // **** Native implementation **** //
    calling_e = clock();
    #ifdef PERF
    region_stop(ctrs, 5, &ts, "gemm2_native");
    #endif
    #ifdef TIME
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    print_metrics("native_mid");
    reset_var();
    #endif

    #ifdef PERF
    region_start(ctrs, 5, &ts);
    #endif
    calling_s = clock();
    // **** Native implementation **** //
    // native_start();
    gemm(N3, M, N2,
        alpha,
        G_t, /*lda*/ N3,
        E_native_og, /*ldb*/ N2,
        beta,
        F_native_og, /*ldc*/ N3);
    // native_stop();
    // **** Native implementation **** //
    calling_e = clock();
    #ifdef PERF
    region_stop(ctrs, 5, &ts, "gemm3_native");
    #endif
    #ifdef TIME
    calling = (double)(calling_e - calling_s) / CLOCKS_PER_SEC * 1000;
    print_metrics("native_pos");
    reset_var();
    #endif

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
        for(int i=0; i<(M*N)/8; i++){
            for (int j=0; j<8; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)C_native[8*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("C naive =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)C_native_og[i*N + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("E native =\n");
        for(int i=0; i<(M*K)/8; i++){
            for (int j=0; j<8; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)E_native[8*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("E naive =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<K; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)E_native_og[K*i + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("F native =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)F_native[i*N + j]);
            }
            printf("\n");
        }
        printf("-----------------------------------------------------\n");
        printf("F naive =\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    printf(", ");
                printf("%d", (int)F_native_og[i*N + j]);
            }
            printf("\n");
        }
    }
    else if ((N<1000)&&(M<1000)&&(K<1000)) {
        FILE *file_nat_c = fopen("c_native.csv","w");
        printf("C native = c_native.csv\n");
        for(int i=0; i<(M*N)/8; i++){
            for (int j=0; j<8; j++){
                if(j>0)
                    fprintf(file_nat_c, ",");
                fprintf(file_nat_c, "%.2f", C_native[8*i + j]);
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
                fprintf(file_nai_c, "%.2f", C_native_og[i*N + j]);
            }
            fprintf(file_nai_c, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nat = fopen("e_native.csv","w");
        printf("E native = e_native.csv\n");
        for(int i=0; i<(M*K)/8; i++){
            for (int j=0; j<8; j++){
                if(j>0)
                    fprintf(file_nat, ",");
                fprintf(file_nat, "%.2f", E_native[i*8 + j]);
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
                fprintf(file_nai, "%.2f", E_native_og[i*K + j]);
            }
            fprintf(file_nai, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nai_f = fopen("f_native.csv","w");
        printf("F native = f_native.csv\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    fprintf(file_nai_f, ",");
                fprintf(file_nai_f, "%.2f", F_native[i*N + j]);
            }
            fprintf(file_nai_f, "\n");
        }
        printf("-----------------------------------------------------\n");
        FILE *file_nat_f = fopen("f_naive.csv","w");
        printf("F naive = f_naive.csv\n");
        for(int i=0; i<M; i++){
            for (int j=0; j<N; j++){
                if(j>0)
                    fprintf(file_nat_f, ",");
                fprintf(file_nat_f, "%.2f", F_native_og[i*N + j]);
            }
            fprintf(file_nat_f, "\n");
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
        for(int j = 0; j<N3; j++){
            rel_diff = fabsf(F_native[i*N3+j] - F_native_og[i*N3+j]) / (fabsf(F_native_og[i*N3+j]) + 1e-8f);
            if (rel_diff > 1e-4)
                return 1;
        }
    }

    return 0;
}