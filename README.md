# GEMMBench

A lightweight benchmarking framework for evaluating custom GEMM microkernels.

---

This repository benchmarks the microkernel implementation of GEMM operations based on a lightweight version of the OpenBLAS repository.

OpenBLAS is an open-source Basic Linear Algebra Subprogram (BLAS) implementation that tiles the GEMM operation based on hardware-related information of the target architectures, providing performance comparable to BLAS implementations from architecture vendors, such as Intel MKL.

GEMMVBench removes all other level 1, 2, and 3 operations except for GEMM, refactoring it to eliminate the boilerplate code present in OpenBLAS to support multiple targets. However, it carefully maintains the same file dependencies and tiling strategy as the original repository, facilitating future integration of the evaluated kernel into OpenBLAS.

## Dependencies

Depending on the selected kernel, a specific compiler and simulator may be required to evaluate the benchmark.

For RISC-V, the main target architecture of this repository, the [RISC-V toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) with LLVM Clang and QEMU-RISCV64 simulator is needed.

## Repository Description

```bash
GEMMVBench
|---- main.c
|---- Makefile
|---- Scripts
|       |---- run_file.sh
|---- Dataset
|       |---- small_gemm.csv
|       |---- medium_gemm.csv
|       |---- large_gemm.csv
|---- driver
|       |---- interface.c/interface.h (similar to gemm.c in OpenBLAS)
|       |---- level3.c (similar to level3.c in OpenBLAS)
|---- kernel
|       |---- riscv64
|       |       |---- parameter.yaml (similar to param.h)
|       |       |---- gemm_icopy.c
|       |       |---- gemm_ocopy.c
|       |       |---- gemm_kernel_16x8.c
|       |       |---- gemm_beta.c
```

The `main.c` file contains the main program, responsible for instantiating and populating the matrices, calling the matrix multiplication primitives, and comparing the results between the version using the `Native` SIMD and a naive implementation.

The `dataset` directory contains files in CSV (Comma-Separated Values) format, listing the GEMM problem sizes to be evaluated. Each line must specify the matrix sizes (M, N, K) and the parameters α and β.

Three different GEMM contexts are planned to be evaluated by GEMMBench, *i.e.*, matrix multiplication operations derived from Convolutions, NLP Transformers, and HPC workloads:

- **Convolution:** Convolution operations from real-world Convolutional Neural Networks were transformed into GEMM operations using the IM2COL routine and then categorized into medium (<4 MB) and large (>4 MB) datasets.
- **NLP Transformers:** TO BE DONE.
- **HPC:** TO BE DONE.

The `driver` directory includes a simplified version of OpenBLAS, containing the files `interface.c` and `level3.c`. The `interface.c` file implements the standard GEMM API, initializing auxiliary structures, creating intermediate buffers, and calling the tiling routine in `level3.c`. The `level3.c` file essentially retains the same code as adopted by OpenBLAS.

The `kernel` directory contains the developed kernels. This repository includes a `gemm_kernel.c` kernel file, the packing routines `gemm_icopy.c` and `gemm_ocopy.c`, as well as the tiling parameterization defined in the `parameter.yaml` file, used by the slicing routine in `level3.c`.

Although GEMMBench follows a structure similar to OpenBLAS, it is more compact and contains fewer lines of code, making it easier to experiment with different configurations and reducing compilation time.

## How to use

First, compile the code using the available rules in the `Makefile`. Adjust the Makefile path to match your environment. Then, run the following command:

`make KERNELDIR=<kernel/dir> all`

To verify if the compilation was successful, run:

`make run`

After that, select a set of GEMM operations from the dataset folder and execute them with the following command:

`./scripts/run_file.sh <gemmbench binary> <dataset/context/operation_set.csv>`

This script extensively evaluates the actual kernel implementations based on a list of operations described in a CSV file, returning the number of successful and failed cases at the end.

## Profiling directions

GEMMBench provides an easy way to experiment with GEMM kernels for RISC-V Matrix Multiply Units still in development, usually with no target architecture available. In these scenarios, profiling and performance analysis depend heavily on the simulator implementation.

A common way to measure performance in real instances is to insert time measurement clauses to start and stop the timer around execution regions of interest. However, in simulation, the time and cycle counts usually taken from architectural registers can vary greatly depending on the simulator’s time settings. Moreover, if newly added instructions are not correctly tuned, they may be considered as taking only one cycle to execute, which is not accurate by real-world standards.

That's why analysis from simulators usually relies on post-mortem performance analysis from program traces, assigning average latencies to different kinds of instructions.

In this context, the directions on how and where to insert profiling calls in the GEMMBench source code are given below.

1. Profiling the whole GEMMBench execution is not desirable. It runs both a `native` SIMD-optimized and a `naive` scalar version of the GEMM operation for correctness evaluation.

2. The `native` and `naive` implementations are highlighted in the [main.c](main.c) file by enclosing comments, indicating where to insert start/stop methods.

```c
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
// naive_stop();
// **** Naive implementation **** //
```

3. For the `naive` implementation, there isn’t much else to do. However, as mentioned earlier, the `native` implementation follows OpenBLAS's tiling strategy, which opens more paths for analysis.

    1. Specifically, OpenBLAS's tiling strategy requires data movement from submatrices of A and B to an intermediate contiguous buffer using the `ICOPY_OPERATION` and `OCOPY_OPERATION` macros. These macros handle pointer offsets before calling the packing procedures in  [gemm_icopy.c](kernel/riscv64/gemm_icopy.c) and [gemm_ocopy.c](kernel/riscv64/gemm_ocopy.c). One must check the time spent in packing and its impact on overall GEMM performance.

    ```c
    // **** Native implementation **** //
    // icopy_start();
    ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
    // icopy_stop();
    // **** Native implementation **** //

    [...]

    // **** Native implementation **** //
    // ocopy_start();
	OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs,
			sb + pad_min_l * (jjs - js) * COMPSIZE * l1stride);
    // ocopy_stop();
    // **** Native implementation **** //
    ```
    
    2. Then, the macro `KERNEL_OPERATION` handles pointer offsets before calling the [inner-kernel](kernel/riscv64/gemm_kernel_16x8.c) to compute the GEMM between the just-packed submatrices of A and B. In general, this kernel routine is the most important routine for analysis, providing important insights regarding memory hierarchy issues, latencies, etc.

    ```c
    // **** Native implementation **** //
    // kernel_start();
	KERNEL_OPERATION(min_i, min_jj, min_l, alpha,
			 sa, sb + pad_min_l * (jjs - js)  * COMPSIZE * l1stride, c, ldc, m_from, jjs);
    // kernel_stop();
    // **** Native implementation **** //
    ``` 

> [!NOTE]
> Given the tiling strategy used by OpenBLAS, the `COPY` and `KERNEL` routines are called many times. So the profiling methods should not overwrite previous measurements but accumulate them.

### GEM5 profiling example

TO BE DONE

## Disclaimer

> [!IMPORTANT]
> Source code from [level3.c](driver/level3.c), and [kernel/riscv64](kernel/riscv64) directory are borrowed from [OpenBLAS repository](https://github.com/OpenMathLib/OpenBLAS) with boilerplate code removed for clarity.

> [!NOTE]
> This version comes with the OpenBLAS RVV ZVL256B implementation from OpenBLAS as example. The user should add their own kernel implementation for experimentation.

> [!NOTE]
> Execution profiling such as trace extraction, instruction countings and memory-related metrics are done through simulator options and plugins.

> [!WARNING]
> Given float point type usage, rounding problems from operation order may arise. For this adjust the `EVAL_THRESHOLD` parameter of parameters.json.
