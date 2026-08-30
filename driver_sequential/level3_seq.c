#include "interface.h"
#include "lp_prof.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#define COMPSIZE 1

// ── Constrained-case parallelism for the is-loop (M-blocking) ──────────────
// Block sizes here are data-dependent (they halve near the tail of M), so
// the sequential is-loop can't
// be turned into a plain indexed `for` for OpenMP without first materializing
// the block boundaries. compute_m_blocks() runs the identical size formula
// the sequential loop uses, sequentially and cheaply (it's O(M/GEMM_P)
// arithmetic, not compute), so the block layout -- and therefore the
// propagated-layout addressing every consumer stage depends on -- is
// bit-for-bit identical to the single-threaded case regardless of thread
// count. Only which thread computes each block changes, not where it lands.
//
// MAX_M_BLOCKS bounds the number of LAYOUT blocks (M / GEMM_P); 4096 is a
// generous margin over the ~19 the biggest real case (M=8192) needs. The
// finer PARALLEL slices are bounded by MAX_CELLS instead.
#define MAX_M_BLOCKS 4096
// Upper bound on grid cells (row-blocks x col-blocks) the weighted partition
// materialises prefix weights for, and on the 12-column B-packing chunks the
// Pre path distributes. Both are bounded in practice by ~nthreads*cells and
// by GEMM_R/GEMM_UNROLL_N respectively; the code falls back to an unweighted
// partition rather than overflowing if a shape ever exceeds this.
#define MAX_CELLS 8192

static int compute_m_blocks(long m_from, long m_to, long gemm_p_new,
                             long *offsets, long *sizes) {
    int nb = 0;
    long is = m_from;
    while (is < m_to) {
        long min_i = m_to - is;
        if (min_i >= gemm_p_new * 2) {
            min_i = gemm_p_new;
        } else if (min_i > gemm_p_new) {
            min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
        }
        assert(nb < MAX_M_BLOCKS && "increase MAX_M_BLOCKS for this workload's M");
        offsets[nb] = is;
        sizes[nb]   = min_i;
        nb++;
        is += min_i;
    }
    return nb;
}

// ── 2D thread-grid extension: N-axis counterpart to compute_m_blocks() ─────
// Real OpenBLAS parallelizes both M and N (a 2D nthreads_m x nthreads_n grid),
// not just M like this file did before. Its range_N partitioning is
// thread-count-dependent (fine for it -- every call unpacks to canonical
// layout before returning, so intermediate tiling is invisible to the
// caller). LP-GEMM cannot do that on the M axis: Pre's propagated output
// must be addressable by Mid/Pos using the exact same M-block boundaries
// regardless of how many threads ran Pre. compute_n_blocks() gives N the
// same halving-near-tail rule, rounded to GEMM_UNROLL_N (the N-direction
// register tile, matching how the kernel's own internal N chunking works)
// instead of GEMM_UNROLL_M.
static int compute_n_blocks(long n_from, long n_to, long gemm_p_new,
                             long *offsets, long *sizes) {
    int nb = 0;
    long js = n_from;
    while (js < n_to) {
        long min_j = n_to - js;
        if (min_j >= gemm_p_new * 2) {
            min_j = gemm_p_new;
        } else if (min_j > gemm_p_new) {
            min_j = ((min_j / 2 + GEMM_UNROLL_N - 1) / GEMM_UNROLL_N) * GEMM_UNROLL_N;
        }
        assert(nb < MAX_M_BLOCKS && "increase MAX_M_BLOCKS for this workload's N");
        offsets[nb] = js;
        sizes[nb]   = min_j;
        nb++;
        js += min_j;
    }
    return nb;
}

// ── Choosing the decomposition ─────────────────────────────────────────────
//
// MEASURED CONSTRAINT: refining the N axis is
// NOT free. Every cell in a (row-block x col-block) grid needs the A panel
// for its row-block packed into that thread's private buffer, so total
// A-packing work is
//
//     n_n_blocks x (M x k) bytes
//
// -- independent of the M blocking, and LINEAR in the number of N blocks.
// An earlier implementation paid that cost twice over: it re-ran
// ICOPY once per *cell* rather than once per (row-block, k-block), and it
// refined N aggressively as thread count grew. On llama32_1b_n500's Pre at
// T=24 that inflated ICOPY from 5.5 to 65.7 thread-ms -- more time packing A
// than running the micro-kernel (44.0 thread-ms).
//
// Consequences encoded below:
//   * Reuse the packed A panel across every cell in a thread's run that
//     shares an M slice (see the run over `idx` in gemm_tiling_seq).
//   * Choose the M and N split TOGETHER from a traffic model rather than
//     preferring one axis (see pick_decomposition) -- N splitting costs
//     redundant A packing, M splitting costs redundant B streaming, and
//     which dominates depends on the shape.
//
// What is legal on each axis:
//
// * N is freely splittable for all three stages. A cell's propagated write
//   address is `is_b*min_j + (js_b - js)*mi_b` with LDC=mi_b -- closed-form
//   in position and LINEAR in the column offset, never a running sum over
//   preceding blocks -- so splitting N at arbitrary boundaries yields
//   byte-identical output. (Pos's `X=is_b, Y=js_b, LDC=ldc` is ordinary
//   column-major addressing and is trivially split-invariant.)
//
// * The M *layout* blocking is NOT free for Pre/Mid: the propagated format
//   stores row-block bi as a contiguous mi_b x min_j region and uses mi_b as
//   the per-block LDC, and every downstream consumer re-derives those exact
//   boundaries from shape alone. It must stay a pure function of shape.
//
// * The M *parallel* partition IS free, for all three stages, down to
//   GEMM_UNROLL_M granularity -- see pick_decomposition. That decoupling is
//   the whole point: it is the only way to add parallelism without adding
//   traffic.

static int lp_env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    int x = atoi(v);
    return x > 0 ? x : dflt;
}

// Grid cells to aim for per thread. Larger means finer granularity and so
// better load balance across the halving rule's uneven tail blocks, at the
// cost of more redundant traffic (see pick_decomposition). Swept on the
// target at T=8/16/24 over the three representative chains: the geometric
// mean of lp_mid-vs-OpenBLAS over all nine (case, thread-count) points was
// 0.860 at 2, 0.899 at 3 and 0.930 at 4, so 4 it is.
static int lp_cells_per_thread(void) {
    static int v = -1;
    if (v < 0) v = lp_env_int("LP_CELLS_PER_THREAD", 4);
    return v;
}

// ── Decoupling the parallel M partition from the layout M blocking ─────────
//
// The layout blocking (compute_m_blocks at GEMM_P) is fixed by the data
// format: the propagated buffer stores row-block bi as a contiguous
// mi_b x min_j region, and every downstream consumer re-derives those exact
// boundaries. That constraint is on the LAYOUT, though -- not on how many
// threads may cooperate on one row-block.
//
// Within a row-block, the propagated layout places the (m16 x nr) tile
// covering rows [r, r+16) of column group j at
//
//     block_base + (r - is_b) * GEMM_UNROLL_N + j * GEMM_UNROLL_N * mi_b
//
// (the micro-kernel's own `addq $256,%2` per m16 block == 64 floats == 16
// rows x nr=4 columns). So a row-slice starting at any multiple of
// GEMM_UNROLL_M is addressable in closed form, with the block height mi_b
// still passed as LDC to keep the column stride -- exactly the case
// gemm_kernel_pre_stride already handles. That kernel is a strict
// generalisation of gemm_kernel_pre: its inter-column-group advance is
// `LDC*ndim - M*GEMM_UNROLL_N` where gemm_kernel_pre hardcodes
// `LDC*(ndim-3) - M`, and the two are algebraically identical exactly when
// LDC == M (both reduce to M*(ndim-4)). Pre/Mid therefore always dispatch to
// the _stride kernel now, which costs nothing when a slice happens to span a
// whole block and makes sub-block slicing legal when it does not.
//
// This is what real OpenBLAS does: level3_thread.c partitions M among
// threads at GEMM_UNROLL_M granularity, completely independently of its mc
// cache blocking. It matters because it is the ONLY way to add M-direction
// parallelism for free -- total A-packing work is
// `n_n_blocks x (M x k)` bytes regardless of the M blocking, so splitting M
// costs nothing, while every extra N split costs a whole additional
// A-packing pass. Measured before slicing existed: llama32_1b_n500's Pre had
// to be split 10 ways on N just to fill 24 threads, which inflated ICOPY to
// 65.7 thread-ms against 44.0 thread-ms of actual micro-kernel time.
//
// Relative cost of a byte of A-packing against a byte of B-streaming, used
// by pick_decomposition's traffic model. Also swept on the target: 1 beat
// both 2 (0.893) and 4 (0.879) at the chosen cells-per-thread, so the two
// kinds of traffic are treated as equally expensive.
static int lp_apack_weight(void) {
    static int v = -1;
    if (v < 0) v = lp_env_int("LP_APACK_W", 1);
    return v;
}

// LAYOUT block size for Pre/Mid when N % GEMM_UNROLL_N != 0 (see the
// gemm_p_new comment above for why this case can't use sub-slicing). Deliberately
// NOT scaled by thread count -- must stay a pure function of shape. 8x
// GEMM_UNROLL_M gives ceil(M/128) blocks; for the dataset's M=8192 shapes
// that's 64 blocks, comfortably more than the 24 threads tested, leaving
// headroom for the halving-near-tail rule's uneven last block without going
// so fine that per-block ICOPY overhead starts to dominate.
static int lp_narrow_n_block(void) {
    static int v = -1;
    if (v < 0) {
        // Swept on the target at T=8/16/24 on the two N=1 shapes: 2x
        // GEMM_UNROLL_M (32) beat 4x (64), 8x (128, the original guess),
        // and both directions further out -- 1x (16) was WORSE than 32 too
        // (more blocks than this helps with is possible: per-block ICOPY
        // overhead starts to dominate), so this is a real interior optimum,
        // not "smaller is always better". At 32, achieved parallelism on
        // llama32_1b_dec/vgg16_fc_n1 went from 6.6x/8.0x to 19.4x/21.1x at
        // T=24 (~3x), cutting wall-clock time roughly in half.
        v = lp_env_int("LP_NARROW_N_BLOCK", GEMM_UNROLL_M * 2);
        // MUST be a multiple of GEMM_UNROLL_M. The packing kernel (ICOPY)
        // rounds a block's row count UP to the nearest full m16 tile
        // internally -- pick_decomposition's own hs already accounts for
        // this (`hs = ceil(hs/GEMM_UNROLL_M)*GEMM_UNROLL_M`), but this knob
        // is user-supplied via env var and was not being rounded, so an
        // unaligned value (e.g. 24) sized the per-thread packing panel
        // (gemm_p_new*gemm_q_new floats) for the REQUESTED row count while
        // ICOPY wrote the RETURNED, rounded-up one -- a real heap overflow.
        // Confirmed by direct repro + gdb: at LP_NARROW_N_BLOCK=24 on
        // vgg16_fc_n1's Mid stage (T=16), every thread hung forever inside
        // libgomp's end-of-region barrier, consistent with the overflow
        // corrupting gomp's own team/barrier state. 40 (also unaligned)
        // did not visibly hang in one test run but is equally out of
        // bounds by the same reasoning -- treat as equally unsafe, not as
        // evidence it was fine; overflow symptoms are layout-dependent.
        v = ((v + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
        if (v < GEMM_UNROLL_M) v = GEMM_UNROLL_M;
    }
    return v;
}

static long lp_isqrt(long v) {
    if (v <= 0) return 0;
    long r = 1;
    while (r * r < v) r++;
    return (r * r > v) ? r - 1 : r;
}

// Choose BOTH axes of the decomposition together, from a traffic model.
//
// With `s` M-slices and `n` N-blocks, and s*n == C cells fixed by the thread
// count, the two thread-count-dependent traffic terms are
//
//     A-packing  ~  n * M * kc          (each N block repacks all of A)
//     B-streaming~  s * N * kc          (each M slice restreams the B panel)
//
// so total ~ w*n*M + (C/n)*N, minimised at n = sqrt(C*N/(w*M)). Neither axis
// is universally preferable: for llama32_1b_n500's Mid (M=8192, N=500) the
// model picks n=1 -- pure M slicing, zero redundant packing -- while for
// panel_wideshort (M=128, N=2048) it picks a wide N split, because there M is
// too short to slice and restreaming B 8x would cost far more than packing
// A a few extra times. An earlier "always refine M first" rule got the first
// case right and the second wrong (it regressed panel_wideshort's Pre).
//
// Returns the slice height and writes the wanted N-block count. The slice
// height is rounded to GEMM_UNROLL_M so every slice start is a whole m16 tile
// boundary; a block's ragged tail, if the shape has one, always lands inside
// its LAST slice, where the kernel's own m8/m4/m2/m1 tails place it exactly
// where the unsliced block would have.
static long pick_decomposition(long span_m, long span_n, int nthreads,
                               long *want_nb) {
    long C = (long)nthreads * lp_cells_per_thread();
    if (nthreads <= 1 || span_m <= 0 || span_n <= 0 || C < 1) {
        *want_nb = 1;
        return GEMM_P;
    }
    long w  = lp_apack_weight();
    long nn = lp_isqrt((C * span_n + w * span_m - 1) / (w * span_m));
    if (nn < 1) nn = 1;
    if (nn > C) nn = C;
    long ns = C / nn;
    if (ns < 1) ns = 1;
    long hs = (span_m + ns - 1) / ns;
    hs = ((hs + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
    if (hs < GEMM_UNROLL_M) hs = GEMM_UNROLL_M;
    if (hs > GEMM_P)        hs = GEMM_P;
    *want_nb = nn;
    return hs;
}

// N tile. With M sliceable down to GEMM_UNROLL_M, the M axis can nearly
// always supply every thread on its own, so this returns the whole span (one
// N block, zero redundant A-packing) unless M genuinely cannot -- which now
// only happens for very small M.
static long pick_n_tile(long span, int n_mblocks, int nthreads) {
    if (nthreads <= 1 || n_mblocks <= 0 || span <= 0) return GEMM_P;
    long target = (long)nthreads * lp_cells_per_thread();
    if (n_mblocks >= target) return span;   // M alone already feeds every thread
    long want_nchunks = (target + n_mblocks - 1) / n_mblocks;
    if (want_nchunks < 1) want_nchunks = 1;
    long tile = (span + want_nchunks - 1) / want_nchunks;
    tile = ((tile + GEMM_UNROLL_N - 1) / GEMM_UNROLL_N) * GEMM_UNROLL_N;
    if (tile < GEMM_UNROLL_N)  tile = GEMM_UNROLL_N;
    return tile;
}

// Split [0,C) into `nt` CONTIGUOUS runs of near-equal weight, and return
// run `tid`. Contiguity is what preserves the A-panel reuse and the stable
// cell->thread mapping (see lp_range below); weighting is what stops the
// halving rule's uneven tail blocks -- e.g. M=8192 gives seventeen 448-row
// blocks then two 288-row ones -- from handing one thread up to 1.5x its
// share. Measured before this was added: the busiest thread carried 1.6-1.9x
// the mean across every stage at T=24, which was the whole remaining gap once
// the packing amplification was gone.
static inline void lp_range_w(const long *pw, long C, int tid, int nt,
                              long *lo, long *hi) {
    long total = pw[C];
    if (total <= 0) { *lo = *hi = 0; return; }
    long bound[2];
    for (int e = 0; e < 2; e++) {
        int t = tid + e;
        if (t == 0)       { bound[e] = 0; continue; }
        if (t >= nt)      { bound[e] = C; continue; }
        long want = (long)(((double)t * (double)total) / (double)nt);
        long i = 0, j = C;                       // lower_bound(pw, want)
        while (i < j) { long m = (i + j) >> 1; if (pw[m] < want) i = m + 1; else j = m; }
        bound[e] = i;
    }
    *lo = bound[0]; *hi = bound[1];
    if (*hi < *lo) *hi = *lo;
}

// Static, contiguous partition of [0,total) over nt threads. Contiguous (not
// round-robin) for two reasons: a thread's cells then mostly share a
// row-block, so the A-panel reuse below actually fires; and the same cell
// index maps to the same thread in every k-block iteration, so the C region
// a thread accumulates into stays resident in that core's private L2 for the
// whole call instead of migrating between cores at every k-block boundary.
static inline void lp_range(long total, int tid, int nt, long *lo, long *hi) {
    long q = total / nt, r = total % nt;
    *lo = (long)tid * q + (tid < r ? tid : r);
    *hi = *lo + q + (tid < r ? 1 : 0);
}

// ── Per-thread packing panels ──────────────────────────────────────────────
// Root cause of the bug documented in this file's git history (search
// "DISABLED" in the pre-rewrite version): giving each thread a panel that is
// an OFFSET into the shared sa/sb scratch block (sa + tid*panel_floats) meant
// that any overrun past a thread's nominal panel bound -- from the
// kernel/icopy reading or writing slightly more than panel_floats, which
// apparently happens for some (M,K,N) combinations -- silently corrupted an
// ADJACENT thread's panel, or corrupted `sb` itself while other threads were
// still reading it. Verified experimentally: giving each thread a fully
// independent malloc()'d buffer instead makes the exact same failing shapes
// bit-identical. This does not identify the exact line that overruns (a real
// fix would), but removes the failure mode it exploits -- so the panels stay
// SEPARATE allocations here, deliberately, and must not be merged into one.
//
// CACHED ACROSS CALLS (this is a change): they used to be malloc'd and freed
// inside every gemm_tiling_seq() call. Measured on the target, that showed up
// as up to 2.5ms of malloc/free per call at T=24 plus a much larger secondary
// cost -- freshly mmap'd panels fault on first touch, which lands inside
// ICOPY/OCOPY and inflated panel_wideshort's Pre OCOPY from 1.4 to 21.5
// thread-ms. Allocated on demand, grown as needed, released at exit. Only
// ever called from the serial region before `#pragma omp parallel`, so it
// needs no locking of its own.
static float **g_panels     = NULL;
static int     g_panels_n   = 0;
static long    g_panels_sz  = 0;

static void lp_release_panels(void) {
    if (!g_panels) return;
    for (int t = 0; t < g_panels_n; t++) free(g_panels[t]);
    free(g_panels);
    g_panels = NULL; g_panels_n = 0; g_panels_sz = 0;
}
__attribute__((destructor)) static void lp_panels_dtor(void) { lp_release_panels(); }

static float **get_thread_panels(int nthreads, long panel_floats) {
    if (g_panels && g_panels_n >= nthreads && g_panels_sz >= panel_floats)
        return g_panels;
    lp_release_panels();
    g_panels_n  = nthreads;
    g_panels_sz = panel_floats;
    g_panels    = malloc((size_t)nthreads * sizeof(float *));
    for (int t = 0; t < nthreads; t++) {
        g_panels[t] = malloc((size_t)panel_floats * sizeof(float));
        // Fault the pages in now, once, on the allocating thread rather than
        // lazily inside the first ICOPY of the first parallel region.
        memset(g_panels[t], 0, (size_t)panel_floats * sizeof(float));
    }
    return g_panels;
}

#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC, STRIDE) \
	GEMM_BETA((M_TO) - (M_FROM), ((N_TO - N_FROM) * STRIDE), 0, \
    BETA, NULL, 0, NULL, 0, \
    (float *)(C) + ((M_FROM) + (N_FROM) * (LDC)), LDC)

#define A	args -> a
#define LDA	args -> lda
#define B	args -> b
#define LDB	args -> ldb
#define C	args -> c
#define LDC	args -> ldc
#define M	args -> m
#define N	args -> n
#define K	args -> k

int gemm_tiling_seq(arg_t *args, long *range_m, long *range_n, float *sa, float *sb,
           copy_op_func_t ICOPY_OPERATION,
           copy_op_func_t OCOPY_OPERATION,
           kernel_op_func_t KERNEL_OPERATION,
           const enum CBLAS_SEQ Seq,
           int TileN, int StrideN){
    long k, lda, ldb, ldc;
    float alpha, beta;
    float *a, *b;
    float *c;
    long m_from, m_to, n_from, n_to;
    #ifdef DEBUG
    FILE *file_nat_c = fopen("b_pack.csv","w");
    #endif

    long ls, is, js;
    long min_l, min_i, min_j;
    long jjs, min_jj;

    long l1stride, gemm_p, l2size;

    k = K;

    a = (float *)A;
    b = (float *)B;
    c = (float *)C;

    lda = LDA;
    ldb = LDB;
    ldc = LDC;

    alpha = args -> alpha;
    beta  = args -> beta;

    m_from = 0;
    m_to   = M;

    if (range_m) {
        m_from = *(((long *)range_m) + 0);
        m_to   = *(((long *)range_m) + 1);
    }

    n_from = 0;
    n_to   = N;

    if (range_n) {
        n_from = *(((long *)range_n) + 0);
        n_to   = *(((long *)range_n) + 1);
    }

    // beta application, done ONCE here (before any is-block writes, for
    // either the sequential or parallel path below) rather than per-block or
    // per-ls-iteration: standard GotoBLAS design is that the caller-facing
    // wrapper applies beta exactly once, and every subsequent kernel call
    // accumulates onto that (there is no per-call beta parameter threaded
    // into KERNEL_OPERATION anywhere in this file -- accumulation is the
    // kernel's only mode).
    //
    // Restricted to StrideN == 0 ONLY after a real regression: the actual
    // caller in ../simple_attention (main_validation.cpp's
    // multi_head_attention_blas, CblasMid with nonzero StrideN) passes a `c`
    // pointer offset into the MIDDLE of a much larger buffer (per-head
    // output slicing) together with an outer `ldc` that does NOT match the
    // stride the kernel actually writes with internally. Zeroing the full
    // region with the outer `ldc` in that situation overruns the real buffer.
    if (beta != 1 && !StrideN) {
        BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc, 1);
    }

    if ((k == 0) || (alpha == 0)) return 0;

    l2size = GEMM_P * GEMM_Q;

    // gemm_p_new is the LAYOUT M tile -- the block size every consumer of a
    // propagated buffer re-derives. It must stay a pure function of shape
    // (never of thread count -- Mid/Pos must independently land on the same
    // boundaries Pre used, and nothing here checks that every stage of a
    // chain actually ran with the same T). TileN (badly named: it is the M
    // tile override) wins when the caller sets it.
    //
    // Narrow-N exception: sub-block M-SLICING (pick_decomposition, guarded by
    // can_slice below) assumes a uniform GEMM_UNROLL_N-wide row stride within
    // a layout block, which only holds when every n-group the kernel emits
    // is a full GEMM_UNROLL_N wide -- false whenever N % GEMM_UNROLL_N != 0
    // (SAVE_h_m16n1 advances 64B/row-block, not the 256B slicing assumes).
    // For those shapes (N=1 GEMV-shaped links, chiefly) slicing is disabled
    // entirely, capping usable parallel units at ceil(M/GEMM_P) regardless of
    // thread count -- measured directly: llama32_1b_dec's Mid reached only
    // 6.6x/24 achieved parallelism, a ~19-block ceiling for M=8192.
    //
    // Fix used here is NOT sub-slicing (which is what breaks): make the
    // LAYOUT block itself smaller for this case, so compute_m_blocks
    // produces more, but still individually UNSLICED, blocks. Each stays
    // addressed as `is_b*min_j` (r0 == is_b always, since hs stays
    // GEMM_P+1's "never split further" sentinel below) -- the exact formula
    // already proven correct for arbitrary N by the exhaustive sweep, just
    // at finer layout granularity. This sidesteps the row-stride bug rather
    // than fixing it, at zero kernel/assembly risk. Kept thread-count-
    // INDEPENDENT (LP_NARROW_N_BLOCK is a fixed constant, not derived from
    // nthreads) so the "pure function of shape" invariant stays exactly as
    // strong as it already was -- this does not lean on the same-T-across-
    // the-chain contract any harder than the unmodified GEMM_P path already
    // does.
    int nthreads = omp_get_max_threads();
    long gemm_p_new;
    if (TileN != -1) {
        gemm_p_new = TileN;
    } else if (Seq != CblasPos && !StrideN && (N % GEMM_UNROLL_N != 0)) {
        gemm_p_new = lp_narrow_n_block();
    } else {
        gemm_p_new = GEMM_P;
    }
    long gemm_q_new = GEMM_Q;

    long m_blk_off[MAX_M_BLOCKS], m_blk_sz[MAX_M_BLOCKS];
    long n_blk_off[MAX_M_BLOCKS], n_blk_sz[MAX_M_BLOCKS];
    long cell_pw[MAX_CELLS + 1];   // prefix cell weights for lp_range_w
    long ch_off[MAX_CELLS], ch_sz[MAX_CELLS];   // Pre's B-packing chunk list
    // M slices: the parallel partition. sl_r0/sl_h are the slice's own rows;
    // sl_ldc is the LAYOUT block height it lives in (passed to the kernel as
    // LDC so the propagated column stride is the block's, not the slice's);
    // sl_xb is the slice's flat write offset within the panel, excluding the
    // column-block term which is added per cell.
    // (These plus the block/chunk/weight arrays are ~0.6MB of stack. That is
    // fine for the master thread, which is the only caller -- gemm_seq_buf is
    // not re-entrant from inside a parallel region.)
    long sl_r0[MAX_CELLS], sl_h[MAX_CELLS], sl_ldc[MAX_CELLS], sl_xb[MAX_CELLS];
    float **panels = NULL;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        int n_mb = compute_m_blocks(m_from, m_to, gemm_p_new, m_blk_off, m_blk_sz);

        // Build the slice list. StrideN keeps one slice per layout block:
        // that path's addressing (plain row offset `is_b`, LDC = StrideN) was
        // verified against the real attention caller and is deliberately left
        // exactly as it was.
        //
        // Sub-block slicing is legal for Pre/Mid only when every n-group the
        // micro-kernel will emit is at least GEMM_UNROLL_N wide. The row
        // stride inside a group is the group width CAPPED at GEMM_UNROLL_N,
        // not GEMM_UNROLL_N unconditionally: SAVE_h_m16n1 advances the C
        // pointer by 64 bytes per m16 block (1 float/row), SAVE_h_m16n2 by
        // 128 (2 floats/row), and only SAVE_h_m16n4 and wider by 256 (4
        // floats/row). A panel whose width is a multiple of GEMM_UNROLL_N
        // decomposes entirely into groups of 24/20/16/12/8/4, leaving no
        // narrow trailing group, so a single (r0-is_b)*GEMM_UNROLL_N offset
        // is exact. A width like N=1 (llama32_1b_dec, decode) does not --
        // caught by the exhaustive sweep, which failed there with
        // rel=1.26 before this guard. Pos writes canonical column-major
        // output and is always sliceable.
        int can_slice = (Seq == CblasPos) || (min_j % GEMM_UNROLL_N == 0);
        int n_sl = 0;
        long want_nb = 1;
        long hs;
        if (StrideN || nthreads <= 1 || !can_slice) {
            hs = GEMM_P + 1;                          /* never splits */
        } else {
            hs = pick_decomposition(m_to - m_from, min_j, nthreads, &want_nb);
        }
        for (int bi = 0; bi < n_mb && n_sl < MAX_CELLS; bi++) {
            long is_b = m_blk_off[bi], mi_b = m_blk_sz[bi];
            for (long r0 = is_b; r0 < is_b + mi_b && n_sl < MAX_CELLS; r0 += hs) {
                long h = is_b + mi_b - r0;
                if (h > hs) h = hs;
                sl_r0[n_sl]  = r0;
                sl_h[n_sl]   = h;
                sl_ldc[n_sl] = mi_b;
                sl_xb[n_sl]  = (Seq == CblasPos) ? r0
                             : (StrideN)         ? is_b
                             : is_b * min_j + (r0 - is_b) * GEMM_UNROLL_N;
                n_sl++;
            }
        }

        int n_nb;
        if (StrideN) {
            // StrideN's addressing was verified specifically for the M-only
            // case against the real caller (simple_attention/
            // main_validation.cpp). Keep that path on exactly one N block so
            // it never enters an N-split decomposition it was not verified
            // for.
            n_blk_off[0] = js; n_blk_sz[0] = min_j; n_nb = 1;
        } else {
            long ntile;
            if (nthreads <= 1 || !can_slice) {
                // No M slicing available: fall back to sizing the N split off
                // the layout-block count alone.
                ntile = pick_n_tile(min_j, n_sl, nthreads);
            } else {
                ntile = (min_j + want_nb - 1) / want_nb;
                ntile = ((ntile + GEMM_UNROLL_N - 1)/GEMM_UNROLL_N)*GEMM_UNROLL_N;
                if (ntile < GEMM_UNROLL_N) ntile = GEMM_UNROLL_N;
            }
            n_nb = compute_n_blocks(js, js + min_j, ntile, n_blk_off, n_blk_sz);
        }

        int go_parallel = (nthreads > 1) && ((long)n_sl * n_nb > 1);

        // The parallel Pre path packs the whole js-panel of B into `sb` at
        // stride-1 column-block offsets so that every cell can read its own
        // slice concurrently. The sequential path only needs that footprint
        // when l1stride==1 (it collapses to a single reused chunk otherwise),
        // so guard explicitly rather than assume a caller-supplied sb is
        // large enough. Falls back to the sequential path if it is not.
        if (go_parallel && Seq == CblasPre &&
            (long)gemm_q_new * min_j * (long)sizeof(float) >
                (long)BUFFER_SIZE - GEMM_SB_OFFSET_BYTES)
            go_parallel = 0;

        // B-packing chunk list for Pre, built once per js-panel rather than
        // per k-block: the chunk boundaries depend only on min_j. Chunked over
        // the WHOLE js-panel (not per column-block) so the packing phase can
        // be spread over every thread even when the compute grid has only a
        // few column-blocks -- otherwise, with n_nb small, one thread packs
        // all of B while the rest sit at the barrier. The byte layout is
        // unaffected: every chunk width is a multiple of GEMM_UNROLL_N, so a
        // concatenation of chunks is the same sequence of packed nr-column
        // groups as one wide pack, exactly as the sequential path produces.
        int n_ch = 0;
        if (go_parallel && Seq == CblasPre) {
            long jj = js;
            while (jj < js + min_j) {
                long mjj = js + min_j - jj;
                if (mjj >= 3*GEMM_UNROLL_N)   mjj = 3*GEMM_UNROLL_N;
                else if (mjj > GEMM_UNROLL_N) mjj = GEMM_UNROLL_N;
                if (n_ch >= MAX_CELLS) { go_parallel = 0; break; }
                ch_off[n_ch] = jj; ch_sz[n_ch] = mjj; n_ch++;
                jj += mjj;
            }
        }

        // Prefix weights over the row-major-flattened cell grid, for the
        // weight-balanced contiguous partition. Falls back to an equal-count
        // partition if the grid is somehow larger than the prefix array.
        long n_cells = (long)n_sl * n_nb;
        int  weighted = go_parallel && (n_cells <= MAX_CELLS);
        if (weighted) {
            cell_pw[0] = 0;
            for (long idx = 0; idx < n_cells; idx++)
                cell_pw[idx + 1] = cell_pw[idx]
                                 + sl_h[idx / n_nb] * n_blk_sz[idx % n_nb];
        }

        if (go_parallel) {
            if (!panels)
                panels = get_thread_panels(nthreads, gemm_p_new * gemm_q_new);

            // ONE parallel region for the entire k-loop, not one per k-block.
            // Measured: llama32_1b_n500's Pos (K=8192 -> 19 k-blocks) spent
            // 14.2 of its 27.9ms wall at T=24 outside any counted work, with
            // 19 fork/joins under OMP_WAIT_POLICY=passive (which the harness
            // must use, so every fork re-wakes 24 sleeping threads). Hoisting
            // also makes the cell->thread mapping
            // stable across k-blocks, which keeps each thread's slice of C in
            // its own L2 for the whole accumulation.
            //
            // Mid and Pos need NO synchronization inside at all: cells write
            // disjoint regions of C, read A and B read-only, and each thread
            // owns its packing panel. Pre needs two barriers per k-block
            // because all threads share the packed-B panel in `sb`.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                float *sa_t = panels[tid];
                long c0, c1, q0, q1;
                if (weighted) lp_range_w(cell_pw, n_cells, tid, nt, &c0, &c1);
                else          lp_range(n_cells, tid, nt, &c0, &c1);
                lp_range(n_ch, tid, nt, &q0, &q1);   // B-packing chunks (Pre)

                long l_ls, l_min_l = 0;
                for (l_ls = 0; l_ls < k; l_ls += l_min_l) {
                    l_min_l = k - l_ls;
                    if (l_min_l >= gemm_q_new * 2) {
                        l_min_l = gemm_q_new;
                    } else if (l_min_l > gemm_q_new) {
                        l_min_l = ((l_min_l / 2 + GEMM_UNROLL_M - 1)
                                   / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                    }

                    if (Seq == CblasPre) {
                        // Phase 1: pack this thread's share of B into the
                        // shared sb, at the same stride-1 offsets the
                        // sequential path uses. Chunks are disjoint, so this
                        // is race-free without any lock; the following barrier
                        // is what makes it visible to the cells that read it.
                        for (long q = q0; q < q1; q++)
                            OCOPY_OPERATION(l_min_l, ch_sz[q], b, ldb, l_ls, ch_off[q],
                                            sb + l_min_l * (ch_off[q] - js) * COMPSIZE);
                        #pragma omp barrier
                    }

                    // Phase 2: this thread's contiguous run of cells. ICOPY
                    // fires only when the M slice changes, so the packed A
                    // panel is reused across every cell in the run that
                    // shares it -- the single biggest source of the old
                    // path's work amplification.
                    int last_si = -1;
                    for (long idx = c0; idx < c1; idx++) {
                        int  si   = (int)(idx / n_nb);
                        int  bj   = (int)(idx % n_nb);
                        long r0   = sl_r0[si],  h    = sl_h[si];
                        long ldcv = sl_ldc[si], xb   = sl_xb[si];
                        long js_b = n_blk_off[bj], nj_b = n_blk_sz[bj];

                        if (si != last_si) {
                            ICOPY_OPERATION(l_min_l, h, a, lda, l_ls, r0, sa_t);
                            last_si = si;
                        }

                        if (Seq == CblasPos) {
                            KERNEL_OPERATION(h, nj_b, l_min_l, alpha, sa_t,
                                             b + l_ls * n_to + js_b * l_min_l, c,
                                             ldc, xb, js_b);
                        } else if (Seq == CblasMid) {
                            // kernel_op_pre_stride computes its write address
                            // as `C + (X + Y*LDC)` -- Y is NOT a free slot. The
                            // complete column offset for this cell is already
                            // composed into X, so Y must stay the panel-index
                            // term `js`, never js_b (that would double-count
                            // it).
                            KERNEL_OPERATION(h, nj_b, l_min_l, alpha, sa_t,
                                             b + l_ls * n_to + js_b * l_min_l, c,
                                             (StrideN) ? StrideN : ldcv,
                                             (StrideN) ? xb : xb + (js_b - js)*ldcv,
                                             js);
                        } else { /* CblasPre */
                            KERNEL_OPERATION(h, nj_b, l_min_l, alpha, sa_t,
                                             sb + l_min_l * (js_b - js) * COMPSIZE, c,
                                             (StrideN) ? StrideN : ldcv,
                                             (StrideN) ? xb : xb + (js_b - js)*ldcv,
                                             js / GEMM_R * m_to);
                        }
                    }

                    if (Seq == CblasPre) {
                        // Nothing may start overwriting sb for the next
                        // k-block until every cell has finished reading it.
                        #pragma omp barrier
                    }
                }
            }
            continue;
        }

        // ── Sequential path (T==1, or a shape with nothing to distribute) ──
        // Left structurally as it was: this is the already-verified code, and
        // keeping it byte-for-byte identical is what guarantees the parallel
        // path above can be checked against it.
        for (ls = 0; ls < k; ls += min_l) {

            min_l = k - ls;

            if (min_l >= gemm_q_new * 2) {
                min_l  = gemm_q_new;
            } else {
                if (min_l > gemm_q_new) {
                    min_l = ((min_l / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }
                gemm_p = ((l2size / min_l + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                while (gemm_p * min_l > l2size) gemm_p -= GEMM_UNROLL_M;
                (void)gemm_p;
            }

            long pad_min_l = min_l;

            /* First, we have to move data A to L2 cache */
            min_i = m_to - m_from;
            l1stride = 1;

            if (min_i >= gemm_p_new * 2) {
                min_i = gemm_p_new;
            } else {
                if (min_i > gemm_p_new) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                } else {
                    l1stride = 0;
                }
            }

            if (Seq == CblasPre) {
                icopy_s = clock();
                ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa);
                icopy_e = clock();
                icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;

                for(jjs = js; jjs < js + min_j; jjs += min_jj){
                    min_jj = min_j + js - jjs;
                    if (min_jj >= 3*GEMM_UNROLL_N) min_jj = 3*GEMM_UNROLL_N;
                    else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                    ocopy_s = clock();
                    OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, sb + pad_min_l * (jjs - js) * COMPSIZE * l1stride);
                    ocopy_e = clock();
                    ocopy += (double)(ocopy_e - ocopy_s) / CLOCKS_PER_SEC * 1000;

                    kernel_s = clock();
                    KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa, sb + pad_min_l * (jjs - js) * COMPSIZE * l1stride, c, (StrideN) ? StrideN : min_i, m_from, jjs);
                    kernel_e = clock();
                    kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;
                    #ifdef DEBUG
                    for (int j=0; j<min_jj*min_l; j++){
                        if(j % 4 == 0 && j > 0)
                            fprintf(file_nat_c, "\n");
                        else if (j > 0)
                            fprintf(file_nat_c, ",");
                        fprintf(file_nat_c, "%.8f", sb[pad_min_l * (jjs - js)  * COMPSIZE * l1stride + j]);
                    }
                    fprintf(file_nat_c, "\n");
                    #endif
                }

                for(is = m_from + min_i; is < m_to; is += min_i){
                    min_i = m_to - is;

                    if (min_i >= gemm_p_new * 2) {
                        min_i = gemm_p_new;
                    } else if (min_i > gemm_p_new) {
                        min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                    }

                    icopy_s = clock();
                    ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
                    icopy_e = clock();
                    icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;

                    kernel_s = clock();
                    KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c,
                                      (StrideN) ? StrideN : min_i,
                                      (StrideN) ? is : is*min_j,
                                      js/GEMM_R * m_to);
                    kernel_e = clock();
                    kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;
                } /* end of is */
            } else {
                for(is = m_from; is < m_to; is += min_i){
                    min_i = m_to - is;

                    if (min_i >= gemm_p_new * 2) {
                        min_i = gemm_p_new;
                    } else if (min_i > gemm_p_new) {
                        min_i = ((min_i / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                    }

                    icopy_s = clock();
                    ICOPY_OPERATION(min_l, min_i, a, lda, ls, is, sa);
                    icopy_e = clock();
                    icopy += (double)(icopy_e - icopy_s) / CLOCKS_PER_SEC * 1000;

                    kernel_s = clock();
                    if (Seq == CblasPos) {
                        KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, b + ls * n_to + js * min_l, c, ldc, is, js);
                    } else {
                        KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, b + ls * n_to + js * min_l, c,
                                          (StrideN) ? StrideN : min_i,
                                          (StrideN) ? is : is*min_j,
                                          js);
                    }
                    kernel_e = clock();
                    kernel += (double)(kernel_e - kernel_s) / CLOCKS_PER_SEC * 1000;
                } /* end of is */
            }
        } /* end of ls */
    } /* end of js */

    return 0;
}
