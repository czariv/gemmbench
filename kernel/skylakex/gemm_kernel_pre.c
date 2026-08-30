/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 for k_count, %5 for c_store */
/* r10 to assist prefetch, r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const), r15 = %1 + 3r12 */

#include "interface.h"
#include <stdint.h>

/* m = 16 */ /* zmm8-zmm31 for accumulators, zmm4-zmm7 for temporary use, zmm0 for alpha */
#define KERNEL_k1m16n1 \
    "vmovups (%0),%%zmm4; addq $64,%0;"\
    "vbroadcastss (%1),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,%%zmm8;"\
    "addq $4,%1;"
#define KERNEL_h_k1m16n2 \
    "vmovsldup (%0),%%zmm4; vmovshdup (%0),%%zmm5; prefetcht0 512(%0); addq $64,%0;"\
    "vbroadcastsd (%1),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,%%zmm8; vfmadd231ps %%zmm5,%%zmm6,%%zmm9;"
#define KERNEL_k1m16n2 KERNEL_h_k1m16n2 "addq $8,%1;"
#define KERNEL_h_k1m16n4 KERNEL_h_k1m16n2 "vbroadcastsd 8(%1),%%zmm7; vfmadd231ps %%zmm4,%%zmm7,%%zmm10; vfmadd231ps %%zmm5,%%zmm7,%%zmm11;"
#define KERNEL_k1m16n4 KERNEL_h_k1m16n4 "addq $16,%1;"
#define unit_kernel_k1m16n4(c1,c2,c3,c4, ...) \
    "vbroadcastsd  ("#__VA_ARGS__"),%%zmm6; vfmadd231ps %%zmm4,%%zmm6,"#c1"; vfmadd231ps %%zmm5,%%zmm6,"#c2";"\
    "vbroadcastsd 8("#__VA_ARGS__"),%%zmm7; vfmadd231ps %%zmm4,%%zmm7,"#c3"; vfmadd231ps %%zmm5,%%zmm7,"#c4";"
#define KERNEL_h_k1m16n8 KERNEL_h_k1m16n4 unit_kernel_k1m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15,%1,%%r12,1)
#define KERNEL_k1m16n8 KERNEL_h_k1m16n8 "addq $16,%1;"
#define KERNEL_h_k1m16n12 KERNEL_h_k1m16n8 unit_kernel_k1m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19,%1,%%r12,2)
#define KERNEL_k1m16n12 KERNEL_h_k1m16n12 "addq $16,%1;"
#define KERNEL_h_k1m16n16 KERNEL_k1m16n12 unit_kernel_k1m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23,%%r15)
#define KERNEL_k1m16n16 KERNEL_h_k1m16n16 "addq $16,%%r15;"
#define KERNEL_h_k1m16n20 KERNEL_h_k1m16n16 unit_kernel_k1m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27,%%r15,%%r12,1)
#define KERNEL_k1m16n20 KERNEL_h_k1m16n20 "addq $16,%%r15;"
#define KERNEL_h_k1m16n24 KERNEL_h_k1m16n20 unit_kernel_k1m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31,%%r15,%%r12,2)
#define KERNEL_k1m16n24 KERNEL_h_k1m16n24 "addq $16,%%r15;"
#define INIT_m16n1 "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define INIT_m16n2 INIT_m16n1 "vpxorq %%zmm9,%%zmm9,%%zmm9;"
#define INIT_m16n4 INIT_m16n2 "vpxorq %%zmm10,%%zmm10,%%zmm10;vpxorq %%zmm11,%%zmm11,%%zmm11;"
#define unit_init_m16n4(c1,c2,c3,c4) \
    "vpxorq "#c1","#c1","#c1";vpxorq "#c2","#c2","#c2";vpxorq "#c3","#c3","#c3";vpxorq "#c4","#c4","#c4";"
#define INIT_m16n8 INIT_m16n4 unit_init_m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15)
#define INIT_m16n12 INIT_m16n8 unit_init_m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19)
#define INIT_m16n16 INIT_m16n12 unit_init_m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23)
#define INIT_m16n20 INIT_m16n16 unit_init_m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27)
#define INIT_m16n24 INIT_m16n20 unit_init_m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31)
#define SAVE_h_m16n1 "vfmadd213ps (%2),%%zmm0,%%zmm8; vmovups %%zmm8,(%2); addq $64,%2;"
#define unit_save_m16n2(c1,c2) \
    "vshuff32x4 $0x44, "#c2","#c1", %%zmm4; vshuff32x4 $0xEE, "#c2","#c1", %%zmm5;"\
    "vpermpd %%zmm4, %%zmm1, %%zmm6; vpermpd %%zmm5, %%zmm1, %%zmm7;"\
    "vfmadd213ps (%5),%%zmm0,%%zmm6; vfmadd213ps 64(%5),%%zmm0,%%zmm7;"\
    "vmovups %%zmm6,(%5); vmovups %%zmm7,64(%5); leaq (%5,%3,2),%5;"
#define unit_save_m16n4(c1,c2,c3,c4) \
    "vshuff32x4 $0x44, "#c3","#c1", %%zmm4; vshuff32x4 $0x44, "#c4","#c2", %%zmm5;"\
    "vshuff32x4 $0xEE, "#c3","#c1", %%zmm6; vshuff32x4 $0xEE, "#c4","#c2", %%zmm7;"\
    "vmovapd %%zmm4, "#c1"; vpermt2pd %%zmm5, %%zmm3, %%zmm4; vpermt2pd "#c1", %%zmm2, %%zmm5;"\
    "vmovapd %%zmm6, "#c2"; vpermt2pd %%zmm7, %%zmm3, %%zmm6; vpermt2pd "#c2", %%zmm2, %%zmm7;"\
    "vfmadd213ps (%5),%%zmm0,%%zmm4; vfmadd213ps 64(%5),%%zmm0,%%zmm5;"\
    "vfmadd213ps 128(%5),%%zmm0,%%zmm6; vfmadd213ps 192(%5),%%zmm0,%%zmm7;"\
    "vmovups %%zmm4,(%5); vmovups %%zmm5,64(%5);"\
    "vmovups %%zmm6,128(%5); vmovups %%zmm7,192(%5); leaq (%5,%3,4),%5;"
#define SAVE_h_m16n2 "movq %2,%5;" "addq $128,%2;" unit_save_m16n2(%%zmm8,%%zmm9)
#define SAVE_h_m16n4 "movq %2,%5;"  "addq $256,%2;" unit_save_m16n4(%%zmm8,%%zmm9,%%zmm10,%%zmm11)
#define SAVE_h_m16n8  SAVE_h_m16n4 unit_save_m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15)
#define SAVE_h_m16n12 SAVE_h_m16n8 unit_save_m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19)
#define SAVE_h_m16n16 SAVE_h_m16n12 unit_save_m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23)
#define SAVE_h_m16n20 SAVE_h_m16n16 unit_save_m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27)
#define SAVE_h_m16n24 SAVE_h_m16n20 unit_save_m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31)
#define SAVE_m16(ndim) SAVE_h_m16n##ndim
#define COMPUTE_m16(ndim) \
    INIT_m16n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15; movq %2,%5; xorq %%r10,%%r10;"\
    "cmpq $16,%4; jb "#ndim"016162f;"\
    #ndim"016161:\n\t"\
    "cmpq $126,%%r10; movq $126,%%r10; cmoveq %3,%%r10;"\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    "prefetcht1 (%5); subq $63,%5; addq %%r10,%5;"\
    KERNEL_k1m16n##ndim\
    KERNEL_k1m16n##ndim\
    "prefetcht1 (%6); addq $32,%6;"\
    "subq $4,%4; cmpq $16,%4; jnb "#ndim"016161b;"\
    "movq %2,%5;"\
    #ndim"016162:\n\t"\
    "testq %4,%4; jz "#ndim"016164f;"\
    #ndim"016163:\n\t"\
    "prefetcht0 (%5); prefetcht0 63(%5); prefetcht0 (%5,%3,1); prefetcht0 63(%5,%3,1);"\
    KERNEL_k1m16n##ndim\
    "leaq (%5,%3,2),%5; decq %4; jnz "#ndim"016163b;"\
    #ndim"016164:\n\t"\
    "prefetcht0 (%%r14); prefetcht0 64(%%r14);"\
    SAVE_m16(ndim)
/* Software-pipelined m16n24 save path, adapted to the PROPAGATED layout.
 *
 * Upstream OpenBLAS pipelines this loop by deferring half of each m16 block's
 * columns into a scratch buffer and flushing them one block later, so a single
 * store covers two adjacent m16 blocks at once. That works because in the
 * canonical column-major layout consecutive m16 blocks sit 64 B apart, making
 * the natural unit of deferral one column (16 floats) and the merged store
 * 128 B.
 *
 * LP-GEMM's propagated layout instead stores each (m16 x n4) tile as 256 B
 * contiguous, with consecutive m16 blocks 256 B apart (see unit_save_m16n4 and
 * the "addq $256,%2" block advance). The pipeline carries over unchanged in
 * structure, but at n4-group granularity: three of the six n4 groups are
 * deferred (3 * 4 * 64 B = 768 B, exactly filling scr[192]) and the merged
 * store spans 512 B.
 *
 * These macros used to be the upstream column-major ones
 * used verbatim: they emitted a canonical-layout footprint and left part of the
 * propagated footprint unwritten. That was the COMPUTE_n24 bug.
 */
#define unit_save_m16n4_wscr(c1,c2,c3,c4,scr_off) \
    "vshuff32x4 $0x44, "#c3","#c1", %%zmm4; vshuff32x4 $0x44, "#c4","#c2", %%zmm5;"\
    "vshuff32x4 $0xEE, "#c3","#c1", %%zmm6; vshuff32x4 $0xEE, "#c4","#c2", %%zmm7;"\
    "vmovapd %%zmm4, "#c1"; vpermt2pd %%zmm5, %%zmm3, %%zmm4; vpermt2pd "#c1", %%zmm2, %%zmm5;"\
    "vmovapd %%zmm6, "#c2"; vpermt2pd %%zmm7, %%zmm3, %%zmm6; vpermt2pd "#c2", %%zmm2, %%zmm7;"\
    "vmovups %%zmm4,"#scr_off"(%7); vmovups %%zmm5,"#scr_off"+64(%7);"\
    "vmovups %%zmm6,"#scr_off"+128(%7); vmovups %%zmm7,"#scr_off"+192(%7);"
/* Flush the previous m16 block's deferred n4 group from scratch into the 256 B
 * immediately below the current tile, then store the current tile. c1-c4 are
 * dead once the permutes above have consumed them, so they are reused as the
 * scratch-load temporaries. */
#define unit_save_m16n4_rscr(c1,c2,c3,c4,scr_off) \
    "vshuff32x4 $0x44, "#c3","#c1", %%zmm4; vshuff32x4 $0x44, "#c4","#c2", %%zmm5;"\
    "vshuff32x4 $0xEE, "#c3","#c1", %%zmm6; vshuff32x4 $0xEE, "#c4","#c2", %%zmm7;"\
    "vmovapd %%zmm4, "#c1"; vpermt2pd %%zmm5, %%zmm3, %%zmm4; vpermt2pd "#c1", %%zmm2, %%zmm5;"\
    "vmovapd %%zmm6, "#c2"; vpermt2pd %%zmm7, %%zmm3, %%zmm6; vpermt2pd "#c2", %%zmm2, %%zmm7;"\
    "vmovups "#scr_off"(%7),"#c1"; vfmadd213ps -256(%5),%%zmm0,"#c1"; vmovups "#c1",-256(%5);"\
    "vmovups "#scr_off"+64(%7),"#c2"; vfmadd213ps -192(%5),%%zmm0,"#c2"; vmovups "#c2",-192(%5);"\
    "vmovups "#scr_off"+128(%7),"#c3"; vfmadd213ps -128(%5),%%zmm0,"#c3"; vmovups "#c3",-128(%5);"\
    "vmovups "#scr_off"+192(%7),"#c4"; vfmadd213ps -64(%5),%%zmm0,"#c4"; vmovups "#c4",-64(%5);"\
    "vfmadd213ps (%5),%%zmm0,%%zmm4; vmovups %%zmm4,(%5);"\
    "vfmadd213ps 64(%5),%%zmm0,%%zmm5; vmovups %%zmm5,64(%5);"\
    "vfmadd213ps 128(%5),%%zmm0,%%zmm6; vmovups %%zmm6,128(%5);"\
    "vfmadd213ps 192(%5),%%zmm0,%%zmm7; vmovups %%zmm7,192(%5); leaq (%5,%3,4),%5;"
/* Deferred-half bookkeeping (scratch slots 0/256/512 always hold the three n4
 * groups currently in flight):
 *   LINIT/LSAVE  store groups 0-2 at their own tile, defer groups 3-5
 *   RSAVE/RTAIL  store groups 3-5 (flushing the deferred 3-5), defer 0-2
 *   LTAIL/RTAIL  terminate the pipeline by storing their second half directly
 */
#define SAVE_p_m16n24_L_direct \
    unit_save_m16n4(%%zmm8,%%zmm9,%%zmm10,%%zmm11)\
    unit_save_m16n4(%%zmm12,%%zmm13,%%zmm14,%%zmm15)\
    unit_save_m16n4(%%zmm16,%%zmm17,%%zmm18,%%zmm19)
#define SAVE_p_m16n24_R_direct \
    unit_save_m16n4(%%zmm20,%%zmm21,%%zmm22,%%zmm23)\
    unit_save_m16n4(%%zmm24,%%zmm25,%%zmm26,%%zmm27)\
    unit_save_m16n4(%%zmm28,%%zmm29,%%zmm30,%%zmm31)
#define SAVE_p_m16n24_L_rscr \
    unit_save_m16n4_rscr(%%zmm8,%%zmm9,%%zmm10,%%zmm11,0)\
    unit_save_m16n4_rscr(%%zmm12,%%zmm13,%%zmm14,%%zmm15,256)\
    unit_save_m16n4_rscr(%%zmm16,%%zmm17,%%zmm18,%%zmm19,512)
#define SAVE_p_m16n24_R_rscr \
    unit_save_m16n4_rscr(%%zmm20,%%zmm21,%%zmm22,%%zmm23,0)\
    unit_save_m16n4_rscr(%%zmm24,%%zmm25,%%zmm26,%%zmm27,256)\
    unit_save_m16n4_rscr(%%zmm28,%%zmm29,%%zmm30,%%zmm31,512)
#define SAVE_p_m16n24_L_wscr \
    unit_save_m16n4_wscr(%%zmm8,%%zmm9,%%zmm10,%%zmm11,0)\
    unit_save_m16n4_wscr(%%zmm12,%%zmm13,%%zmm14,%%zmm15,256)\
    unit_save_m16n4_wscr(%%zmm16,%%zmm17,%%zmm18,%%zmm19,512)
#define SAVE_p_m16n24_R_wscr \
    unit_save_m16n4_wscr(%%zmm20,%%zmm21,%%zmm22,%%zmm23,0)\
    unit_save_m16n4_wscr(%%zmm24,%%zmm25,%%zmm26,%%zmm27,256)\
    unit_save_m16n4_wscr(%%zmm28,%%zmm29,%%zmm30,%%zmm31,512)
/* Shared k-loop body. The %5 walk inside it is prefetch-only -- %5 is reset
 * from %2 immediately before the saves -- so the prefetch distances inherited
 * from the canonical layout affect performance, not correctness. */
#define COMPUTE_m16n24_BODY(tag) \
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15; movq %2,%5;"\
    "cmpq $16,%4; jb " tag "2f; movq $16,%4;"\
    tag "1:\n\t"\
    KERNEL_k1m16n24 "addq $4,%4; testq $12,%4; movq $172,%%r10; cmovz %3,%%r10;"\
    KERNEL_k1m16n24 "prefetcht1 -64(%5); leaq -129(%5,%%r10,1),%5;"\
    KERNEL_k1m16n24 "prefetcht1 (%6); addq $32,%6; cmpq $208,%4; cmoveq %2,%5;"\
    KERNEL_k1m16n24 "cmpq %4,%%r13; jnb " tag "1b;"\
    "movq %2,%5; negq %4; leaq 16(%%r13,%4,1),%4;"\
    tag "2:\n\t"\
    "testq %4,%4; jz " tag "4f; movq %7,%%r10;"\
    tag "3:\n\t"\
    "prefetcht0 -64(%5); prefetcht0 (%5); prefetcht0 63(%5); addq %3,%5;"\
    KERNEL_k1m16n24 "prefetcht0 (%%r10); addq $64,%%r10; decq %4; jnz " tag "3b;"\
    tag "4:\n\t"\
    "prefetcht0 (%%r14); prefetcht0 64(%%r14); movq %2,%5; addq $256,%2;"
#define COMPUTE_m16n24_LINIT \
    INIT_m16n24 COMPUTE_m16n24_BODY("2451616") SAVE_p_m16n24_L_direct SAVE_p_m16n24_R_wscr
#define COMPUTE_m16n24_LSAVE \
    INIT_m16n24 COMPUTE_m16n24_BODY("2471616") SAVE_p_m16n24_L_rscr SAVE_p_m16n24_R_wscr
#define COMPUTE_m16n24_LTAIL \
    INIT_m16n24 COMPUTE_m16n24_BODY("2441616") SAVE_p_m16n24_L_rscr SAVE_p_m16n24_R_direct
#define COMPUTE_m16n24_RSAVE \
    INIT_m16n24 "leaq (%2,%3,8),%2; leaq (%2,%3,4),%2;"\
    COMPUTE_m16n24_BODY("2461616") SAVE_p_m16n24_R_rscr SAVE_p_m16n24_L_wscr\
    "negq %3; leaq (%2,%3,8),%2; leaq (%2,%3,4),%2; negq %3;"
#define COMPUTE_m16n24_RTAIL \
    INIT_m16n24 COMPUTE_m16n24_BODY("2431616") SAVE_p_m16n24_L_direct SAVE_p_m16n24_R_rscr

/* m = 8 *//* zmm0 for alpha, zmm1-2 for perm words, zmm4-7 for temporary use, zmm8-19 for accumulators */
#define KERNEL_k1m8n1 \
    "vbroadcastss (%1),%%ymm4; addq $4,%1; vfmadd231ps (%0),%%ymm4,%%ymm8; addq $32,%0;"
#define KERNEL_k1m8n2 \
    "vmovups (%0),%%ymm4; addq $32,%0;"\
    "vbroadcastss (%1),%%ymm5; vfmadd231ps %%ymm5,%%ymm4,%%ymm8;"\
    "vbroadcastss 4(%1),%%ymm6; vfmadd231ps %%ymm6,%%ymm4,%%ymm9; addq $8,%1;"
#define unit_kernel_k1m8n4(c1,c2,...)\
    "vbroadcastf32x4 ("#__VA_ARGS__"),%%zmm7; vfmadd231ps %%zmm7,%%zmm4,"#c1"; vfmadd231ps %%zmm7,%%zmm5,"#c2";"
#define KERNEL_h_k1m8n4 \
    "vbroadcastf32x4 (%0),%%zmm4; vpermilps %%zmm2,%%zmm4,%%zmm4; vbroadcastf32x4 16(%0),%%zmm5; vpermilps %%zmm2,%%zmm5,%%zmm5; addq $32,%0;"\
    unit_kernel_k1m8n4(%%zmm8,%%zmm9,%1)
#define KERNEL_k1m8n4 KERNEL_h_k1m8n4 "addq $16,%1;"
#define KERNEL_h_k1m8n8 KERNEL_h_k1m8n4 unit_kernel_k1m8n4(%%zmm10,%%zmm11,%1,%%r12,1)
#define KERNEL_k1m8n8 KERNEL_h_k1m8n8 "addq $16,%1;"
#define KERNEL_k1m8n12 KERNEL_h_k1m8n8 unit_kernel_k1m8n4(%%zmm12,%%zmm13,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m8n16 KERNEL_k1m8n12 unit_kernel_k1m8n4(%%zmm14,%%zmm15,%%r15)
#define KERNEL_k1m8n16 KERNEL_h_k1m8n16 "addq $16,%%r15;"
#define KERNEL_h_k1m8n20 KERNEL_h_k1m8n16 unit_kernel_k1m8n4(%%zmm16,%%zmm17,%%r15,%%r12,1)
#define KERNEL_k1m8n20 KERNEL_h_k1m8n20 "addq $16,%%r15;"
#define KERNEL_k1m8n24 KERNEL_h_k1m8n20 unit_kernel_k1m8n4(%%zmm18,%%zmm19,%%r15,%%r12,2) "addq $16,%%r15;"
#define INIT_m8n1 "vpxor %%ymm8,%%ymm8,%%ymm8;"
#define INIT_m8n2 "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9;"
#define unit_init_m8n4(c1,c2) "vpxorq "#c1","#c1","#c1";vpxorq "#c2","#c2","#c2";"
#define INIT_m8n4 unit_init_m8n4(%%zmm8,%%zmm9)
#define INIT_m8n8 INIT_m8n4 unit_init_m8n4(%%zmm10,%%zmm11)
#define INIT_m8n12 INIT_m8n8 unit_init_m8n4(%%zmm12,%%zmm13)
#define INIT_m8n16 INIT_m8n12 unit_init_m8n4(%%zmm14,%%zmm15)
#define INIT_m8n20 INIT_m8n16 unit_init_m8n4(%%zmm16,%%zmm17)
#define INIT_m8n24 INIT_m8n20 unit_init_m8n4(%%zmm18,%%zmm19)
#define SAVE_h_m8n1 "vfmadd213ps (%2),%%ymm0,%%ymm8; vmovups %%ymm8,(%2); addq $32,%2;"
#define SAVE_h_m8n2 \
    "vpermt2ps %%zmm9, %%zmm3, %%zmm8;"\
    "vfmadd213ps (%2),%%zmm0,%%zmm8; vmovups %%zmm8,(%2); addq $64,%2;"
#define unit_save_m8n4(c1_no,c2_no)\
    "vfmadd213ps (%5),%%zmm0,%%zmm"#c1_no"; vfmadd213ps 64(%5),%%zmm0,%%zmm"#c2_no";"\
    "vmovups %%zmm"#c1_no",(%5); vmovups %%zmm"#c2_no",64(%5);"\
    "leaq (%5,%3,4),%5;"
#define SAVE_h_m8n4 "movq %2,%5;" "addq $128,%2;" unit_save_m8n4(8,9)
#define SAVE_h_m8n8 SAVE_h_m8n4 unit_save_m8n4(10,11)
#define SAVE_h_m8n12 SAVE_h_m8n8 unit_save_m8n4(12,13)
#define SAVE_h_m8n16 SAVE_h_m8n12 unit_save_m8n4(14,15)
#define SAVE_h_m8n20 SAVE_h_m8n16 unit_save_m8n4(16,17)
#define SAVE_h_m8n24 SAVE_h_m8n20 unit_save_m8n4(18,19)
#define SAVE_m8(ndim) SAVE_h_m8n##ndim
#define COMPUTE_m8(ndim) \
    INIT_m8n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15;"\
    "testq %4,%4; jz "#ndim"008082f;"\
    #ndim"008081:\n\t"\
    KERNEL_k1m8n##ndim "decq %4; jnz "#ndim"008081b;"\
    #ndim"008082:\n\t"\
    SAVE_m8(ndim)

/* m = 4 *//* zmm0 for alpha, zmm1-2 for perm words, zmm4-7 for temporary use, zmm8-15 for accumulators */
#define KERNEL_k1m4n1 "vbroadcastss (%1),%%xmm4; addq $4,%1; vfmadd231ps (%0),%%xmm4,%%xmm8; addq $16,%0;"
#define KERNEL_k1m4n2 "vmovups (%0),%%xmm4; addq $16,%0;"\
    "vbroadcastss (%1),%%xmm5; vfmadd231ps %%xmm5,%%xmm4,%%xmm8;"\
    "vbroadcastss 4(%1),%%xmm5; vfmadd231ps %%xmm5,%%xmm4,%%xmm9; addq $8,%1;"
#define unit_kernel_k1m4n4(c1,...) "vbroadcastf32x4 ("#__VA_ARGS__"),%%zmm7; vfmadd231ps %%zmm7,%%zmm4,"#c1";"
#define KERNEL_h_k1m4n4 "vbroadcastf32x4 (%0),%%zmm4; vpermilps %%zmm2,%%zmm4,%%zmm4; addq $16,%0;" unit_kernel_k1m4n4(%%zmm8,%1)
#define KERNEL_k1m4n4 KERNEL_h_k1m4n4 "addq $16,%1;"
#define KERNEL_h_k1m4n8 KERNEL_h_k1m4n4 unit_kernel_k1m4n4(%%zmm9,%1,%%r12,1)
#define KERNEL_k1m4n8 KERNEL_h_k1m4n8 "addq $16,%1;"
#define KERNEL_k1m4n12 KERNEL_h_k1m4n8 unit_kernel_k1m4n4(%%zmm10,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m4n16 KERNEL_k1m4n12 unit_kernel_k1m4n4(%%zmm11,%%r15)
#define KERNEL_k1m4n16 KERNEL_h_k1m4n16 "addq $16,%%r15;"
#define KERNEL_h_k1m4n20 KERNEL_h_k1m4n16 unit_kernel_k1m4n4(%%zmm12,%%r15,%%r12,1)
#define KERNEL_k1m4n20 KERNEL_h_k1m4n20 "addq $16,%%r15;"
#define KERNEL_h_k1m4n24 KERNEL_h_k1m4n20 unit_kernel_k1m4n4(%%zmm13,%%r15,%%r12,2)
#define KERNEL_k1m4n24 KERNEL_h_k1m4n24 "addq $16,%%r15;"
#define INIT_m4n1 "vpxor %%xmm8,%%xmm8,%%xmm8;"
#define INIT_m4n2 "vpxor %%xmm8,%%xmm8,%%xmm8; vpxor %%xmm9,%%xmm9,%%xmm9;"
#define INIT_m4n4 "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define INIT_m4n8 INIT_m4n4 "vpxorq %%zmm9,%%zmm9,%%zmm9;"
#define INIT_m4n12 INIT_m4n8 "vpxorq %%zmm10,%%zmm10,%%zmm10;"
#define INIT_m4n16 INIT_m4n12 "vpxorq %%zmm11,%%zmm11,%%zmm11;"
#define INIT_m4n20 INIT_m4n16 "vpxorq %%zmm12,%%zmm12,%%zmm12;"
#define INIT_m4n24 INIT_m4n20 "vpxorq %%zmm13,%%zmm13,%%zmm13;"
#define SAVE_h_m4n1 "vfmadd213ps (%2),%%xmm0,%%xmm8; vmovups %%xmm8,(%2); addq $16,%2;"
#define SAVE_h_m4n2\
    "vpermt2ps %%zmm9, %%zmm3, %%zmm8;"\
    "vfmadd213ps (%2),%%zmm0,%%zmm8; vmovups %%zmm8,(%2); addq $32,%2;"
#define unit_save_m4n4(c1_no)\
    "vfmadd213ps (%5),%%zmm0,%%zmm"#c1_no";"\
    "vmovups %%zmm"#c1_no",(%5);"\
    "leaq (%5,%3,4),%5;"
#define SAVE_h_m4n4 "movq %2,%5;" "addq $64,%2;" unit_save_m4n4(8)
#define SAVE_h_m4n8 SAVE_h_m4n4 unit_save_m4n4(9)
#define SAVE_h_m4n12 SAVE_h_m4n8 unit_save_m4n4(10)
#define SAVE_h_m4n16 SAVE_h_m4n12 unit_save_m4n4(11)
#define SAVE_h_m4n20 SAVE_h_m4n16 unit_save_m4n4(12)
#define SAVE_h_m4n24 SAVE_h_m4n20 unit_save_m4n4(13)
#define SAVE_m4(ndim) SAVE_h_m4n##ndim
#define COMPUTE_m4(ndim) \
    INIT_m4n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15;"\
    "testq %4,%4; jz "#ndim"004042f;"\
    #ndim"004041:\n\t"\
    KERNEL_k1m4n##ndim "decq %4; jnz "#ndim"004041b;"\
    #ndim"004042:\n\t"\
    SAVE_m4(ndim)

/* m = 2 *//* xmm0 for alpha, xmm1-xmm3 for temporary use, xmm4-xmm15 for accumulators */
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m2n1 \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,%1;"
#define SAVE_h_m2n1 "vmovsd (%2),%%xmm1; vfmadd213ps %%xmm1,%%xmm0,%%xmm4; vmovsd %%xmm4,(%2);  addq $8,%2;"
#define INIT_m2n2 INIT_m2n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define KERNEL_k1m2n2 \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss  (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "vbroadcastss 4(%1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm5;"\
    "addq $8,%1;"
#define SAVE_h_m2n2 "vunpcklps %%xmm5, %%xmm4, %%xmm5; vfmadd213ps (%2),%%xmm0,%%xmm5; vmovups %%xmm5,(%2); addq $16,%2;"
#define INIT_m2n4  INIT_m2n2
#define INIT_m2n8  INIT_m2n4 "vpxor %%xmm6,%%xmm6,%%xmm6; vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m2n12 INIT_m2n8 "vpxor %%xmm8,%%xmm8,%%xmm8; vpxor %%xmm9,%%xmm9,%%xmm9;"
#define INIT_m2n16 INIT_m2n12 "vpxor %%xmm10,%%xmm10,%%xmm10; vpxor %%xmm11,%%xmm11,%%xmm11;"
#define INIT_m2n20 INIT_m2n16 "vpxor %%xmm12,%%xmm12,%%xmm12; vpxor %%xmm13,%%xmm13,%%xmm13;"
#define INIT_m2n24 INIT_m2n20 "vpxor %%xmm14,%%xmm14,%%xmm14; vpxor %%xmm15,%%xmm15,%%xmm15;"
#define KERNEL_h_k1m2n4 \
    "vbroadcastss (%0),%%xmm1; vbroadcastss 4(%0),%%xmm2; addq $8,%0;"\
    "vmovups (%1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm4; vfmadd231ps %%xmm2,%%xmm3,%%xmm5;"
#define KERNEL_k1m2n4 KERNEL_h_k1m2n4 "addq $16,%1;"
#define KERNEL_h_k1m2n8 KERNEL_h_k1m2n4 "vmovups (%1,%%r12,1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm6; vfmadd231ps %%xmm2,%%xmm3,%%xmm7;"
#define KERNEL_k1m2n8 KERNEL_h_k1m2n8 "addq $16,%1;"
#define KERNEL_k1m2n12 KERNEL_h_k1m2n8 \
    "vmovups (%1,%%r12,2),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm8; vfmadd231ps %%xmm2,%%xmm3,%%xmm9; addq $16,%1;"
#define KERNEL_h_k1m2n16 KERNEL_k1m2n12 "vmovups (%%r15),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm10; vfmadd231ps %%xmm2,%%xmm3,%%xmm11;"
#define KERNEL_k1m2n16 KERNEL_h_k1m2n16 "addq $16,%%r15;"
#define KERNEL_h_k1m2n20 KERNEL_h_k1m2n16 "vmovups (%%r15,%%r12,1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm12; vfmadd231ps %%xmm2,%%xmm3,%%xmm13;"
#define KERNEL_k1m2n20 KERNEL_h_k1m2n20 "addq $16,%%r15;"
#define KERNEL_h_k1m2n24 KERNEL_h_k1m2n20 "vmovups (%%r15,%%r12,2),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm14; vfmadd231ps %%xmm2,%%xmm3,%%xmm15;"
#define KERNEL_k1m2n24 KERNEL_h_k1m2n24 "addq $16,%%r15;"
#define unit_save_m2n4(c1,c2) \
    "vfmadd213ps (%5),%%xmm0,"#c1"; vfmadd213ps 16(%5),%%xmm0,"#c2";"\
    "vmovups "#c1",(%5); vmovups "#c2", 16(%5);"\
    "leaq (%5,%3,4),%5;"
#define SAVE_h_m2n4  "movq %2,%5;" "addq $32,%2;" unit_save_m2n4(%%xmm4,%%xmm5)
#define SAVE_h_m2n8  SAVE_h_m2n4   unit_save_m2n4(%%xmm6,%%xmm7)
#define SAVE_h_m2n12 SAVE_h_m2n8   unit_save_m2n4(%%xmm8,%%xmm9)
#define SAVE_h_m2n16 SAVE_h_m2n12  unit_save_m2n4(%%xmm10,%%xmm11)
#define SAVE_h_m2n20 SAVE_h_m2n16  unit_save_m2n4(%%xmm12,%%xmm13)
#define SAVE_h_m2n24 SAVE_h_m2n20  unit_save_m2n4(%%xmm14,%%xmm15)
#define SAVE_m2(ndim) SAVE_h_m2n##ndim
#define COMPUTE_m2(ndim) \
    INIT_m2n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15;"\
    "testq %4,%4; jz "#ndim"002022f;"\
    #ndim"002021:\n\t"\
    KERNEL_k1m2n##ndim "decq %4; jnz "#ndim"002021b;"\
    #ndim"002022:\n\t"\
    SAVE_m2(ndim)

/* m = 1 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm9 for accumulators */
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m1n1 \
    "vmovss (%1),%%xmm3; addq $4,%1;"\
    "vmovss (%0),%%xmm1; vfmadd231ss %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_h_m1n1 "vfmadd213ss (%2),%%xmm0,%%xmm4; vmovss %%xmm4,(%2);"
#define INIT_m1n2 INIT_m1n1
#define KERNEL_k1m1n2 \
    "vmovsd (%1),%%xmm3; addq $8,%1;"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_h_m1n2 \
    "vmovsd (%2),%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm4;"\
    "vmovsd %%xmm4,(%2); addq $8,%2;"
#define INIT_m1n4  INIT_m1n2
#define INIT_m1n8  INIT_m1n4 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m1n12 INIT_m1n8 "vpxor %%xmm6,%%xmm6,%%xmm6;"
#define INIT_m1n16 INIT_m1n12 "vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m1n20 INIT_m1n16 "vpxor %%xmm8,%%xmm8,%%xmm8;"
#define INIT_m1n24 INIT_m1n20 "vpxor %%xmm9,%%xmm9,%%xmm9;"
#define KERNEL_h_k1m1n4 \
    "vbroadcastss (%0),%%xmm1; addq $4,%0; vfmadd231ps (%1),%%xmm1,%%xmm4;"
#define KERNEL_k1m1n4 KERNEL_h_k1m1n4 "addq $16,%1;"
#define KERNEL_h_k1m1n8 KERNEL_h_k1m1n4 "vfmadd231ps (%1,%%r12,1),%%xmm1,%%xmm5;"
#define KERNEL_k1m1n8 KERNEL_h_k1m1n8 "addq $16,%1;"
#define KERNEL_k1m1n12 KERNEL_h_k1m1n8 "vfmadd231ps (%1,%%r12,2),%%xmm1,%%xmm6; addq $16,%1;"
#define KERNEL_h_k1m1n16 KERNEL_k1m1n12 "vfmadd231ps (%%r15),%%xmm1,%%xmm7;"
#define KERNEL_k1m1n16 KERNEL_h_k1m1n16 "addq $16,%%r15;"
#define KERNEL_h_k1m1n20 KERNEL_h_k1m1n16 "vfmadd231ps (%%r15,%%r12,1),%%xmm1,%%xmm8;"
#define KERNEL_k1m1n20 KERNEL_h_k1m1n20 "addq $16,%%r15;"
#define KERNEL_h_k1m1n24 KERNEL_h_k1m1n20 "vfmadd231ps (%%r15,%%r12,2),%%xmm1,%%xmm9;"
#define KERNEL_k1m1n24 KERNEL_h_k1m1n24 "addq $16,%%r15;"
#define unit_save_m1n4(c1) \
    "vfmadd213ps (%5),%%xmm0,"#c1";"\
    "vmovups "#c1",(%5);"\
    "leaq (%5,%3,4),%5;"
#define SAVE_h_m1n4 "movq %2,%5;" "addq $16,%2;"unit_save_m1n4(%%xmm4)
#define SAVE_h_m1n8  SAVE_h_m1n4  unit_save_m1n4(%%xmm5)
#define SAVE_h_m1n12 SAVE_h_m1n8  unit_save_m1n4(%%xmm6)
#define SAVE_h_m1n16 SAVE_h_m1n12 unit_save_m1n4(%%xmm7)
#define SAVE_h_m1n20 SAVE_h_m1n16 unit_save_m1n4(%%xmm8)
#define SAVE_h_m1n24 SAVE_h_m1n20 unit_save_m1n4(%%xmm9)
#define SAVE_m1(ndim) SAVE_h_m1n##ndim
#define COMPUTE_m1(ndim) \
    INIT_m1n##ndim\
    "movq %%r13,%4; movq %%r14,%1; leaq (%1,%%r12,2),%%r15; addq %%r12,%%r15;"\
    "testq %4,%4; jz "#ndim"001012f;"\
    #ndim"001011:\n\t"\
    KERNEL_k1m1n##ndim "decq %4; jnz "#ndim"001011b;"\
    #ndim"001012:\n\t"\
    SAVE_m1(ndim)

/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 = "+r"(K), %5 = "+r"(ctemp) */
/* %6 = "+r"(next_b), %7 = "m"(ALPHA), %8 = "m"(M) */
/* r11 = m_counter, r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const), r15 = %1 + 3r12 */

#define COMPUTE(ndim) {\
    next_b = b_pointer + ndim * K;\
    __asm__ __volatile__(\
    "vbroadcastss %7,%%zmm0; vmovups %13,%%zmm1; vmovups %12,%%zmm2; vmovups %11,%%zmm3;"\
    "movq %4,%%r13; movq %4,%%r12; salq $4,%%r12; movq %1,%%r14; movq %8,%%r11;"\
    "cmpq $16,%%r11;jb 33101"#ndim"f;"\
    "33109"#ndim":\n\t"\
    COMPUTE_m16(ndim)\
    "subq $16,%%r11;cmpq $16,%%r11;jnb 33109"#ndim"b;"\
    "33101"#ndim":\n\t"\
    "vmovups %9,%%zmm1; vmovups %10,%%zmm2; vmovups %14,%%zmm3;"\
    "cmpq $8,%%r11;jb 33102"#ndim"f;"\
    COMPUTE_m8(ndim)\
    "subq $8,%%r11;"\
    "33102"#ndim":\n\t"\
    "cmpq $4,%%r11;jb 33103"#ndim"f;"\
    COMPUTE_m4(ndim)\
    "subq $4,%%r11;"\
    "33103"#ndim":\n\t"\
    "cmpq $2,%%r11;jb 33104"#ndim"f;"\
    COMPUTE_m2(ndim)\
    "subq $2,%%r11;"\
    "33104"#ndim":\n\t"\
    "testq %%r11,%%r11;jz 33105"#ndim"f;"\
    COMPUTE_m1(ndim)\
    "33105"#ndim":\n\t"\
    "movq %%r13,%4; movq %%r14,%1; vzeroupper;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(ldc_in_bytes),"+r"(K),"+r"(ctemp),"+r"(next_b):"m"(ALPHA),"m"(M),"m"(perm[0]),"m"(permil[0]),"m"(permt1[0]),"m"(permt2[0]),"m"(permt3[0]),"m"(shuff[0])\
    :"r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14",\
    "zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",\
    "cc","memory");\
    a_pointer -= M * K; b_pointer += ndim * K; c_pointer += LDC * (ndim - 3) - M;\
}

#define COMPUTE_n24 {\
    next_b = b_pointer + 24 * K;\
    __asm__ __volatile__(\
    "vbroadcastss %8,%%zmm0; vmovups %14,%%zmm1; vmovups %13,%%zmm2; vmovups %12,%%zmm3;"\
    "movq %4,%%r13; movq %4,%%r12; salq $4,%%r12; movq %1,%%r14; movq %9,%%r11;"\
    "cmpq $32,%%r11;jb 3310024f;"\
    COMPUTE_m16n24_LINIT "subq $16,%%r11; cmpq $32,%%r11;jb 3310724f;"\
    "3310924:\n\t"\
    COMPUTE_m16n24_RSAVE "subq $16,%%r11; cmpq $32,%%r11;jb 3310824f;"\
    COMPUTE_m16n24_LSAVE "subq $16,%%r11; cmpq $32,%%r11;jnb 3310924b;"\
    "3310724:\n\t"\
    COMPUTE_m16n24_RTAIL "subq $16,%%r11; jmp 3310124f;"\
    "3310824:\n\t"\
    COMPUTE_m16n24_LTAIL "subq $16,%%r11; jmp 3310124f;"\
    "3310024:\n\t"\
    "cmpq $16,%%r11;jb 3310124f;"\
    COMPUTE_m16(24)\
    "subq $16,%%r11;"\
    "3310124:\n\t"\
    "cmpq $8,%%r11;jb 3310224f;"\
    "vmovups %10,%%zmm1; vmovups %11,%%zmm2;vmovups %15,%%zmm3;"\
    COMPUTE_m8(24)\
    "subq $8,%%r11;"\
    "3310224:\n\t"\
    "cmpq $4,%%r11;jb 3310324f;"\
    COMPUTE_m4(24)\
    "subq $4,%%r11;"\
    "3310324:\n\t"\
    "cmpq $2,%%r11;jb 3310424f;"\
    COMPUTE_m2(24)\
    "subq $2,%%r11;"\
    "3310424:\n\t"\
    "testq %%r11,%%r11;jz 3310524f;"\
    COMPUTE_m1(24)\
    "3310524:\n\t"\
    "movq %%r13,%4; movq %%r14,%1; vzeroupper;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(ldc_in_bytes),"+r"(K),"+r"(ctemp),"+r"(next_b),"+r"(wscr):"m"(ALPHA),"m"(M),"m"(perm[0]),"m"(permil[0]),"m"(permt1[0]),"m"(permt2[0]),"m"(permt3[0]),"m"(shuff[0])\
    :"r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14",\
    "zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31",\
    "cc","memory");\
    a_pointer -= M * K; b_pointer += 24 * K; c_pointer += LDC * 21 - M;\
}

int __attribute__ ((noinline))
gemm_kernel_pre(long m, long n, long k, float alpha, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, long LDC)
{
    if(m==0||n==0||k==0||alpha==(float)0.0) return 0;
    float scr[192]; float *wscr = scr;
    int8_t mu = 0, su = 0;
    int64_t ldc_in_bytes = (int64_t)LDC * sizeof(float);float ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k;
    int32_t perm[16] = {0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};
    int32_t permil[16] = {0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
    uint64_t permt1[8] = {0,4,8,12,1,5,9,13};
    uint64_t permt2[8] = {10,14,2,6,11,15,3,7};
    uint64_t permt3[8] = {0,4,1,5,2,6,3,7};
    int32_t shuff[16] = {0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23};
    long n_count = n;
    float *a_pointer = A,*b_pointer = B,*c_pointer = C,*ctemp = C,*next_b = B;
    // COMPUTE_n24 (the software-pipelined LSAVE/RSAVE m16 loop) is now
    // correct -- its save macros had been inherited verbatim from upstream
    // OpenBLAS and never adapted to the propagated layout, so they emitted a
    // canonical column-major footprint and left part of the propagated one
    // unwritten. They have been re-derived at n4-group granularity (see the
    // unit_save_m16n4_{r,w}scr comment above) and verified bit-identical to
    // COMPUTE(24) over 2025 kernel shapes and the 15-case x 9-thread driver
    // sweep.
    //
    // COMPUTE(24) is nevertheless kept as the active path: measured in
    // isolation, the two are indistinguishable (44.4 GFLOP/s on every k=448
    // shape tried), and at small k the pipelined path is ~1-4% SLOWER, in both
    // this kernel and the canonical one. Pipelining exists upstream to merge
    // 64 B column-strided stores into 128 B ones; the propagated layout already
    // stores 256 B fully-contiguous per (m16 x n4) tile, so there is nothing
    // left for it to recover. Keeping both kernels on COMPUTE(24) is therefore
    // a measured choice, not a workaround.
    for(;n_count>23;n_count-=24) COMPUTE(24)
    for(;n_count>19;n_count-=20) COMPUTE(20)
    for(;n_count>15;n_count-=16) COMPUTE(16)
    for(;n_count>11;n_count-=12) COMPUTE(12)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    if(n_count>1) mu = 2;
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0){
        // if (LDC < 3) su = 1;
        c_pointer += LDC * mu + su;
        COMPUTE(1)
    }
    return 0;
}
#include <immintrin.h>
//#include "sgemm_direct_skylakex.c"
