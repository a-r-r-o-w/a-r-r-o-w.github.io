---
{
  "title": "Anatomy of high performance matrix multiplication on M4 Max",
  "authors": ["Aryan V S"],
  "code": "TODO",
  "date": "2025-10-08",
  "tags": ["hpc", "matrix-multiplication", "mlsys", "simd", "openmp", "neon"]
}
---

<details>
<summary> Boring back story you can skip </summary>

It was early September. I had just quit my job at Hugging Face to try something new. My new role would not start for another week. During this time, I had no fancy GPUs to play with. Bored out of my mind after a complete day of relaxing, I wanted to challenge myself to something I'd never done before--writing a high performance CPU matmul kernel.

As a perk of my new role, I purchased a fancy new Apple M4 Max (16-core CPU, 40-core GPU) to play with. Being a ML practitioner with a tiny background in kernel authoring with CUDA, although nothing of real significance, this was the simplest difficult challenge for me.

By this point in life, I had read and learnt all about the general optimization ideas that make up a fast matmul kernel on GPUs - loop reordering, block/warp/thread tiling, coalesced memory reads and writes, async load/store, using tensor cores, pipelining and instruction-level parallelism, autotuning, etc. Knowing the general ideas, it should have been easy to write a decent kernel for the M4 CPU, right? That's what I thought, but boy was I so wrong. In principle, a lot of the ideas still apply for CPUs--you want to reduce the number of slow memory reads/writes from storage that is slower to access, and increase reuse of data already available in fast bandwidth caches from past operations, overlap memory loads to caches with ongoing computation, and so on. ......

My aim with this exploration was to learn [Neon](https://developer.arm.com/Architectures/Neon). After several weeks of painful exercises, I think I understand Neon decently well now. However, despite a lot of hours trying to write a fast Neon kernel, I could not reach the maximal performance theoretically possibly by the hardware. To reach higher perf, an online friend, [Thien](https://x.com/gaunernst), pointed me to [AMX](https://github.com/corsix/amx). It is the instruction set for a co-processor built into Apple's M-series chips, designed specifically for matmuls and convs with dedicated tile registers and multiply-accumulate (MAC) units that blow past what Neon can do alone. This writeup documents my step-by-step process of writing a decently fast matmul kernel using Neon and AMX for M4 CPUs.

</details>

### Naive Matrix Multiplication

```c
static void sgemm_1(const float *restrict a, const float *restrict b, float *restrict c, size_t m, size_t n, size_t k) {
  for (size_t x = 0; x < m; ++x) {
    for (size_t y = 0; y < n; ++y) {
      float accum = 0.0f;
      for (size_t z = 0; z < k; ++z)
        accum += a[x * k + z] * b[z * n + y];
      c[x * n + y] = accum;
    }
  }
}
```

<details>
<summary> Annotated `sgemm_1` assembly (-O0) </summary>

```arm
sgemm_1:
        sub     sp, sp, #80               ; Allocate 80 bytes of stack space
        str     x0, [sp, #72]             ; x0 is the address of `a` stored onto stack
        str     x1, [sp, #64]             ; x1 is the address of `b` stored onto stack
        str     x2, [sp, #56]             ; x2 is the address of `c` stored onto stack
        str     x3, [sp, #48]             ; x3 is the value `m` stored onto stack
        str     x4, [sp, #40]             ; x4 is the value `n` stored onto stack
        str     x5, [sp, #32]             ; x5 is the value `k` stored onto stack
        str     xzr, [sp, #24]            ; initialize `x` to 0 and store onto stack
        b       .LBB0_1                   ; jump to LBB0_1
.LBB0_1:
        ldr     x8, [sp, #24]             ; load x from stack into register x8
        ldr     x9, [sp, #48]             ; load m from stack into register x8
        subs    x8, x8, x9                ; compute (x - m) and store result in x8; the "s" suffix in sub instruction means that the condition flags must be set
        b.hs    .LBB0_12                  ; C flag is set to 1 when a borrow does not occur. If x - M >= 0 (no borrow), then jump to LBB0_12 (return); b.hs means branch if C == 1
        b       .LBB0_2                   ; jump to LBB0_2
.LBB0_2:
        str     xzr, [sp, #16]            ; initialize `y` to 0 and store onto stack
        b       .LBB0_3                   ; jump to LBB0_3
.LBB0_3:
        ldr     x8, [sp, #16]             ; load y from stack into register x8
        ldr     x9, [sp, #40]             ; load n from stack into register x9
        subs    x8, x8, x9                ; compute (y - n) and store result in x8
        b.hs    .LBB0_10                  ; if C flag is set to 1 (i.e. if y - n >= 0), jump to LBB0_10
        b       .LBB0_4                   ; jump to LBB0_4
.LBB0_4:
        movi    d0, #0000000000000000     ; initialize accum = 0.0f
        str     s0, [sp, #12]             ; store accum from s0 onto stack
        str     xzr, [sp]                 ; initialize `z` to 0 and store onto stack
        b       .LBB0_5                   ; jump to LBB0_5
.LBB0_5:
        ldr     x8, [sp]                  ; load `z` into register r8
        ldr     x9, [sp, #32]             ; load `k` into register r9
        subs    x8, x8, x9                ; compute (z - k) and store result in r8
        b.hs    .LBB0_8                   ; if C flag is set to 1 (i.e. if z - k >= 0), jump to LBB0_8
        b       .LBB0_6                   ; jumpt to LBB0_6
.LBB0_6:
        ldr     x8, [sp, #72]             ; load address of `a` into x8
        ldr     x9, [sp, #24]             ; load `x` into x9
        ldr     x10, [sp, #32]            ; load `k` into x10
        mul     x9, x9, x10               ; compute x * k and store into x9
        ldr     x10, [sp]                 ; load `z` into x10
        add     x9, x9, x10               ; compute (x * k) + z and store into x9
        ldr     s0, [x8, x9, lsl #2]      ; s0 = a[x * k + z]. lsl #2, i.e. shift left by 2 or multiply by 4, is to compute byte offset for float precision
        ldr     x8, [sp, #64]             ; x8 = b
        ldr     x9, [sp]                  ; x9 = z
        ldr     x10, [sp, #40]            ; x10 = n
        mul     x9, x9, x10               ; x9 = z * n
        ldr     x10, [sp, #16]            ; x10 = y
        add     x9, x9, x10               ; x9 = z * n + y
        ldr     s1, [x8, x9, lsl #2]      ; s1 = b[z * n + y]
        ldr     s2, [sp, #12]             ; s2 = accum
        fmadd   s0, s0, s1, s2            ; s0 = a[x * k + z]
        str     s0, [sp, #12]             ; accum = s0
        b       .LBB0_7                   ; jump to LBB0_7
.LBB0_7:
        ldr     x8, [sp]                  ; x8 = z
        add     x8, x8, #1                ; x8 = z + 1
        str     x8, [sp]                  ; z = x8
        b       .LBB0_5                   ; jump back to LBB0_5
.LBB0_8:
        ldr     s0, [sp, #12]             # s0 = accum; we're here if the innermost z-loop is completed
        ldr     x8, [sp, #56]             ; x8 = c
        ldr     x9, [sp, #24]             ; x9 = x
        ldr     x10, [sp, #40]            ; x10 = n
        mul     x9, x9, x10               ; x9 = x * n
        ldr     x10, [sp, #16]            ; x10 = y
        add     x9, x9, x10               ; x9 = x * n + y
        str     s0, [x8, x9, lsl #2]      ; s0 = c[x * n + y]
        b       .LBB0_9                   ; jump to LBB0_9
.LBB0_9:
        ldr     x8, [sp, #16]             ; x8 = y
        add     x8, x8, #1                ; x8 = y + 1
        str     x8, [sp, #16]             ; y = x8
        b       .LBB0_3                   ; jump to LBB0_3
.LBB0_10:
        b       .LBB0_11                  ; we're here if the middle y-loop is completed
.LBB0_11:
        ldr     x8, [sp, #24]             ; x8 = x
        add     x8, x8, #1                ; x8 = x + 1
        str     x8, [sp, #24]             ; x = x8
        b       .LBB0_1                   ; jump to LBB0_1
.LBB0_12:
        add     sp, sp, #80               ; we're here if the outer x-loop is completed
        ret
```

</details>

<details>
<summary> Annotated `sgemm_1` assembly (-O1) </summary>

```arm
_sgemm_1:
	cbz	x3, LBB77_9                    ; if (m == 0) return
	mov	x8, #0                         ; x = 0
	lsl	x9, x4, #2                     ; byte_stride_n = n * 4
	lsl	x10, x5, #2                    ; byte_stride_k = k * 4
	b	LBB77_3                          

LBB77_2:                             ; end of y loop
	add	x8, x8, #1                     ; ++x
	add	x0, x0, x10                    ; a += k (start next `a` row)
	cmp	x8, x3                         ; 
	b.eq	LBB77_9                      ; if (x == m) return

LBB77_3:                             ; y loop
	cbz	x4, LBB77_2                    ; if (n == 0) continue
	mov	x11, #0                        ; y = 0
	mul	x12, x8, x4                    ; offset = x * n
	add	x12, x2, x12, lsl #2           ; &c[x * n]
	mov	x13, x1                        ; b_col = b
	b	LBB77_6                          

LBB77_5:                             ; end of z loop
	str	s0, [x12, x11, lsl #2]         ; c[x * n + y] = accum
	add	x11, x11, #1                   ; ++y
	add	x13, x13, #4                   ; ++b_col
	cmp	x11, x4                        ; 
	b.eq	LBB77_2                      ; if (y == n) next x

LBB77_6:                             ; z loop
	movi	d0, #0000000000000000        ; accum = 0.0f
	cbz	x5, LBB77_5                    ; if (k == 0) store
	mov	x14, x0                        ; a_elem = &a[x * k]
	mov	x15, x13                       ; b_elem = &b[y]
	mov	x16, x5                        ; z = k

LBB77_8:                             ; z loop body
	ldr	s1, [x14], #4                  ; a[x * k + z]
	ldr	s2, [x15]                      ; b[z * n + y]
	fmadd	s0, s1, s2, s0               ; accum += a[x * k + z] * b[z * n + y]
	add	x15, x15, x9                   ; b_elem += n
	subs	x16, x16, #1                 ; --z
	b.ne	LBB77_8                      ; if (z != 0) continue z loop body
	b	LBB77_5                          ; goto end of z loop

LBB77_9:
	ret                                ; return
```

</details>

<details>
<summary> Annotated `sgemm_1` assembly (-O2) </summary>

```arm
_sgemm_1:
	cbz	x3, LBB77_22                       ; if (m == 0) return
	stp	x24, x23, [sp, #-48]!
	stp	x22, x21, [sp, #16]
	stp	x20, x19, [sp, #32]
	mov	x8, #0                             ; x = 0
	cmp	x5, #3                             ; if (k > 3)
	ccmp	x4, #1, #0, hi                   ;    and (n == 1)
	cset	w9, eq                           ;    set vectorization flag
	and	x10, x5, #0xfffffffffffffff0       ; k & ~15 (16-element chunk end)
	and	x11, x5, #0xc                      ; k & 12 (4-element tail)
	and	x12, x5, #0xfffffffffffffffc       ; k & ~3 (4-element chunk end)
	add	x13, x0, #32                       ; a_row_offset = &a[8] (because: 32 / sizeof(float) = 8)
	lsl	x14, x5, #2                        ; byte_stride_k = k * 4
	add	x15, x1, #32                       ; &b[8]
	neg	x16, x12                           ; -(4-element chunk end)
	lsl	x17, x4, #2                        ; byte_stride_n = n * 4
	b	LBB77_3                              ; goto LBB77_3

LBB77_2:                                 ; x loop
	add	x8, x8, #1                         ; ++x
	add	x13, x13, x14                      ; a_row_offset += byte_stride_k
	add	x0, x0, x14                        ; a += k
	cmp	x8, x3                             ; if (x == m)
	b.eq	LBB77_21                         ;     return

LBB77_3:
	cbz	x4, LBB77_2                        ; if (n == 0) goto LBB77_2
	mov	x6, #0                             ; y = 0
	mul	x7, x8, x4                         ; byte_stride_n = x * n
	add	x7, x2, x7, lsl #2                 ; &c[byte_stride_n]
	mov	x19, x1                            ; b_col = b
	mov	x20, x15                           ; b_col_offset = b + 8
	b	LBB77_7                              ; goto LBB77_&

LBB77_5:                                 ; handle (k == 0)
	movi	d0, #0000000000000000            ; accum = 0.0f

LBB77_6:                                 ; end of z loop
	str	s0, [x7, x6, lsl #2]               ; c[x * n + y] = accum
	add	x6, x6, #1                         ; ++y
	add	x20, x20, #4                       ; b_col_offset += 4
	add	x19, x19, #4                       ; b_col += 4
	cmp	x6, x4                             ; if (y == n)
	b.eq	LBB77_2                          ;     goto LBB77_2

LBB77_7:
	cbz	x5, LBB77_5                        ; if (k == 0) goto LBB77_5 (i.e., set accum = 0.0f)
	tbz	w9, #0, LBB77_11                   ; test bit 0 of w9 register. if zero, goto LBB77_11 (non-vectorizable case)
	cmp	x5, #16                            ; if (k >= 16)
	b.hs	LBB77_12                         ;     goto LBB77_12 (vectorized 16-value loop)
	mov	x22, #0                            ; z = 0
	movi	d0, #0000000000000000            ; accum = 0.0f
	b	LBB77_16                             ; goto LBB77_16 (4-element tail)

LBB77_11:
	mov	x21, #0
	movi	d0, #0000000000000000            ; accum = 0.0f
	b	LBB77_19                             ; goto LBB77_19 (scalar loop)

LBB77_12:
	movi	d0, #0000000000000000            ; accum = 0.0f
	mov	x21, x20                           ; inner_b_col = b_col_offset
	mov	x22, x13                           ; inner_a_row = a_row_offset
	mov	x23, x10                           ; chunk_end = k & ~15

LBB77_13:                                ; for (z = 0; z < chunk_end; z += 16)
	ldp	q1, q2, [x22, #-32]                ; q1 = inner_a_row[z + 0 : z + 4], q2 = inner_a_row[z + 4 : z + 8]
	ldp	q3, q4, [x22], #64                 ; q3 = inner_a_row[z + 8 : z + 12], q4 = inner_a_row[z + 12 : z + 16]
	ldp	q5, q6, [x21, #-32]                ; q5 = inner_b_col[z + 0 : z + 4], q6 = inner_b_col[z + 4 : z + 8]
	ldp	q7, q16, [x21], #64                ; q7 = inner_b_col[z + 8 : z + 12], q8 = inner_b_col[z + 12 : z + 16]
	fmul.4s	v1, v1, v5                     ; multiply 4 fp32 values element-wise in v1 and v5 registers (https://developer.arm.com/documentation/102374/0103/Registers-in-AArch64---general-purpose-registers)
	mov	s5, v1[3]                          ; move 32 bits (single fp32 value) in s5 from 4th value of 128-bit v1-register
	mov	s17, v1[2]                         ; move single fp32 value into s17 from 3rd value of 128-bit v1 register
	mov	s18, v1[1]                         ; move single fp32 value into s18 from 2nd value of 128-bit v1 register
	fmul.4s	v2, v2, v6                     ; multiply 4 fp32 values element-wise in v2 and v6 registers
	mov	s6, v2[3]                          ; move 32 bits (single fp32 value) in s6 from 4th value of 128-bit v-register
	mov	s19, v2[2]                         ; move single fp32 value into s19 from 3rd value of 128-bit v2 register
	mov	s20, v2[1]                         ; move single fp32 value into s20 from 2nd value of 128-bit v2 register
	fmul.4s	v3, v3, v7                     ; multiply 4 fp32 values element-wise in v3 and v7 registers
	mov	s7, v3[3]                          ; ...
	mov	s21, v3[2]                         ; ...
	mov	s22, v3[1]                         ; ...
	fmul.4s	v4, v4, v16                    ; multiply 4 fp32 values element-wise in v4 and v16 registers
	mov	s16, v4[3]                         ; ...
	mov	s23, v4[2]                         ; ...
	mov	s24, v4[1]                         ; ...
	fadd	s0, s0, s1                       ; accumulate all 16 products into s0
	fadd	s0, s0, s18
	fadd	s0, s0, s17
	fadd	s0, s0, s5
	fadd	s0, s0, s2
	fadd	s0, s0, s20
	fadd	s0, s0, s19
	fadd	s0, s0, s6
	fadd	s0, s0, s3
	fadd	s0, s0, s22
	fadd	s0, s0, s21
	fadd	s0, s0, s7
	fadd	s0, s0, s4
	fadd	s0, s0, s24
	fadd	s0, s0, s23
	fadd	s0, s0, s16
	subs	x23, x23, #16                    ; chunk_end -= 16
	b.ne	LBB77_13                         ; if (chunk_end != 0) goto LBB_13
	cmp	x5, x10                            ; if (k == 16-element chunk end)
	b.eq	LBB77_6                          ;     goto LBB77_6
	mov	x21, x10                           ; z = 16-element chunk end
	mov	x22, x10                           ;
	cbz	x11, LBB77_19                      ; if (4-element tail == 0) goto LBB77_19

LBB77_16:                                ; 4-element chunks
	add	x21, x16, x22                      ; remain = -(k & ~3) + z
	lsl	x23, x22, #2                       ; byte_offset = z * 4
	add	x22, x19, x23                      ; &b[z * n + y]
	add	x23, x0, x23                       ; &a[x * k + z]

LBB77_17:                                ; for (; z + 4 <= k; z += 4)
	ldr	q1, [x23], #16                     ; load 2 fp32 values from a
	ldr	q2, [x22], #16                     ; load 2 fp32 values from b
	fmul.4s	v1, v1, v2                     ; multiply 4 fp32 values element-wise (2 registers are just 0-valued)
	mov	s2, v1[3]                          ; move fp32 elements into 32-bit registers from 128-bit register
	mov	s3, v1[2]                          ; ...
	mov	s4, v1[1]                          ; ...
	fadd	s0, s0, s1                       ; accumulate all 4 products
	fadd	s0, s0, s4
	fadd	s0, s0, s3
	fadd	s0, s0, s2
	adds	x21, x21, #4                     ; z += 4
	b.ne	LBB77_17                         ; continue if more 4-element chunks available
	mov	x21, x12                           ; z = k & ~3 (4-element chunk end)
	cmp	x5, x12                            ; if (k == z)
	b.eq	LBB77_6                          ;     goto LBB77_6 (end of z loop)

LBB77_19:
	mul	x22, x17, x21                      ; b_offset = z * n * 4

LBB77_20:
	ldr	s1, [x0, x21, lsl #2]              ; s1 = a[x * k + z]
	ldr	s2, [x19, x22]                     ; s2 = b[z * n + y]
	fmadd	s0, s1, s2, s0                   ; accum += s1 * s2
	add	x21, x21, #1                       ; ++z
	add	x22, x22, x17                      ; b_offset += n * 4
	cmp	x5, x21                            ; if (z != k)
	b.ne	LBB77_20                         ;     goto LBB77_20
	b	LBB77_6                              ; goto LBB77_6 (end of z loop)

LBB77_21:
	ldp	x20, x19, [sp, #32]                ; restore register
	ldp	x22, x21, [sp, #16]                ; restore register
	ldp	x24, x23, [sp], #48                ; restore register

LBB77_22:
	ret
```

</details>

For generated assembly with `-O3`, we don't discuss the annotated assembly as it is mostly the same as `-O2` for this very simple implementation of matmul.

### Loop reordering

```c
static void sgemm_2(const float *restrict a, const float *restrict b, float *restrict c, size_t m, size_t n, size_t k) {
  for (size_t x = 0; x < m; ++x) {
    for (size_t z = 0; z < k; ++z) {
      const float axz = a[x * k + z];
      for (size_t y = 0; y < n; ++y)
        c[x * n + y] += axz * b[z * n + y];
    }
  }
}
```

<details>
<summary> Annotated `sgemm_2` assembly </summary>

</details>

### Tiling the K dimension

## References

Goto, Kazushige, and Robert A. Van De Geijn. *“Anatomy of High-Performance Matrix Multiplication.” ACM Transactions on Mathematical Software 34, no. 3 (2008): 1–25. .*

Salykova, Amanzhol. *"Advanced Matrix Multiplication Optimization on Modern Multi-Core Processors." http://salykova.github.io/matmul-cpu.*

Cawley, Peter. *"Apple AMX Instruction Set". https://github.com/corsix/amx*

Spangler, Michael. *"Statically linking on MacOS." https://michaelspangler.io/posts/statically-linking-on-macos.html*
