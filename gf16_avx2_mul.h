#ifndef GF16_AVX2_MUL_INCLUDED
#define GF16_AVX2_MUL_INCLUDED

#include "gf16.h"
#include "immintrin.h"

#include <stddef.h>

// Initalizes the libary
extern void gf16_avx2_mul_init();

// Performs GF16 multiplication with a constant on an array of `n` elements
// in `src`, and XOR's the result with `dst`.
//
// That is, it computes the equivalent of:
//
//  for (size_t i = 0; i < n; ++i) {
//      dst[i] ^= gf16_mul(src[i], c);
//  }
//
// `n` MUST be a multiple of 16.
// The source and destination ranges MUST NOT overlap.
//
void gf16_avx2_mul_and_xor(
    gf16_t *restrict       dst,
    const gf16_t *restrict src,
    gf16_t                 c,
    size_t                 n);

//
// Following implementation details are exposed for gf16_vt_batch_mul.h
//

typedef struct gf16_nibtab_avx2 {
    __m256i lo[4];
    __m256i hi[4];
} gf16_nibtab_avx2_t;

// Generates the nibble table for multiplication by a constant x.
extern gf16_nibtab_avx2_t gf16_avx2_gen_nibtab(gf16_t x);

// Performs batch multiplication like gf16_avx2_mul_and_xor() but using an
// explicit nibble table pointer instead.
extern void gf16_avx2_nibmul_and_xor(
    gf16_t *restrict          dst,
    const gf16_t *restrict    src,
    const gf16_nibtab_avx2_t* tab,
    size_t                    n);

#endif  // ndef GF16_AVX2_MUL_INCLUDED
