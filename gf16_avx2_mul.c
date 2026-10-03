// Implements GF(2^16) multiplication by a constant using nibble tables.
//
// The idea is that multiplication is a linear operation:
//
//  gf16_mul(x, a + b) = gf16_mul(x, a) + gf16_mul(x, b)
//
// Of course "+" in the finite field is just "^" (XOR).
//
// That means to calculate gf16_mul(x, y) where x is constant, we can split y
// into 4 x 4 bits (nibbles):
//
//   a = y & 0x000f
//   b = y & 0x00f0
//   c = y & 0x0f00
//   d = y & 0xf000
//
// and calculate:
//
//  gf16_mul(x, y) = gf16_mul(x, a) ^ gf16_mul(x, b) ^ gf16_mul(x, c)  ^ gf16_mul(x, d)
//
// Note that each variable varies only in 4 bits. For a fixed value of x, we can
// precompute the possible products for each nibble in an array:
//
//      gf16_t mul[4][16]    // 4 * 16 * 2 = 128 bits
//
// Defined as:
//
//      mul[i][j] = gf16_mul(x, j << (4*i))
//
// Then we can write:
//
//  gf16_mul(x, y) = mul[0][a] ^ mul[1][b] ^ mul[2][c] ^ mul[3][d]
//
// The array `mul` is the nibble table for the constant `x`.
//
// Instead of storing it as an array, however, we will store it directly in
// 256 bit AVX registers, and use PSHUFB to do the indexing. However, since
// PSHUFB only shuffles bytes, we need to split the product into two parts,
// effectively creating two tables:
//
//    mul_lo[i][j] = gf16_mul(x, j << (4*i)) & 0xff
//    mul_hi[i][j] = gf16_mul(x, j << (4*i)) >> 8

#include "gf16_avx2_mul.h"

#include "gf16.h"

#include <assert.h>
#include <immintrin.h>

#include <stdio.h> // debug printing

static gf16_nibtab_avx2_t gf16_nibtab_avx2[GF16_ORDER];

static void debug_print(void *buf, size_t len) {
    uint8_t *p = buf;
    while (len--) printf("%02x", *p++);
    printf("\n");
}

static void debug_print_reg(__m256i reg) {
    uint8_t bytes[32];
    _mm256_storeu_si256((__m256i*)bytes, reg);
    debug_print(bytes, sizeof(bytes));
}

// Generates the nibble tables for constant x.
//
// Note: because PSHUFB (used in lookup16) treats 256-bit registers as two lanes
// of 128 bits each, we use broadcast to double each result.
gf16_nibtab_avx2_t gf16_avx2_gen_nibtab(gf16_t x) {
    gf16_nibtab_avx2_t res;
    for (int k = 0; k < 4; ++k) {
        uint8_t lo_bytes[16];
        uint8_t hi_bytes[16];
        for (int n = 0; n < 16; ++n) {
            gf16_t y = n << (4*k);
            gf16_t z = gf16_mul(x, y);
            lo_bytes[n] = (z >> 0) & 0xff;
            hi_bytes[n] = (z >> 8) & 0xff;
        }
        __m128i lo = _mm_loadu_si128((const __m128i *)lo_bytes);
        __m128i hi = _mm_loadu_si128((const __m128i *)hi_bytes);
        res.lo[k] = _mm256_broadcastsi128_si256(lo);
        res.hi[k] = _mm256_broadcastsi128_si256(hi);
    }
    return res;
}

void gf16_avx2_mul_init() {
    static char initialized = 0;
    if (initialized) return;

    for (int i = 0; i < GF16_ORDER; ++i) {
        gf16_nibtab_avx2[i] = gf16_avx2_gen_nibtab(i);
    }

    initialized = 1;
}

// Uses the nibble tables to look up 16 values at once.
//
// idx contains 16 x 16-bit integers below 16.
//
// lo_tab and hi_tab contain the low and high byte of the multiplication
// results, each doubled since PSHUFB
//
// Note the shuffle instruction actually looks up 32 bytes (not 16 16-bit
// values). We rely on the fact that multiplication by 0 is 0 so the zero
// bytes resolve to zero in the lookup:
//
//   i0          0  i1          0  i2          0  ..  i15         0
//   lo_tab[i0]  0  lo_tab[i1]  0  lo_tab[i2]  0  ..  lo_tab[15]  0
//
// Then after shifting `idx` 8 bits to the left:
//
//   0  i0          0  i1          0  i2          ..  0  i15
//   0  hi_tab[i0]  0  hi_tab[i1]  0  hi_tab[i2]  ..  0  hi_tab[idx15]
//
// and OR'ing the result together the result vector becomes:
//
//  lo_tab[i0] hi_tab[i0] lo_tab[i1] hi_tab[i1] .. lo_tab[i15] hi_tab[i15]
//
static inline __m256i lookup16(__m256i lo_tab, __m256i hi_tab, __m256i idx)
{
    __m256i lo = _mm256_shuffle_epi8(lo_tab, idx);
    __m256i hi_idx = _mm256_slli_epi16(idx, 8);
    __m256i hi = _mm256_shuffle_epi8(hi_tab, hi_idx);
    return _mm256_or_si256(lo, hi);
}

// Multiplies 16 elements by the same constant implied by `tab`.
//
// x contains 16 16-bit elements: x0 x1 x2 .. x15
//
// We use XOR and AND operations to create 4 indices corresponding
// with the 4 nibbles:
//
//  i0 contains 16 elements, the i-th is the bits 0..3 of x_i
//  i1 contains 16 elements, the i-th is the bits 4..7 of x_i
//  i2 contains 16 elements, the i-th is the bits 8..11 of x_i
//  i3 contains 16 elements, the i-th is the bits 12..15 of x_i
//
// Then we use those index vectors to look up the products of all nibbles in
// the nibble table in parallel, and finally combine the result with XOR.
static inline __m256i nibmul16(const gf16_nibtab_avx2_t *tab, __m256i x) {
    const __m256i mask = _mm256_set1_epi16(0x000f);

    __m256i i0 = _mm256_and_si256(_mm256_srli_epi16(x,  0), mask);
    __m256i i1 = _mm256_and_si256(_mm256_srli_epi16(x,  4), mask);
    __m256i i2 = _mm256_and_si256(_mm256_srli_epi16(x,  8), mask);
    __m256i i3 = _mm256_and_si256(_mm256_srli_epi16(x, 12), mask);

    __m256i r0 = lookup16(tab->lo[0], tab->hi[0], i0);
    __m256i r1 = lookup16(tab->lo[1], tab->hi[1], i1);
    __m256i r2 = lookup16(tab->lo[2], tab->hi[2], i2);
    __m256i r3 = lookup16(tab->lo[3], tab->hi[3], i3);

    return _mm256_xor_si256(
        _mm256_xor_si256(r0, r1),
        _mm256_xor_si256(r2, r3));
}

void gf16_avx2_nibmul_and_xor(
        gf16_t *restrict          dst,
        const gf16_t *restrict    src,
        const gf16_nibtab_avx2_t* tab,
        size_t                    n)
{
    assert(n % 16 == 0);
    for (size_t i = 0; i + 16 <= n; i += 16) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i b = nibmul16(tab, a);
        __m256i c = _mm256_loadu_si256((const __m256i *)(dst + i));
        __m256i d = _mm256_xor_si256(b, c);
        _mm256_storeu_si256((__m256i *)(dst + i), d);
    }
}

void gf16_avx2_mul_and_xor(
        gf16_t *restrict       dst,
        const gf16_t *restrict src,
        gf16_t                 c,
        size_t                 n)
{
    gf16_avx2_nibmul_and_xor(dst, src, &gf16_nibtab_avx2[c], n);
}
