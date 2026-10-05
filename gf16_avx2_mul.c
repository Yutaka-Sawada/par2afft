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

// Generates the nibble tables for constant x without table lookup.
gf16_nibtab_avx2_t gf16_avx2_gen_nibtab_fly(gf16_t x)
{
    gf16_nibtab_avx2_t res;
    __m128i xmm0, xmm1, xmm2, xmm3, mask8;
    __m256i ymm0, ymm1, base, poly, mask16;

    // create mask for 8-bit
    mask8 = _mm_setzero_si128();
    mask8 = _mm_cmpeq_epi16(mask8, mask8);	// 0xFFFF *8
    mask8 = _mm_srli_epi16(mask8, 8);		// 0x00FF *8

    // put x*(1,2,4,8) in 128-bit registers
    xmm0 = _mm_cvtsi32_si128((int)x);			// [_][_][_][_][_][_][_][1] x*1
    xmm1 = _mm_setzero_si128();
    x = (x << 1) ^ (((short)x >> 15) & 0x100B);
    xmm1 = _mm_insert_epi16(xmm1, (int)x, 1);	// [_][_][_][_][_][_][2][_] x*2
    xmm2 = _mm_setzero_si128();
    x = (x << 1) ^ (((short)x >> 15) & 0x100B);
    xmm2 = _mm_insert_epi16(xmm2, (int)x, 4);	// [_][_][_][4][_][_][_][_] x*4
    xmm1 = _mm_unpacklo_epi16(xmm1, xmm1);		// [_][_][_][_][2][2][_][_]
    x = (x << 1) ^ (((short)x >> 15) & 0x100B);
    xmm3 = _mm_cvtsi32_si128((int)x);			// [_][_][_][_][_][_][_][8] x*8

    // construct x*(7~0) and x*(15~8) in 128-bit registers
    xmm0 = _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(0, 1, 0, 1));	// [_][_][_][_][1][_][1][_]
    xmm3 = _mm_unpacklo_epi16(xmm3, xmm3);						// [_][_][_][_][_][_][8][8]
    xmm0 = _mm_xor_si128(xmm0, xmm1);							// [_][_][_][_][3][2][1][_]
    xmm2 = _mm_shufflehi_epi16(xmm2, _MM_SHUFFLE(0, 0, 0, 0));	// [4][4][4][4][_][_][_][_]
    xmm0 = _mm_unpacklo_epi64(xmm0, xmm0);						// [3][2][1][_][3][2][1][_]
    xmm3 = _mm_shuffle_epi32(xmm3, _MM_SHUFFLE(0, 0, 0, 0));	// [8][8][8][8][8][8][8][8]
    xmm2 = _mm_xor_si128(xmm2, xmm0);							// [ 7][ 6][ 5][ 4][ 3][ 2][1][0] x*(7~0)
    xmm3 = _mm_xor_si128(xmm3, xmm2);							// [15][14][13][12][11][10][9][8] x*(15~8)

    // gather x*(15~0) in 256-bit register
    poly = _mm256_set1_epi32(0x100B100B);	// PRIM_POLY = 0x1100B * 16
    mask16 = _mm256_cmpeq_epi16(poly, poly);
    mask16 = _mm256_srli_epi16(mask16, 8);	// 0x00FF *16
    base = _mm256_setzero_si256();
    base = _mm256_inserti128_si256(base, xmm2, 0);
    base = _mm256_inserti128_si256(base, xmm3, 1);	// x*(15~0)

    // split to low and high
    xmm0 = _mm_and_si128(xmm2, mask8);
    xmm1 = _mm_and_si128(xmm3, mask8);
    xmm0 = _mm_packus_epi16(xmm0, xmm1);	// lower  8-bit * 16
    xmm2 = _mm_srli_epi16(xmm2, 8);
    xmm3 = _mm_srli_epi16(xmm3, 8);
    xmm2 = _mm_packus_epi16(xmm2, xmm3);	// higher 8-bit * 16
    res.lo[0] = _mm256_broadcastsi128_si256(xmm0);
    res.hi[0] = _mm256_broadcastsi128_si256(xmm2);

    // calculate other nibble tables by multipling 16
    for (int k = 1; k < 4; k++){
        // multiply by 2
        ymm0 = _mm256_slli_epi16(base, 1);
        ymm1 = _mm256_srai_epi16(base, 15);
        ymm1 = _mm256_and_si256(ymm1, poly);
        base = _mm256_xor_si256(ymm1, ymm0);

        // multiply by 2
        ymm0 = _mm256_slli_epi16(base, 1);
        ymm1 = _mm256_srai_epi16(base, 15);
        ymm1 = _mm256_and_si256(ymm1, poly);
        base = _mm256_xor_si256(ymm1, ymm0);

        // multiply by 2
        ymm0 = _mm256_slli_epi16(base, 1);
        ymm1 = _mm256_srai_epi16(base, 15);
        ymm1 = _mm256_and_si256(ymm1, poly);
        base = _mm256_xor_si256(ymm1, ymm0);

        // multiply by 2
        ymm0 = _mm256_slli_epi16(base, 1);
        ymm1 = _mm256_srai_epi16(base, 15);
        ymm1 = _mm256_and_si256(ymm1, poly);
        base = _mm256_xor_si256(ymm1, ymm0);

        ymm0 = _mm256_and_si256(base, mask16);	// lower  8-bit * 16
        ymm1 = _mm256_srli_epi16(base, 8);		// higher 8-bit * 16
        ymm0 = _mm256_packus_epi16(ymm0, ymm0);		// lower  8-bit * 32
        ymm1 = _mm256_packus_epi16(ymm1, ymm1);		// higher 8-bit * 32
        res.lo[k] = _mm256_permute4x64_epi64(ymm0, 0x88);	// swap order
        res.hi[k] = _mm256_permute4x64_epi64(ymm1, 0x88);
    }

    return res;
}

void gf16_avx2_mul_init() {
    static char initialized = 0;
    if (initialized) return;

    for (int i = 0; i < GF16_ORDER; ++i) {
        gf16_nibtab_avx2[i] = gf16_avx2_gen_nibtab(i);
        //gf16_nibtab_avx2[i] = gf16_avx2_gen_nibtab_fly(i); // test instant table generation
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

    __m256i i0 = _mm256_and_si256(x, mask); // no need shift for the lowest 4-bit
    __m256i i1 = _mm256_and_si256(_mm256_srli_epi16(x,  4), mask);
    __m256i i2 = _mm256_and_si256(_mm256_srli_epi16(x,  8), mask);
    __m256i i3 = _mm256_srli_epi16(x, 12);  // no need mask for the highest 4-bit

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

// remove inline functions, but no speed difference
void gf16_avx2_nibmul16_and_xor(
        gf16_t *restrict          dst,
        const gf16_t *restrict    src,
        const gf16_nibtab_avx2_t* tab,
        size_t                    n)
{
    assert(n % 16 == 0);
    // The number of using YMM registers is 16.
    const __m256i mask = _mm256_set1_epi16(0x000f);
    __m256i x, xx, r;
    __m256i i0, i1, i2, i3;
    __m256i tl0, tl1, tl2, tl3, th0, th1, th2, th3;

    tl0 = _mm256_loadu_si256((__m256i const*)tab->lo);
    tl1 = _mm256_loadu_si256((__m256i const*)(tab->lo + 1));
    tl2 = _mm256_loadu_si256((__m256i const*)(tab->lo + 2));
    tl3 = _mm256_loadu_si256((__m256i const*)(tab->lo + 3));
    th0 = _mm256_loadu_si256((__m256i const*)tab->hi);
    th1 = _mm256_loadu_si256((__m256i const*)(tab->hi + 1));
    th2 = _mm256_loadu_si256((__m256i const*)(tab->hi + 2));
    th3 = _mm256_loadu_si256((__m256i const*)(tab->hi + 3));

    for (size_t i = 0; i + 16 <= n; i += 16) {
        x = _mm256_loadu_si256((const __m256i *)(src + i));

        i0 = _mm256_and_si256(x, mask); // no need shift for the lowest 4-bit
        i1 = _mm256_and_si256(_mm256_srli_epi16(x,  4), mask);
        i2 = _mm256_and_si256(_mm256_srli_epi16(x,  8), mask);
        i3 = _mm256_srli_epi16(x, 12);  // no need mask for the highest 4-bit

        xx = _mm256_shuffle_epi8(tl0, i0);
        i0 = _mm256_slli_epi16(i0, 8);
        r  = _mm256_shuffle_epi8(th0, i0); // result of look up
        xx = _mm256_xor_si256(xx, r);      // combine results

        r  = _mm256_shuffle_epi8(tl1, i1);
        xx = _mm256_xor_si256(xx, r);
        i1 = _mm256_slli_epi16(i1, 8);
        r  = _mm256_shuffle_epi8(th1, i1);
        xx = _mm256_xor_si256(xx, r);

        r = _mm256_shuffle_epi8(tl2, i2);
        xx = _mm256_xor_si256(xx, r);
        i2 = _mm256_slli_epi16(i2, 8);
        r = _mm256_shuffle_epi8(th2, i2);
        xx = _mm256_xor_si256(xx, r);

        r = _mm256_shuffle_epi8(tl3, i3);
        xx = _mm256_xor_si256(xx, r);
        i3 = _mm256_slli_epi16(i3, 8);
        r = _mm256_shuffle_epi8(th3, i3);
        xx = _mm256_xor_si256(xx, r);

        x = _mm256_loadu_si256((const __m256i *)(dst + i));
        x = _mm256_xor_si256(x, xx);
        _mm256_storeu_si256((__m256i *)(dst + i), x);
    }
}

// Multiplies 32 elements by the same constant implied by `tab`.
//
// x contains 16 16-bit elements: x0 x1 x2 .. x15
// y contains 16 16-bit elements: y0 y1 y2 .. y15
//
// We use XOR and AND operations to create 8 indices corresponding
// with the 4 nibbles:
//
//  i0 contains 32 elements, the i-th is the bits 0..3 of x_i and 0..3 of y_i
//  i1 contains 32 elements, the i-th is the bits 4..7 of x_i and 4..7 of y_i
//  i2 contains 32 elements, the i-th is the bits 8..11 of x_i and 8..11 of y_i
//  i3 contains 32 elements, the i-th is the bits 12..15 of x_i and 12..15 of y_i
//
// Then we use those index vectors to look up the products of all nibbles in
// the nibble table in parallel, and finally combine the result with XOR.
void gf16_avx2_nibmul32_and_xor(
        gf16_t *restrict          dst,
        const gf16_t *restrict    src,
        const gf16_nibtab_avx2_t* tab,
        size_t                    n)
{
    assert(n % 16 == 0);
    size_t i;
    const __m256i mask = _mm256_set1_epi16(0x000f);
    const __m256i mask_lo = _mm256_set1_epi16(0x00ff);
    const __m256i mask_hi = _mm256_set1_epi16(0xff00);
    __m256i x, y, lo, hi, xx, yy;
    __m256i i0, i1, i2, i3;

    // process 64-bytes per loop
    for (i = 0; i + 32 <= n; i += 32) {
        x = _mm256_loadu_si256((const __m256i *)(src + i     ));
        y = _mm256_loadu_si256((const __m256i *)(src + i + 16));

        i0 = _mm256_and_si256(x, mask); // pick 16 elements of x's 4-bit
        yy = _mm256_and_si256(y, mask); // pick 16 elements of y's 4-bit
        yy = _mm256_slli_epi16(yy, 8);  // move y's 16 elements to upper 8-bit
        i0 = _mm256_xor_si256(i0, yy);  // combine them to construct 32 elements of 4-bit

        i1 = _mm256_and_si256(_mm256_srli_epi16(x,  4), mask);
        yy = _mm256_and_si256(_mm256_srli_epi16(y,  4), mask);
        yy = _mm256_slli_epi16(yy, 8);
        i1 = _mm256_xor_si256(i1, yy);

        i2 = _mm256_and_si256(_mm256_srli_epi16(x,  8), mask);
        yy = _mm256_and_si256(_mm256_srli_epi16(y,  8), mask);
        yy = _mm256_slli_epi16(yy, 8);
        i2 = _mm256_xor_si256(i2, yy);

        i3 = _mm256_srli_epi16(x, 12);
        yy = _mm256_srli_epi16(y, 12);
        yy = _mm256_slli_epi16(yy, 8);
        i3 = _mm256_xor_si256(i3, yy);

        lo = _mm256_shuffle_epi8(tab->lo[0], i0); // 32 elements of lower  8-bit
        hi = _mm256_shuffle_epi8(tab->hi[0], i0); // 32 elements of higher 8-bit
        xx = _mm256_and_si256(lo, mask_lo); // 16 elements of x's lower  8-bit
        yy = _mm256_srli_epi16(lo, 8);      // 16 elements of y's lower  8-bit
        x  = _mm256_slli_epi16(hi, 8);      // 16 elements of x's higher 8-bit
        y  = _mm256_and_si256(hi, mask_hi); // 16 elements of y's higher 8-bit
        xx = _mm256_xor_si256(xx, x); // combine high and low to construct 16 elements of x's 16-bit
        yy = _mm256_xor_si256(yy, y); // combine high and low to construct 16 elements of x's 16-bit

        lo = _mm256_shuffle_epi8(tab->lo[1], i1);
        hi = _mm256_shuffle_epi8(tab->hi[1], i1);
        x = _mm256_and_si256(lo, mask_lo);
        y = _mm256_srli_epi16(lo, 8);
        xx = _mm256_xor_si256(xx, x);
        yy = _mm256_xor_si256(yy, y);
        x = _mm256_slli_epi16(hi, 8);
        y = _mm256_and_si256(hi, mask_hi);
        xx = _mm256_xor_si256(xx, x);
        yy = _mm256_xor_si256(yy, y);

        lo = _mm256_shuffle_epi8(tab->lo[2], i2);
        hi = _mm256_shuffle_epi8(tab->hi[2], i2);
        x = _mm256_and_si256(lo, mask_lo);
        y = _mm256_srli_epi16(lo, 8);
        xx = _mm256_xor_si256(xx, x);
        yy = _mm256_xor_si256(yy, y);
        x = _mm256_slli_epi16(hi, 8);
        y = _mm256_and_si256(hi, mask_hi);
        xx = _mm256_xor_si256(xx, x);
        yy = _mm256_xor_si256(yy, y);

        lo = _mm256_shuffle_epi8(tab->lo[3], i3);
        hi = _mm256_shuffle_epi8(tab->hi[3], i3);
        x = _mm256_and_si256(lo, mask_lo);
        y = _mm256_srli_epi16(lo, 8);
        xx = _mm256_xor_si256(xx, x);
        yy = _mm256_xor_si256(yy, y);
        x = _mm256_slli_epi16(hi, 8);
        y = _mm256_and_si256(hi, mask_hi);
        xx = _mm256_xor_si256(xx, x);
        yy = _mm256_xor_si256(yy, y);

        x = _mm256_loadu_si256((const __m256i *)(dst + i     ));
        y = _mm256_loadu_si256((const __m256i *)(dst + i + 16));
        x = _mm256_xor_si256(x, xx);
        y = _mm256_xor_si256(y, yy);
        _mm256_storeu_si256((__m256i *)(dst + i     ), x);
        _mm256_storeu_si256((__m256i *)(dst + i + 16), y);
    }

    // process rest 32-bytes
    if (i + 16 <= n) {
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
    //gf16_avx2_nibmul_and_xor(dst, src, &gf16_nibtab_avx2[c], n);

    gf16_nibtab_avx2_t tab = gf16_avx2_gen_nibtab_fly(c); // generate nibble tables on the fly
    gf16_avx2_nibmul_and_xor(dst, src, &tab, n);
    //gf16_avx2_nibmul32_and_xor(dst, src, &tab, n); // process 64-bytes per loop
}
