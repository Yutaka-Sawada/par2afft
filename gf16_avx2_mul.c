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
//
// This code is implemented by Anime Tosho, who is author of ParPar and par2cmdline-turbo.
gf16_nibtab_avx2_t gf16_avx2_gen_nibtab_fly(gf16_t x)
{
	gf16_nibtab_avx2_t res;
	__m128i xmm0, xmm1;
	__m256i base;

	// calc x*2, x*4, x*8
	gf16_t x2 = (x << 1) ^ (((short)x >> 15) & 0x100B);
	gf16_t x4 = (x2 << 1) ^ (((short)x2 >> 15) & 0x100B);
	gf16_t x8 = (x4 << 1) ^ (((short)x4 >> 15) & 0x100B);

	// put x*(1,2,4,8) in 128-bit registers
	xmm0 = _mm_cvtsi32_si128((int)x << 16);			// [_][_][_][_][_][_][1][0] x*1
	xmm0 = _mm_insert_epi16(xmm0, x2, 2);			// [_][_][_][_][_][2][1][0]
	xmm0 = _mm_insert_epi16(xmm0, x2 ^ x, 3);		// [_][_][_][_][3][2][1][0]
	xmm1 = _mm_set1_epi16(x4);						// [4][4][4][4][4][4][4][4] x*4
	xmm1 = _mm_xor_si128(xmm1, xmm0);				// [4][4][4][4][7][6][5][4]
	xmm0 = _mm_unpacklo_epi64(xmm0, xmm1);			// [7][6][5][4][3][2][1][0] x*(7~0)
	xmm1 = _mm_set1_epi16(x8);						// [8][8][8][8][8][8][8][8] x*8
	xmm1 = _mm_xor_si128(xmm1,xmm0);				// [f][e][d][c][b][a][9][8] x*(0xf~8)

	// combine into a single YMM register
	base = _mm256_inserti128_si256(_mm256_castsi128_si256(xmm0), xmm1, 1);

	// move high bytes into bottom half, bottom bytes into top half
	base = _mm256_shuffle_epi8(base, _mm256_set_epi32(
		0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200,
		0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200
	));
	base = _mm256_permute4x64_epi64(base, _MM_SHUFFLE(2,0,3,1));
	// for the first two words, [ab][cd] and [ef][gh],
	// where each letter represents a nibble and brackets denote a byte,
	// they're now arranged in base like:
	// ...[gh][cd]...[ef][ab]

	// broadcast halves for final table
	res.lo[0] = _mm256_permute4x64_epi64(base, _MM_SHUFFLE(3,2,3,2));
	res.hi[0] = _mm256_inserti128_si256(base, _mm256_castsi256_si128(base), 1);

	const __m256i mask = _mm256_set1_epi8(0xf);
	// this is the multiply-by-16 reduction (lower half) constant - see code below for how it's computed
	// we only need the lower half because the upper half of the 0x1100b polynomial can be done with just a shift+xor
	const __m128i reduce = _mm_set_epi32(0x69627f74, 0x454e5358, 0x313a272c, 0x1d160b00);
	__m256i lo, hi, idx;
	__m128i reduced;

	// multiply by 64 three times
	// this essentially is done via `(base << 4) ^ reduction[base >> 12]` but in vector form,
	// and with split high/low bytes
	for (int k = 1; k < 4; k++) {
		// isolate nibbles
		hi = _mm256_andnot_si256(mask, base);		// ...[g ][c ]...[e ][a ]
		lo = _mm256_and_si256(mask, base);			// ...[ h][ d]...[ f][ b]

		// lookup top 4 bits of each word to get the result of modular reduction
		// remember that the top half of each word is in the *lower* half of the YMM register,
		// so we only need to do the lookup on the lower half of the YMM
		idx = _mm256_srli_epi16(hi, 4);				// ...[ g][ c]...[ e][ a]
		reduced = _mm_shuffle_epi8(reduce, _mm256_castsi256_si128(idx));

		// overwrite the low half of each word with the reduction result
		// because the polynomial is 0x1100b, where the top half is 0x110,
		// the top 4 bits of each word will be shifted down by 4 and xor'd with bits 8-11.
		// Since multiplying by 16 shifts everything up by 4,
		// these top 4 bits don't need to move, and can just stay in place
		hi = _mm256_inserti128_si256(hi, reduced, 1);	// ...[rr][rr]...[e ][a ]
		// `hi` now has the full result of modular reduction

		// multiply by 16 by shifting left
		// next line multiplies bits 0-3 and 8-11
		lo = _mm256_slli_epi16(lo, 4);				// ...[h ][d ]...[f ][b ]
		// bits 4-7 needs to be moved from the top half of the YMM register to the bottom,
		// because it wraps around into the next byte
		idx = _mm256_zextsi128_si256(_mm256_extracti128_si256(idx, 1));
		lo = _mm256_or_si256(lo, idx);				// ...[h ][d ]...[fg][bc]

		// combine the product with modular reduction
		base = _mm256_xor_si256(lo, hi);

		// broadcast halves for final table
		res.lo[k] = _mm256_permute4x64_epi64(base, _MM_SHUFFLE(3,2,3,2));
		res.hi[k] = _mm256_inserti128_si256(base, _mm256_castsi256_si128(base), 1);
	}

    return res;

	/*
`reduce` can be computed via

	uint16_t _poly[16];
	int polynomial = 0x1100b;
	__m128i tmp1, tmp2;
	for(int i=0; i<16; i++) {
		int p = 0;
		if(i & 8) p ^= polynomial << 3;
		if(i & 4) p ^= polynomial << 2;
		if(i & 2) p ^= polynomial << 1;
		if(i & 1) p ^= polynomial << 0;
		
		_poly[i] = p & 0xffff;
	}
	tmp1 = _mm_loadu_si128((__m128i*)_poly);
	tmp2 = _mm_loadu_si128((__m128i*)_poly + 1);
	tmp1 = _mm_shuffle_epi8(tmp1, _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200));
	tmp2 = _mm_shuffle_epi8(tmp2, _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200));
	reduce = _mm_unpacklo_epi64(tmp1, tmp2);

	*/
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
//
// This code is implemented by Anime Tosho, who is author of ParPar and par2cmdline-turbo.
void gf16_avx2_nibmul32_and_xor(
        gf16_t *restrict          dst,
        const gf16_t *restrict    src,
        const gf16_nibtab_avx2_t* tab,
        size_t                    n)
{
	assert(n % 16 == 0);
	size_t i;
	const __m256i mask = _mm256_set1_epi8(0x0f);
	// shuffle vector to split 8x 16b words,
	// moving low halves to the low 8 bytes of the vector
	// and upper halves to the upper 8 bytes
	const __m256i sep_lo_hi = _mm256_set_epi32(
		0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200,
		0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200
	);
	__m256i x, y, lo, hi, xx, yy;
	__m256i i0, i1, i2, i3;

	// process 64-bytes per loop
	for (i = 0; i + 32 <= n; i += 32) {
		x = _mm256_loadu_si256((const __m256i *)(src + i     ));
		y = _mm256_loadu_si256((const __m256i *)(src + i + 16));

		// deinterleave bytes
		x = _mm256_shuffle_epi8(x, sep_lo_hi);
		y = _mm256_shuffle_epi8(y, sep_lo_hi);
		i0 = _mm256_unpacklo_epi64(x, y);   // low bytes of 32x words
		i2 = _mm256_unpackhi_epi64(x, y);   // high bytes of 32x words

		// isolate lookup nibbles
		i1 = _mm256_srli_epi16(i0, 4);
		i1 = _mm256_and_si256(i1, mask);
		i3 = _mm256_srli_epi16(i2, 4);
		i3 = _mm256_and_si256(i3, mask);
		i0 = _mm256_and_si256(i0, mask);
		i2 = _mm256_and_si256(i2, mask);

		// perform nibble lookups
		xx = _mm256_shuffle_epi8(tab->lo[0], i0); // 32 elements of lower  8-bit
		yy = _mm256_shuffle_epi8(tab->hi[0], i0); // 32 elements of higher 8-bit

		x = _mm256_shuffle_epi8(tab->lo[1], i1);
		y = _mm256_shuffle_epi8(tab->hi[1], i1);
		xx = _mm256_xor_si256(xx, x);
		yy = _mm256_xor_si256(yy, y);

		x = _mm256_shuffle_epi8(tab->lo[2], i2);
		y = _mm256_shuffle_epi8(tab->hi[2], i2);
		xx = _mm256_xor_si256(xx, x);
		yy = _mm256_xor_si256(yy, y);

		x = _mm256_shuffle_epi8(tab->lo[3], i3);
		y = _mm256_shuffle_epi8(tab->hi[3], i3);
		xx = _mm256_xor_si256(xx, x);
		yy = _mm256_xor_si256(yy, y);

		// interleave the bytes to get words
		lo = _mm256_unpacklo_epi8(xx, yy);
		hi = _mm256_unpackhi_epi8(xx, yy);

		x = _mm256_loadu_si256((const __m256i *)(dst + i     ));
		y = _mm256_loadu_si256((const __m256i *)(dst + i + 16));
		x = _mm256_xor_si256(x, lo);
		y = _mm256_xor_si256(y, hi);
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
    gf16_avx2_nibmul_and_xor(dst, src, &gf16_nibtab_avx2[c], n);

    //gf16_nibtab_avx2_t tab = gf16_avx2_gen_nibtab_fly(c); // generate nibble tables on the fly
    //gf16_avx2_nibmul_and_xor(dst, src, &tab, n);
    //gf16_avx2_nibmul32_and_xor(dst, src, &tab, n); // process 32 words per loop
}
