// This file was adapted from output produced by ChatGPT 5.6 Sol (high)
// which generated the original recursive formulations. The iterative algorithms
// were written by hand.

#include "gf16_vandermonde_transpose.h"
#include "gf16.h"

#include <assert.h>
#include <stdint.h>

#ifndef ITERATIVE_AFFT
#define ITERATIVE_AFFT 1
#endif

#ifndef SUBSPACE_POLY_LUT
#define SUBSPACE_POLY_LUT 1
#endif

gf16_t cantor_permutation[GF16_ORDER];

gf16_t cantor_beta[GF16_BITS];

// beta_prefix[i] = is the XOR sum of the first i elements of beta
// for 0 <= i <= 16. Additionally, we have beta_prefix[17] = beta_prefix[16]
// which allows us to drop bounds checking.
static gf16_t beta_prefix[GF16_BITS + 2];

#if SUBSPACE_POLY_LUT
static gf16_t subspace_poly_eval_lo[16][256];
static gf16_t subspace_poly_eval_hi[16][256];
#endif

gf16_t subspace_poly_for_afft[GF16_ORDER - 1];

/*
 * Full-field transpose Vandermonde example over
 *
 *   GF(2^16) = GF(2)[z] / (z^16 + z^12 + z^3 + z + 1)
 *
 * polynomial encoded as 0x1100B.  A field element is the 16-bit
 * polynomial-basis representation in z.
 *
 * We compute, for arbitrary a[x], x = 0..65535,
 *
 *   y[k] = sum_{x in GF(2^16)} a[x] * x^k,   k = 0..65535.
 *
 * Equivalently y = V^T a where V[x,k] = x^k.
 *
 * Internally:
 *   V = P^{-1} F C
 *
 * where
 *   C : monomial coefficients -> Cantor novel-basis coefficients,
 *   F : additive FFT, output in Cantor-coordinate point order,
 *   P : permutation between polynomial-basis field labels and Cantor order.
 *
 * Hence
 *   V^T = C^T F^T P.
 *
 * The monomial<->novel conversion is recursive and multiplication-free.
 * No 65536-entry "lower[]" table is used.
 */

/* Absolute trace GF(2^16) -> GF(2). */
static gf16_t gf16_trace(gf16_t a)
{
    gf16_t t = 0;
    gf16_t x = a;

    for (unsigned i = 0; i < GF16_BITS; ++i) {
        t ^= x;
        x = gf16_sqr(x);
    }

    assert(t == 0 || t == 1);
    return t;
}

/* ------------------------------------------------------------------------- */
/* Cantor basis generation                                                   */
/* ------------------------------------------------------------------------- */

/*
 * Let A(x) = x^2 + x.  A Cantor chain satisfies
 *
 *   beta[0] = 1,
 *   A(beta[i]) = beta[i-1].
 *
 * For extension degree 16, pick any absolute-trace-one beta[15], then
 * repeatedly apply A downward.  We choose the numerically smallest trace-one
 * element so the result is deterministic and reproducible.
 */
static void generate_cantor_basis()
{
    gf16_t top = 0;

    for (uint32_t x = 1; x < GF16_ORDER; ++x) {
        if (gf16_trace((gf16_t)x) == 1) {
            top = (gf16_t)x;
            break;
        }
    }

    assert(top != 0);

    cantor_beta[GF16_BITS - 1] = top;
    for (unsigned i = GF16_BITS - 1; i > 0; --i)
        cantor_beta[i - 1] = gf16_sqr(cantor_beta[i]) ^ cantor_beta[i];

    assert(cantor_beta[0] == 1);

    for (unsigned i = 1; i < GF16_BITS; ++i)
        assert((gf16_sqr(cantor_beta[i]) ^ cantor_beta[i]) == cantor_beta[i - 1]);

    beta_prefix[0] = 0;
    for (unsigned j = 0; j < GF16_BITS; ++j)
        beta_prefix[j + 1] = beta_prefix[j] ^ cantor_beta[j];
    beta_prefix[GF16_BITS + 1] = beta_prefix[GF16_BITS];
}

static void build_cantor_permutation()
{
    cantor_permutation[0] = 0;
    for (uint32_t j = 1; j < GF16_ORDER; ++j) {
        const unsigned b = __builtin_ctz(j);
        cantor_permutation[j] = cantor_permutation[j ^ (1 << b)] ^ cantor_beta[b];
    }
}

/* ------------------------------------------------------------------------- */
/* Subspace polynomials                                                      */
/* ------------------------------------------------------------------------- */


#if SUBSPACE_POLY_LUT

// Faster version of subspace_poly_eval() using a split lookup table.
//
// This works because A(x) = x^2 + x is linear in fields of characteristic 2:
//
//  A(x + y) = (x + y)^2 + (x + y)
//           = x^2 + 2xy + y^2 + x + y      (note 2xy vanishes in GF(2))
//           = x^2 + x + y^2 + y
//           = A(x) + A(y)
//
// Note: unlike the original implementation, this one requires i < 16.
gf16_t subspace_poly_eval(unsigned i, gf16_t x)
{
    return
        subspace_poly_eval_lo[i][x & 0xff] ^
        subspace_poly_eval_hi[i][x >> 8];
}

static void build_subspace_poly_eval_tables(void)
{
    for (int b = 0; b < 256; ++b) {
        subspace_poly_eval_lo[0][b] = b;
        subspace_poly_eval_hi[0][b] = b << 8;
    }

    for (int i = 1; i < 16; ++i) {
        for (int b = 0; b < 256; ++b) {
            gf16_t x_lo = subspace_poly_eval_lo[i - 1][b];
            subspace_poly_eval_lo[i][b] = gf16_sqr(x_lo) ^ x_lo;

            gf16_t x_hi = subspace_poly_eval_hi[i - 1][b];
            subspace_poly_eval_hi[i][b] = gf16_sqr(x_hi) ^ x_hi;
        }
    }
}

#else  // !SUBSPACE_POLY_LUT

/* Evaluate s_i(x) = A^i(x), where A(x)=x^2+x. */
gf16_t subspace_poly_eval(unsigned i, gf16_t x)
{
    for (unsigned r = 0; r < i; ++r)
        x = gf16_sqr(x) ^ x;
    return x;
}

#endif

static void build_subspace_poly_for_afft(void) {
    gf16_t *p = subspace_poly_for_afft;
    for (int i = 0; i < GF16_BITS; ++i) {
        assert(p == subspace_poly_for_afft + SUBSPACE_POLY_OFFSET(i));
        const int block_count = GF16_ORDER >> (i + 1);
        gf16_t alpha = 0;
        for (int block_index = 0; block_index < block_count; ++block_index) {
            gf16_t c = subspace_poly_eval(i, alpha);
            gf16_t log_c = c == 0 ? 65535 : gf16_log(c);
            *p++ = log_c;

            // Invariant: block_alpha is the XOR sum of beta[j] for all bits j set in block_index.
            const unsigned z = __builtin_ctz(block_index + 1);
            alpha ^= beta_prefix[i + z + 2] ^ beta_prefix[i + 1];
        }
    }
    assert(p - subspace_poly_for_afft == GF16_ORDER - 1);
}

#if !ITERATIVE_AFFT
/*
 * Transpose C^T.
 *
 * Forward division uses operations
 *
 *   a[t+e] ^= a[h+t]
 *
 * in descending t order.  Transposition reverses operation order and swaps
 * source/destination:
 *
 *   a[h+t] ^= a[t+e]
 *
 * in ascending t order.  Because forward C recurses after division, C^T
 * recurses first and applies the transposed division afterward.
 */
static void monomial_to_novel_transpose_rec(gf16_t *a, unsigned m)
{
    if (m == 0)
        return;

    const unsigned i = m - 1;
    const uint32_t h = 1u << i;

    monomial_to_novel_transpose_rec(a,     m - 1);
    monomial_to_novel_transpose_rec(a + h, m - 1);

    for (uint32_t t = 0; t < h; ++t) {
        gf16_t c = a[h + t];
        for (unsigned r = 0; r < shape[i].count; ++r)
            c ^= a[t + shape[i].exponent[r]];
        a[h + t] = c;
    }
}

#define monomial_to_novel_transpose(a) \
        monomial_to_novel_transpose_rec(a, GF16_BITS)

#else

/* Iterative version of monomial_to_novel_transpose_rec() defined above.

This was unrolled manually by an overeducated ape based on the shape[] table
generated by build_subspace_shapes():

  i= 0 count= 0
  i= 1 count= 1   1
  i= 2 count= 1   1
  i= 3 count= 3   1 2 4
  i= 4 count= 1   1
  i= 5 count= 3   1 2 16
  i= 6 count= 3   1 4 16
  i= 7 count= 7   1 2 4 8 16 32 64
  i= 8 count= 1   1
  i= 9 count= 3   1 2 256
  i=10 count= 3   1 4 256
  i=11 count= 7   1 2 4 8 256 512 1024
  i=12 count= 3   1 16 256
  i=13 count= 7   1 2 16 32 256 512 4096
  i=14 count= 7   1 4 16 64 256 1024 4096
  i=15 count=15   1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384

Doing this increased benchmark speed by ~70%.

Most of the benefit comes from unrolling the loop of length shape[i].count
which the compiler could not predict. I left in some of the "silly" loops like
"for (int t = 0; t < 2; ++t)" since the compiler is smart enough to unroll
those itself.
*/
void monomial_to_novel_transpose(gf16_t a[GF16_ORDER])
{
    // i == 1
    for (int j = 0; j < GF16_ORDER; j += 4) {
        for (int t = 0; t < 2; ++t) {
            a[j + 2 + t] ^= a[j + t + 1];
        }
    }
    // i == 2
    for (int j = 0; j < GF16_ORDER; j += 8) {
        for (int t = 0; t < 4; ++t) {
            a[j + 4 + t] ^= a[j + t + 1];
        }
    }
    // i == 3
    for (int j = 0; j < GF16_ORDER; j += 16) {
        for (int t = 0; t < 8; ++t) {
            a[j + 8 + t] ^=
                a[j + t + 1] ^
                a[j + t + 2] ^
                a[j + t + 4];
        }
    }
    // i == 4
    for (int j = 0; j < GF16_ORDER; j += 32) {
        for (int t = 0; t < 16; ++t) {
            a[j + 16 + t] ^= a[j + t + 1];
        }
    }
    // i == 5
    for (int j = 0; j < GF16_ORDER; j += 64) {
        for (int t = 0; t < 32; ++t) {
            a[j + 32 + t] ^=
                a[j + t +  1] ^
                a[j + t +  2] ^
                a[j + t + 16];
        }
    }
    // i == 6
    for (int j = 0; j < GF16_ORDER; j += 128) {
        for (int t = 0; t < 64; ++t) {
            a[j + 64 + t] ^=
                a[j + t +  1] ^
                a[j + t +  4] ^
                a[j + t + 16];
        }
    }
    // i == 7
    for (int j = 0; j < GF16_ORDER; j += 256) {
        for (int t = 0; t < 128; ++t) {
            a[j + 128 + t] ^=
                a[j + t +  1] ^
                a[j + t +  2] ^
                a[j + t +  4] ^
                a[j + t +  8] ^
                a[j + t + 16] ^
                a[j + t + 32] ^
                a[j + t + 64];
        }
    }
    // i == 8
    for (int j = 0; j < GF16_ORDER; j += 512) {
        for (int t = 0; t < 256; ++t) {
            a[j + 256 + t] ^= a[j + t +  1];
        }
    }
    // i == 9
    for (int j = 0; j < GF16_ORDER; j += 1024) {
        for (int t = 0; t < 512; ++t) {
            a[j + 512 + t] ^=
                a[j + t +   1] ^
                a[j + t +   2] ^
                a[j + t + 256];
        }
    }
    // i == 10
    for (int j = 0; j < GF16_ORDER; j += 2048) {
        for (int t = 0; t < 1024; ++t) {
            a[j + 1024 + t] ^=
                a[j + t +   1] ^
                a[j + t +   4] ^
                a[j + t + 256];
        }
    }
    // i == 11
    for (int j = 0; j < GF16_ORDER; j += 4096) {
        for (int t = 0; t < 2048; ++t) {
            a[j + 2048 + t] ^=
                a[j + t +    1] ^
                a[j + t +    2] ^
                a[j + t +    4] ^
                a[j + t +    8] ^
                a[j + t +  256] ^
                a[j + t +  512] ^
                a[j + t + 1024];
        }
    }
    // i == 12
    for (int j = 0; j < GF16_ORDER; j += 8192) {
        for (int t = 0; t < 4096; ++t) {
            a[j + 4096 + t] ^=
                a[j + t +    1] ^
                a[j + t +   16] ^
                a[j + t +  256];
        }
    }
    // i == 13
    for (int j = 0; j < GF16_ORDER; j += 16384) {
        for (int t = 0; t < 8192; ++t) {
            a[j + 8192 + t] ^=
                a[j + t +    1] ^
                a[j + t +    2] ^
                a[j + t +   16] ^
                a[j + t +   32] ^
                a[j + t +  256] ^
                a[j + t +  512] ^
                a[j + t + 4096];
        }
    }
    // i == 14
    for (int j = 0; j < GF16_ORDER; j += 32768) {
        for (int t = 0; t < 16384; ++t) {
            a[j + 16384 + t] ^=
                a[j + t +    1] ^
                a[j + t +    4] ^
                a[j + t +   16] ^
                a[j + t +   64] ^
                a[j + t +  256] ^
                a[j + t + 1024] ^
                a[j + t + 4096];
        }
    }
    // i == 15
    for (int t = 0; t < 32768; ++t) {
        a[32768 + t] ^=
            a[t +     1] ^
            a[t +     2] ^
            a[t +     4] ^
            a[t +     8] ^
            a[t +    16] ^
            a[t +    32] ^
            a[t +    64] ^
            a[t +   128] ^
            a[t +   256] ^
            a[t +   512] ^
            a[t +  1024] ^
            a[t +  2048] ^
            a[t +  4096] ^
            a[t +  8192] ^
            a[t + 16384];
    }
}
#endif  // ITERATIVE_AFFT

#if !ITERATIVE_AFFT
/*
 * Exact transpose of additive_fft().
 *
 * Transpose of
 *   L = f0 + c*f1
 *   R = L + f1
 * is
 *   f0' = L' + R'
 *   f1' = c*(L'+R') + R'.
 */
static void additive_fft_transpose_rec(gf16_t *a, unsigned m, gf16_t alpha)
{
    if (m == 1) {
        const gf16_t c = alpha;
        for (uint32_t t = 0; t < 1; ++t) {
            const gf16_t L = a[t];
            const gf16_t R = a[1 + t];
            const gf16_t f0 = L ^ R;
            const gf16_t f1 = gf16_mul(c, f0) ^ R;
            a[t]     = f0;
            a[1 + t] = f1;
        }
        return;
    }

    const unsigned i = m - 1;
    const uint32_t h = 1u << i;

    /* Reverse the forward recursion first. */
    additive_fft_transpose_rec(a,     m - 1, alpha);
    additive_fft_transpose_rec(a + h, m - 1, alpha ^ beta[i]);

    const gf16_t c = subspace_poly_eval(i, alpha);

    for (uint32_t t = 0; t < h; ++t) {
        const gf16_t L = a[t];
        const gf16_t R = a[h + t];
        const gf16_t f0 = L ^ R;
        const gf16_t f1 = gf16_mul(c, f0) ^ R;
        a[t]     = f0;
        a[h + t] = f1;
    }
}

#define additive_fft_transpose(a) additive_fft_transpose_rec(a, GF16_BITS, 0)

#else  // ITERATIVE_AFFT

// Iterative implementation of additive_fft_transpose_rec() follows.

__attribute__((always_inline))
static inline void additive_fft_transpose_level_i(gf16_t a[GF16_ORDER], const int i) {
    const uint32_t n = GF16_ORDER;

    const int h          = 1 << i;
    const int block_size = 2 << i;

    // This can also be inlined in the loop below, but doing it separately can
    // be optimized better by the compiler.
    for (gf16_t *block = a; block < a + n; block += block_size) {
        for (uint32_t t = 0; t < h; ++t)
            block[t] ^= block[h + t];
    }

    const gf16_t *p = subspace_poly_for_afft + SUBSPACE_POLY_OFFSET(i);
    for (gf16_t *block = a; block < a + n; block += block_size) {
        const gf16_t log_c = *p++;
        if (log_c != 65535) {
            for (uint32_t t = 0; t < h; ++t)
                block[h + t] ^= gf16_mul_log(block[t], log_c);
        }
    }
}

void additive_fft_transpose(gf16_t a[GF16_ORDER])
{
    additive_fft_transpose_level_i(a,  0);
    additive_fft_transpose_level_i(a,  1);
    additive_fft_transpose_level_i(a,  2);
    additive_fft_transpose_level_i(a,  3);
    additive_fft_transpose_level_i(a,  4);
    additive_fft_transpose_level_i(a,  5);
    additive_fft_transpose_level_i(a,  6);
    additive_fft_transpose_level_i(a,  7);
    additive_fft_transpose_level_i(a,  8);
    additive_fft_transpose_level_i(a,  9);
    additive_fft_transpose_level_i(a, 10);
    additive_fft_transpose_level_i(a, 11);
    additive_fft_transpose_level_i(a, 12);
    additive_fft_transpose_level_i(a, 13);
    additive_fft_transpose_level_i(a, 14);
    additive_fft_transpose_level_i(a, 15);
}

#endif  // ITERATIVE_AFFT

/* ------------------------------------------------------------------------- */
/* Fast V and V^T in external polynomial-basis field-element ordering        */
/* ------------------------------------------------------------------------- */

/* Compute y[k] = sum_x a[x] x^k. */
void gf16_vandermonde_transpose_multiply(const gf16_t a[GF16_ORDER], gf16_t y[GF16_ORDER]) {
    /* P: gather external field-label order into Cantor-coordinate order. */
    for (uint32_t j = 0; j < GF16_ORDER; ++j)
        y[j] = a[cantor_permutation[j]];

    additive_fft_transpose(y);
    monomial_to_novel_transpose(y);
}

void gf16_vandermonde_transpose_init() {
    generate_cantor_basis();
#if SUBSPACE_POLY_LUT
    build_subspace_poly_eval_tables();
#endif
    build_subspace_poly_for_afft();
    build_cantor_permutation();

    /* Print out shape table contents for manual unrolling.
for (int i = 0; i < GF16_BITS; ++i) {
    printf("i=%d count=%d", i, shape[i].count);
    for (int j = 0; j < shape[i].count; ++j) printf(" %d", shape[i].exponent[j]);
    printf("\n");
}
    */
}
