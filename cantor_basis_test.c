#include "gf16.h"
#include "gf16_vt_mul.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>

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
static void generate_cantor_basis(gf16_t beta[GF16_BITS])
{
    gf16_t top = 0;

    for (uint32_t x = 1; x < GF16_ORDER; ++x) {
        if (gf16_trace((gf16_t)x) == 1) {
            top = (gf16_t)x;
            break;
        }
    }

    assert(top != 0);

    beta[GF16_BITS - 1] = top;
    for (int i = GF16_BITS - 1; i > 0; --i)
        beta[i - 1] = gf16_sqr(beta[i]) ^ beta[i];

    assert(beta[0] == 1);

    for (int i = 1; i < GF16_BITS; ++i)
        assert((gf16_sqr(beta[i]) ^ beta[i]) == beta[i - 1]);
}

int main() {
    gf16_init();

    gf16_t beta[GF16_BITS];
    generate_cantor_basis(beta);

    if (memcmp(beta, cantor_beta, sizeof(gf16_t) * GF16_BITS) == 0) {
        printf("PASS\n");
    } else {
        for (int i = 0; i < GF16_BITS; ++i) {
            fprintf(stderr, "beta[%2d] = %04x %s %04x\n", i, beta[i],
                 beta[i] == cantor_beta[i] ? "==" : "!=",
                cantor_beta[i]);
        }
        fprintf(stderr, "FAIL\n");
        return 1;
    }
}
