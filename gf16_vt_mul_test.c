#include "gf16.h"
#include "gf16_vt_mul.h"
#include "w1rand.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned count;
    uint32_t exponent[GF16_BITS];
} subspace_shape_t;

static subspace_shape_t shape[GF16_BITS];

/*
 * With this Cantor basis, the vanishing polynomial of
 * V_i = span(beta[0],...,beta[i-1]) is
 *
 *   s_i(X) = A^i(X),  A(X)=X^2+X.
 *
 * Since A = Frobenius + identity,
 *
 *   s_i(X) = sum_{j=0}^i C(i,j) X^(2^j)   over GF(2).
 *
 * C(i,j) is odd iff j is a bit-submask of i (Lucas's theorem).
 * Thus s_i is sparse.  We only need its lower exponents, excluding its
 * leading X^(2^i) term.
 */

static void build_subspace_shapes()
{
    for (unsigned i = 0; i < GF16_BITS; ++i) {
        shape[i].count = 0;
        for (unsigned j = 0; j < i; ++j) {
            if ((j & ~i) == 0) {
                shape[i].exponent[shape[i].count++] = 1u << j;
            }
        }
    }
}

/*
 * At a node of dimension m, h=2^(m-1), divide
 *
 *   f(X) = f0(X) + s_{m-1}(X) f1(X),
 *
 * with deg(f0), deg(f1) < h.
 *
 * Because s_{m-1} is monic and sparse, polynomial long division is only XOR.
 * The upper half of a[] becomes f1, the lower half becomes f0.  We then
 * recurse on both halves.
 */
static void monomial_to_novel_rec(gf16_t *a, unsigned m)
{
    if (m == 0)
        return;

    const unsigned i = m - 1;
    const uint32_t h = 1u << i;

    /* Long division by s_i, from highest term downward. */
    for (uint32_t t = h; t-- > 0;) {
        const gf16_t c = a[h + t];

        /* c may be zero, but avoiding the branch is often competitive. */
        if (c != 0) {
            for (unsigned r = 0; r < shape[i].count; ++r)
                a[t + shape[i].exponent[r]] ^= c;
        }
    }

    monomial_to_novel_rec(a,     m - 1);
    monomial_to_novel_rec(a + h, m - 1);
}

/*
 * Input is in novel basis for dimension m.
 * Output is evaluations at alpha + V_m in Cantor-coordinate order.
 *
 * Split f = f0 + s_i f1, i=m-1.  On alpha+V_i, s_i is the constant
 * c=s_i(alpha); on alpha+beta_i+V_i it is c+1 because s_i(beta_i)=1.
 *
 * One-multiply butterfly:
 *
 *   L = f0 + c*f1
 *   R = L  + f1
 */
static void additive_fft(gf16_t *a, unsigned m, gf16_t alpha)
{
    if (m == 0)
        return;

    const unsigned i = m - 1;
    const uint32_t h = 1u << i;
    const gf16_t c = subspace_poly_eval(i, alpha);

    for (uint32_t t = 0; t < h; ++t) {
        const gf16_t f0 = a[t];
        const gf16_t f1 = a[h + t];
        const gf16_t left = f0 ^ gf16_mul(c, f1);
        a[t]     = left;
        a[h + t] = left ^ f1;
    }

    additive_fft(a,     m - 1, alpha);
    additive_fft(a + h, m - 1, alpha ^ cantor_beta[i]);
}

/* ------------------------------------------------------------------------- */
/* Reference helpers                                                         */
/* ------------------------------------------------------------------------- */

/* Directly compute one output y[k] in O(GF16_ORDER log k) field operations. */
static gf16_t vandermonde_transpose_one_naive(const gf16_t a[GF16_ORDER], uint32_t k)
{
    gf16_t sum = 0;

    for (uint32_t x = 0; x < GF16_ORDER; ++x)
        sum ^= gf16_mul(a[x], gf16_pow((gf16_t)x, k));

    return sum;
}

/* Faster direct check of selected k: walk x^k by exponentiation per x. */
static void check_selected_outputs(const gf16_t a[GF16_ORDER],
                                   const gf16_t y[GF16_ORDER],
                                   const uint32_t *ks,
                                   size_t nks)
{
    for (size_t q = 0; q < nks; ++q) {
        const uint32_t k = ks[q];
        const gf16_t ref = vandermonde_transpose_one_naive(a, k);
        if (ref != y[k]) {
            fprintf(stderr,
                    "FAIL at k=%u: fast=%04X naive=%04X\n",
                    k, y[k], ref);
            exit(1);
        }
        printf("  k=%5u : %04X  [OK]\n", k, y[k]);
    }
}

/* Evaluate monomial polynomial c at every field element. */
static void vandermonde_forward(const gf16_t c[GF16_ORDER], gf16_t values[GF16_ORDER]) {
    memcpy(values, c, GF16_ORDER * sizeof(gf16_t));

    monomial_to_novel_rec(values, GF16_BITS);
    additive_fft(values, GF16_BITS, 0);

    /* values currently indexed by Cantor coordinates; scatter to x labels. */
    gf16_t *tmp = malloc(GF16_ORDER * sizeof(gf16_t));
    if (!tmp) {
        fprintf(stderr, "allocation failed\n");
        exit(2);
    }
    memcpy(tmp, values, GF16_ORDER * sizeof(gf16_t));

    for (uint32_t j = 0; j < GF16_ORDER; ++j)
        values[cantor_permutation[j]] = tmp[j];

    free(tmp);
}

/* Field-valued dot product. */
static gf16_t dot_product(const gf16_t *a, const gf16_t *b)
{
    gf16_t s = 0;
    for (uint32_t i = 0; i < GF16_ORDER; ++i)
        s ^= gf16_mul(a[i], b[i]);
    return s;
}

int gf16_vt_mul_test() {
    gf16_t *a    = malloc(GF16_ORDER * sizeof(gf16_t));
    gf16_t *y    = malloc(GF16_ORDER * sizeof(gf16_t));
    gf16_t *c    = malloc(GF16_ORDER * sizeof(gf16_t));
    gf16_t *Vc   = malloc(GF16_ORDER * sizeof(gf16_t));

    if (!a || !y || !c || !Vc) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    printf("Cantor basis for modulus 0x1100B:\n");
    for (unsigned i = 0; i < GF16_BITS; ++i)
        printf("  beta[%2u] = 0x%04X\n", i, cantor_beta[i]);

    /* Verify cantor_permutation[] is really a permutation of all 65536 field elements. */
    uint8_t *seen = calloc(GF16_ORDER, 1);
    assert(seen);
    for (uint32_t j = 0; j < GF16_ORDER; ++j) {
        assert(!seen[cantor_permutation[j]]);
        seen[cantor_permutation[j]] = 1;
    }
    free(seen);

    printf("\nTest 1: a[x] = x\n");

    for (uint32_t x = 0; x < GF16_ORDER; ++x)
        a[x] = (gf16_t)x;

    gf16_vt_mul(y, a);

    /*
     * y[k] = sum_x x^(k+1).
     * In GF(q), q=65536, this is nonzero only when (q-1)|(k+1).
     * For k=0..65535, the only such k is 65534, and the value is 1.
     */
    for (uint32_t k = 0; k < GF16_ORDER; ++k) {
        const gf16_t expected = (k == GF16_ORDER - 2) ? 1 : 0;
        if (y[k] != expected) {
            fprintf(stderr,
                    "FAIL structured test at k=%u: got=%04X expected=%04X\n",
                    k, y[k], expected);
            return 1;
        }
    }
    printf("  PASS: only y[65534] is 1; all other outputs are 0.\n");

    /*
     * Test 2: arbitrary deterministic vector; compare selected outputs with
     * the literal definition sum_x a[x] x^k.
     */
    printf("\nTest 2: pseudorandom vector, selected direct checks\n");
    uint64_t rand_state = 123456789;
    w1rand_fill(a, GF16_ORDER * sizeof(gf16_t), &rand_state);

    gf16_vt_mul(y, a);

    static const uint32_t ks[] = {
        0, 1, 2, 3, 7, 15, 16, 31,
        255, 256, 257, 1023, 4096, 32767,
        65534, 65535
    };
    check_selected_outputs(a, y, ks, sizeof(ks) / sizeof(ks[0]));
    printf("  PASS: selected outputs match the direct definition.\n");

    /*
     * Test 3: exact transpose identity
     *
     *   <V^T a, c> = <a, V c>
     *
     * for an independent deterministic vector c.
     */
    printf("\nTest 3: transpose inner-product identity\n");
    w1rand_fill(c, GF16_ORDER * sizeof(gf16_t), &rand_state);

    vandermonde_forward(c, Vc);

    const gf16_t lhs = dot_product(y, c);
    const gf16_t rhs = dot_product(a, Vc);

    printf("  <V^T a, c> = %04X\n", lhs);
    printf("  <a, V c>   = %04X\n", rhs);

    if (lhs != rhs) {
        fprintf(stderr, "FAIL: transpose identity does not hold\n");
        return 1;
    }
    printf("  PASS: transpose identity holds.\n");

    free(a);
    free(y);
    free(c);
    free(Vc);

    printf("\nALL TESTS PASSED\n");
    return 0;
}

int main()
{
    gf16_init();
    gf16_vt_mul_init();
    build_subspace_shapes();

    /* Print out shape table contents for manual unrolling.
    for (int i = 0; i < GF16_BITS; ++i) {
        printf("i=%d count=%d", i, shape[i].count);
        for (int j = 0; j < shape[i].count; ++j) printf(" %d", shape[i].exponent[j]);
        printf("\n");
    }
    */

    return gf16_vt_mul_test();
}
