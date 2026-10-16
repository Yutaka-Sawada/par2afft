#ifndef GF16_VT_MUL_H_INCLUDED
#define GF16_VT_MUL_H_INCLUDED

#include "gf16.h"

extern const gf16_t cantor_beta[GF16_BITS];
extern gf16_t cantor_permutation[GF16_ORDER];

// Logarithms of the constants used in the AFFT, in order of usage.
//
// Note that all constants are nonzero, except the first one.
// subspace_poly_for_afft[0] == 65535 to indicate the logarithm is invalid.
//
// For details, see build_subspace_poly_for_afft() where this table is populated
// or additive_fft_transpose_level_i() where it is used.
extern gf16_t subspace_poly_for_afft[GF16_ORDER / 2];

void gf16_vt_mul_init();

// Compute y[k] = sum_x a[x] x^k.
void gf16_vt_mul(gf16_t y[GF16_ORDER], const gf16_t a[GF16_ORDER]);

// Exposed for testing
void monomial_to_novel_transpose(gf16_t a[GF16_ORDER]);
void additive_fft_transpose(gf16_t a[GF16_ORDER]);
gf16_t subspace_poly_eval(unsigned i, gf16_t x);

#endif  // ndef GF16_VT_MUL_H_INCLUDED
