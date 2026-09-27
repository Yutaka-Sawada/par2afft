#ifndef VANDERMONDE_TRANSPOSE_H_INCLUDED
#define VANDERMONDE_TRANSPOSE_H_INCLUDED

#include "gf16.h"

extern gf16_t cantor_permutation[GF16_ORDER];
extern gf16_t cantor_beta[GF16_BITS];

// Logarithms of the constants used in the AFFT, in order of usage.
//
// For a recusion level i and block index j, define:
//
//  alpha = the XOR sum of beta[k] of all bits k set in j
//  c     = subspace_poly_eval(i, alpha)
//  log_c = gf16_log(c) if c != 0 or 65535 otherwise
//
// Then subspace_poly_for_afft[SUBSPACE_POLY_OFFSET(i) + j] = log_c
//
// For details, see build_subspace_poly_for_afft() where this table is populated
// or additive_fft_transpose_level_i() where it is used.
extern gf16_t subspace_poly_for_afft[GF16_ORDER - 1];

#define SUBSPACE_POLY_OFFSET(level)  (GF16_ORDER - (1 << (GF16_BITS - level)))

void gf16_vt_mul_init();

// Compute y[k] = sum_x a[x] x^k.
void gf16_vt_mul(const gf16_t a[GF16_ORDER], gf16_t y[GF16_ORDER]);

// Exposed for testing
void monomial_to_novel_transpose(gf16_t a[GF16_ORDER]);
void additive_fft_transpose(gf16_t a[GF16_ORDER]);
gf16_t subspace_poly_eval(unsigned i, gf16_t x);

#endif  // ndef VANDERMONDE_TRANSPOSE_H_INCLUDED
