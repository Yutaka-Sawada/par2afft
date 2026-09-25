#ifndef VANDERMONDE_TRANSPOSE_H_INCLUDED
#define VANDERMONDE_TRANSPOSE_H_INCLUDED

#include "gf16.h"

#ifndef GF16_VANDERMONDE_TESTS_INCLUDED
#define GF16_VANDERMONDE_TESTS_INCLUDED 0
#endif

extern gf16_t cantor_permutation[GF16_ORDER];

void gf16_vandermonde_transpose_init();

/* Compute y[k] = sum_x a[x] x^k. */
void gf16_vandermonde_transpose_multiply(const gf16_t a[1 << 16], gf16_t y[1 << 16]);

#if GF16_VANDERMONDE_TESTS_INCLUDED
int gf16_vandermonde_transpose_test();
#endif

#endif  // ndef VANDERMONDE_TRANSPOSE_H_INCLUDED
