#ifndef GF16_VT_BATCH_MUL
#define GF16_VT_BATCH_MUL

#include "gf16.h"

#include <stddef.h>

#define GF16_VT_MIN_BATCH_SIZE 16

void gf16_vt_batch_mul_init();

// Computes the product of a transpose Vandermonde matrix of size (2^16)x(2^16)
// defined by vt_ij = j^i with a matrix `a` of size (2^16)x(width), and writes
// the result to the matrix `y` of size (2^16)x(width).
//
// Note that some of the pointers in `a` may be set to NULL to indicate all-zero
// rows, but all pointers in `y` must be present and refer to different arrays
// because they are used in intermediate calculations.
//
// Afterwards, y[i][j] = sum_k a[i][k] x j^k.
//
// `width` must be a multiple of GF16_VT_MIN_BATCH_SIZE
void gf16_vt_batch_mul(
#ifdef _MSC_VER
    gf16_t *const restrict y[GF16_ORDER],
    const gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict y[restrict static GF16_ORDER],
    const gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width);

// TODO: document this
void gf16_vt_batch_mul_inplace(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width);

// Exposed for testing:
void gf16_vt_batch_afft(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width);

void gf16_vt_batch_monomial_to_novel(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width);

#endif  // ndef GF16_VT_BATCH_MUL
