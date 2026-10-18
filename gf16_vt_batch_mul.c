#include "gf16_vt_batch_mul.h"

#include "gf16_avx2_mul.h"
#include "gf16_vt_mul.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>

#ifndef AVX2_MUL_AND_XOR
#define AVX2_MUL_AND_XOR 1
#endif

// Nonbatching versions that could be used to handle the tail.
#if 0
static void xorv_1(gf16_t *restrict dest, const gf16_t *restrict src, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        dest[i] ^= src[i];
    }
}

static void xorv_3(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        size_t width) {
    for (size_t i = 0; i < width; ++i) {
        dest[i] ^= src1[i] ^ src2[i] ^ src3[i];
    }
}

static void xorv_7(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        const gf16_t *restrict src4,
        const gf16_t *restrict src5,
        const gf16_t *restrict src6,
        const gf16_t *restrict src7,
        size_t width) {
    for (size_t i = 0; i < width; ++i) {
        dest[i] ^= src1[i] ^ src2[i] ^ src3[i] ^ src4[i] ^ src5[i] ^ src6[i] ^ src7[i];
    }
}

static void xorv_15(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        const gf16_t *restrict src4,
        const gf16_t *restrict src5,
        const gf16_t *restrict src6,
        const gf16_t *restrict src7,
        const gf16_t *restrict src8,
        const gf16_t *restrict src9,
        const gf16_t *restrict src10,
        const gf16_t *restrict src11,
        const gf16_t *restrict src12,
        const gf16_t *restrict src13,
        const gf16_t *restrict src14,
        const gf16_t *restrict src15,
        size_t width) {
    for (size_t i = 0; i < width; ++i) {
        dest[i] ^= src1[i] ^ src2[i] ^ src3[i] ^ src4[i] ^ src5[i] ^ src6[i] ^ src7[i]
            ^ src8[i] ^ src9[i] ^ src10[i] ^ src11[i] ^ src12[i] ^ src13[i] ^ src14[i] ^ src15[i];
    }
}
#endif

static void xor16_1(gf16_t *restrict dest, const gf16_t *restrict src) {
    for (size_t i = 0; i < 16; ++i) dest[i] ^= src[i];
}

static void xorv_1(gf16_t *restrict dest, const gf16_t *restrict src, size_t width) {
    for (size_t i = 0; i + 16 <= width; i += 16) {
        xor16_1(&dest[i], &src[i]);
    }
}

static void xor16_3(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3) {
    for (size_t i = 0; i < 16; ++i) dest[i] ^= src1[i] ^ src2[i] ^ src3[i];
}

static void xorv_3(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        size_t width) {
    for (size_t i = 0; i + 16 <= width; i += 16) {
        xor16_3(&dest[i], &src1[i], &src2[i], &src3[i]);
    }
}

static void xor16_7(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        const gf16_t *restrict src4,
        const gf16_t *restrict src5,
        const gf16_t *restrict src6,
        const gf16_t *restrict src7) {
    for (size_t i = 0; i < 16; ++i)
        dest[i] ^= src1[i] ^ src2[i] ^ src3[i] ^ src4[i] ^ src5[i] ^ src6[i] ^ src7[i];
}

static void xorv_7(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        const gf16_t *restrict src4,
        const gf16_t *restrict src5,
        const gf16_t *restrict src6,
        const gf16_t *restrict src7,
        size_t width) {
    for (size_t i = 0; i + 16 <= width; i += 16) {
        xor16_7(&dest[i], &src1[i], &src2[i], &src3[i], &src4[i], &src5[i], &src6[i], &src7[i]);
    }
}

static void xor16_15(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        const gf16_t *restrict src4,
        const gf16_t *restrict src5,
        const gf16_t *restrict src6,
        const gf16_t *restrict src7,
        const gf16_t *restrict src8,
        const gf16_t *restrict src9,
        const gf16_t *restrict src10,
        const gf16_t *restrict src11,
        const gf16_t *restrict src12,
        const gf16_t *restrict src13,
        const gf16_t *restrict src14,
        const gf16_t *restrict src15) {
    for (size_t i = 0; i < 16; ++i) {
        dest[i] ^= src1[i] ^ src2[i] ^ src3[i] ^ src4[i] ^ src5[i] ^ src6[i] ^ src7[i] ^
                src8[i] ^ src9[i] ^ src10[i] ^ src11[i] ^ src12[i] ^ src13[i] ^ src14[i] ^ src15[i];
    }
}

static void xorv_15(gf16_t *restrict dest,
        const gf16_t *restrict src1,
        const gf16_t *restrict src2,
        const gf16_t *restrict src3,
        const gf16_t *restrict src4,
        const gf16_t *restrict src5,
        const gf16_t *restrict src6,
        const gf16_t *restrict src7,
        const gf16_t *restrict src8,
        const gf16_t *restrict src9,
        const gf16_t *restrict src10,
        const gf16_t *restrict src11,
        const gf16_t *restrict src12,
        const gf16_t *restrict src13,
        const gf16_t *restrict src14,
        const gf16_t *restrict src15,
        size_t width) {
    for (size_t i = 0; i + 16 <= width; i += 16) {
        xor16_15(&dest[i], &src1[i], &src2[i], &src3[i], &src4[i], &src5[i], &src6[i], &src7[i],
            &src8[i], &src9[i], &src10[i], &src11[i], &src12[i], &src13[i], &src14[i], &src15[i]);
    }
}

void gf16_vt_batch_monomial_to_novel(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width)
{
    assert(width % 16 == 0);

    // i == 1
    for (int j = 0; j < GF16_ORDER; j += 4) {
        for (int t = 0; t < 2; ++t) {
            xorv_1(a[j + 2 + t], a[j + t + 1], width);
        }
    }
    // i == 2
    for (int j = 0; j < GF16_ORDER; j += 8) {
        for (int t = 0; t < 4; ++t) {
            xorv_1(a[j + 4 + t], a[j + t + 1], width);
        }
    }
    // i == 3
    for (int j = 0; j < GF16_ORDER; j += 16) {
        for (int t = 0; t < 8; ++t) {
            xorv_3(a[j + 8 + t],
                a[j + t + 1],
                a[j + t + 2],
                a[j + t + 4],
                width);
        }
    }
    // i == 4
    for (int j = 0; j < GF16_ORDER; j += 32) {
        for (int t = 0; t < 16; ++t) {
            xorv_1(a[j + 16 + t], a[j + t + 1], width);
        }
    }
    // i == 5
    for (int j = 0; j < GF16_ORDER; j += 64) {
        for (int t = 0; t < 32; ++t) {
            xorv_3(a[j + 32 + t],
                a[j + t +  1],
                a[j + t +  2],
                a[j + t + 16],
                width);
        }
    }
    // i == 6
    for (int j = 0; j < GF16_ORDER; j += 128) {
        for (int t = 0; t < 64; ++t) {
            xorv_3(a[j + 64 + t],
                a[j + t +  1],
                a[j + t +  4],
                a[j + t + 16],
                width);
        }
    }
    // i == 7
    for (int j = 0; j < GF16_ORDER; j += 256) {
        for (int t = 0; t < 128; ++t) {
            xorv_7(a[j + 128 + t],
                a[j + t +  1],
                a[j + t +  2],
                a[j + t +  4],
                a[j + t +  8],
                a[j + t + 16],
                a[j + t + 32],
                a[j + t + 64],
                width);
        }
    }
    // i == 8
    for (int j = 0; j < GF16_ORDER; j += 512) {
        for (int t = 0; t < 256; ++t) {
            xorv_1(a[j + 256 + t], a[j + t +  1], width);
        }
    }
    // i == 9
    for (int j = 0; j < GF16_ORDER; j += 1024) {
        for (int t = 0; t < 512; ++t) {
            xorv_3(a[j + 512 + t],
                a[j + t +   1],
                a[j + t +   2],
                a[j + t + 256],
                width);
        }
    }
    // i == 10
    for (int j = 0; j < GF16_ORDER; j += 2048) {
        for (int t = 0; t < 1024; ++t) {
            xorv_3(a[j + 1024 + t],
                a[j + t +   1],
                a[j + t +   4],
                a[j + t + 256],
                width);
        }
    }
    // i == 11
    for (int j = 0; j < GF16_ORDER; j += 4096) {
        for (int t = 0; t < 2048; ++t) {
            xorv_7(a[j + 2048 + t],
                a[j + t +    1],
                a[j + t +    2],
                a[j + t +    4],
                a[j + t +    8],
                a[j + t +  256],
                a[j + t +  512],
                a[j + t + 1024],
                width);
        }
    }
    // i == 12
    for (int j = 0; j < GF16_ORDER; j += 8192) {
        for (int t = 0; t < 4096; ++t) {
            xorv_3(a[j + 4096 + t],
                a[j + t +    1],
                a[j + t +   16],
                a[j + t +  256],
                width);
        }
    }
    // i == 13
    for (int j = 0; j < GF16_ORDER; j += 16384) {
        for (int t = 0; t < 8192; ++t) {
            xorv_7(a[j + 8192 + t],
                a[j + t +    1],
                a[j + t +    2],
                a[j + t +   16],
                a[j + t +   32],
                a[j + t +  256],
                a[j + t +  512],
                a[j + t + 4096],
                width);
        }
    }
    // i == 14
    for (int j = 0; j < GF16_ORDER; j += 32768) {
        for (int t = 0; t < 16384; ++t) {
            xorv_7(a[j + 16384 + t],
                a[j + t +    1],
                a[j + t +    4],
                a[j + t +   16],
                a[j + t +   64],
                a[j + t +  256],
                a[j + t + 1024],
                a[j + t + 4096],
                width);
        }
    }
    // i == 15
    for (int t = 0; t < 32768; ++t) {
        xorv_15(a[32768 + t],
            a[t +     1],
            a[t +     2],
            a[t +     4],
            a[t +     8],
            a[t +    16],
            a[t +    32],
            a[t +    64],
            a[t +   128],
            a[t +   256],
            a[t +   512],
            a[t +  1024],
            a[t +  2048],
            a[t +  4096],
            a[t +  8192],
            a[t + 16384],
            width);
    }
}

#if AVX2_MUL_AND_XOR

static gf16_nibtab_avx2_t nibtab_for_afft[GF16_ORDER / 2];

#endif  // AVX2_MUL_AND_XOR

#ifdef _MSC_VER // MSVC uses __forceinline instead of always_inline.
static __forceinline void gf16_vt_batch_afft_level_i(
#else
__attribute__((always_inline))
static inline void gf16_vt_batch_afft_level_i(
#endif
        const int i, gf16_t *const restrict a[GF16_ORDER], size_t width) {

    const int h           = 1 << i;
    const int block_size  = 1 << (i + 1);
    const int block_count = 1 << (GF16_BITS - 1 - i);

    // This can also be inlined in the loop below, but doing it separately can
    // be optimized better by the compiler.
    for (int i = 0; i < block_count; ++i) {
        gf16_t *const restrict *block = a + i * block_size;
        for (int t = 0; t < h; ++t)
            xorv_1(block[t], block[h + t], width);
    }

#if !AVX2_MUL_AND_XOR
    // Multiply using GF16 lookup tables
    for (int i = 1; i < block_count; ++i) {
        const gf16_t log_c = subspace_poly_for_afft[i];
        gf16_t *const restrict *block = a + i * block_size;
        for (int t = 0; t < h; ++t) {
            const gf16_t *src = block[t];
            gf16_t *dest = block[h + t];
            for (int i = 0; i < width; ++i) {
                dest[i] ^= gf16_mul_log(src[i], log_c);
            }
        }
    }
#else  // AVX2_MUL_AND_XOR
    // Multiply using AVX2 nibble tables
    for (int i = 1; i < block_count; ++i) {
        const gf16_nibtab_avx2_t *tab = &nibtab_for_afft[i];
        gf16_t *const restrict *block = a + i * block_size;
        for (int t = 0; t < h; ++t) {
            gf16_avx2_nibmul_and_xor(block[h + t], block[t], tab, width);
            //gf16_avx2_nibmul32_and_xor(block[h + t], block[t], tab, width); // process 32 words per loop
        }
    }
#endif // AVX2_MUL_AND_XOR
}

void gf16_vt_batch_afft(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width)
{
    gf16_vt_batch_afft_level_i(  0, a, width);
    gf16_vt_batch_afft_level_i(  1, a, width);
    gf16_vt_batch_afft_level_i(  2, a, width);
    gf16_vt_batch_afft_level_i(  3, a, width);
    gf16_vt_batch_afft_level_i(  4, a, width);
    gf16_vt_batch_afft_level_i(  5, a, width);
    gf16_vt_batch_afft_level_i(  6, a, width);
    gf16_vt_batch_afft_level_i(  7, a, width);
    gf16_vt_batch_afft_level_i(  8, a, width);
    gf16_vt_batch_afft_level_i(  9, a, width);
    gf16_vt_batch_afft_level_i( 10, a, width);
    gf16_vt_batch_afft_level_i( 11, a, width);
    gf16_vt_batch_afft_level_i( 12, a, width);
    gf16_vt_batch_afft_level_i( 13, a, width);
    gf16_vt_batch_afft_level_i( 14, a, width);
    gf16_vt_batch_afft_level_i( 15, a, width);
}

void gf16_vt_batch_mul_init() {
    static char initialized = 0;
    if (initialized) return;

    gf16_vt_mul_init();

#if AVX2_MUL_AND_XOR
    nibtab_for_afft[0] = gf16_avx2_gen_nibtab(0);
    for (int i = 1; i < GF16_ORDER / 2; ++i) {
        gf16_t c = gf16_exp(subspace_poly_for_afft[i]);
        nibtab_for_afft[i] = gf16_avx2_gen_nibtab(c);
    }
#endif

    initialized = 1;
}

// TODO make this an inplace function? just need to permute pointers OR assign in node order?
void gf16_vt_batch_mul(
#ifdef _MSC_VER
    gf16_t *const restrict y[GF16_ORDER],
    const gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict y[restrict static GF16_ORDER],
    const gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width)
{
    for (uint32_t j = 0; j < GF16_ORDER; ++j) {
        const gf16_t *src = a[cantor_permutation[j]];
        if (src == NULL) {
            memset(y[j], 0, sizeof(gf16_t) * width);
        } else {
            memcpy(y[j], src, sizeof(gf16_t) * width);
        }
    }
    gf16_vt_batch_mul_inplace(y, width);
}

void gf16_vt_batch_mul_inplace(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width)
{
    gf16_vt_batch_afft(a, width);
    gf16_vt_batch_monomial_to_novel(a, width);
}


// functions to skip empty data
//#define TEST_SKIP
#ifdef TEST_SKIP
int empty_count;
int skip_count = 0;
int task_count = 0;
int batch_count = 0;
#endif

void gf16_vt_batch_monomial_to_novel_skip(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width, int output_limit)
{
    assert(width % 16 == 0);

    // i == 1
    for (int j = 0; j < output_limit - 2; j += 4) {
        for (int t = 0; t < 2; ++t) {
            xorv_1(a[j + 2 + t], a[j + t + 1], width);
        }
    }
    // i == 2
    for (int j = 0; j < output_limit - 4; j += 8) {
        for (int t = 0; t < 4; ++t) {
            xorv_1(a[j + 4 + t], a[j + t + 1], width);
        }
    }
    // i == 3
    for (int j = 0; j < output_limit - 8; j += 16) {
        for (int t = 0; t < 8; ++t) {
            xorv_3(a[j + 8 + t],
                a[j + t + 1],
                a[j + t + 2],
                a[j + t + 4],
                width);
        }
    }
    // i == 4
    for (int j = 0; j < output_limit - 16; j += 32) {
        for (int t = 0; t < 16; ++t) {
            xorv_1(a[j + 16 + t], a[j + t + 1], width);
        }
    }
    // i == 5
    for (int j = 0; j < output_limit - 32; j += 64) {
        for (int t = 0; t < 32; ++t) {
            xorv_3(a[j + 32 + t],
                a[j + t +  1],
                a[j + t +  2],
                a[j + t + 16],
                width);
        }
    }
    // i == 6
    for (int j = 0; j < output_limit - 64; j += 128) {
        for (int t = 0; t < 64; ++t) {
            xorv_3(a[j + 64 + t],
                a[j + t +  1],
                a[j + t +  4],
                a[j + t + 16],
                width);
        }
    }
    // i == 7
    for (int j = 0; j < output_limit - 128; j += 256) {
        for (int t = 0; t < 128; ++t) {
            xorv_7(a[j + 128 + t],
                a[j + t +  1],
                a[j + t +  2],
                a[j + t +  4],
                a[j + t +  8],
                a[j + t + 16],
                a[j + t + 32],
                a[j + t + 64],
                width);
        }
    }
    // i == 8
    for (int j = 0; j < output_limit - 256; j += 512) {
        for (int t = 0; t < 256; ++t) {
            xorv_1(a[j + 256 + t], a[j + t +  1], width);
        }
    }
    // i == 9
    for (int j = 0; j < output_limit - 512; j += 1024) {
        for (int t = 0; t < 512; ++t) {
            xorv_3(a[j + 512 + t],
                a[j + t +   1],
                a[j + t +   2],
                a[j + t + 256],
                width);
        }
    }
    // i == 10
    for (int j = 0; j < output_limit - 1024; j += 2048) {
        for (int t = 0; t < 1024; ++t) {
            xorv_3(a[j + 1024 + t],
                a[j + t +   1],
                a[j + t +   4],
                a[j + t + 256],
                width);
        }
    }
    // i == 11
    for (int j = 0; j < output_limit - 2048; j += 4096) {
        for (int t = 0; t < 2048; ++t) {
            xorv_7(a[j + 2048 + t],
                a[j + t +    1],
                a[j + t +    2],
                a[j + t +    4],
                a[j + t +    8],
                a[j + t +  256],
                a[j + t +  512],
                a[j + t + 1024],
                width);
        }
    }
    // i == 12
    for (int j = 0; j < output_limit - 4096; j += 8192) {
        for (int t = 0; t < 4096; ++t) {
            xorv_3(a[j + 4096 + t],
                a[j + t +    1],
                a[j + t +   16],
                a[j + t +  256],
                width);
        }
    }
    // i == 13
    // When "output_limit <= 8192", it can skip whole this step.
    // When "8192 < output_limit <= 24576", it may skip 3 of 4 loops.
    // When "24576 < output_limit <= 40960", it may skip 2 of 4 loops.
    // When "40960 < output_limit <= 57344", it may skip 1 of 4 loops.
    // When "57344 < output_limit <= 65535", it cannot skip.
    for (int j = 0; j < output_limit - 8192; j += 16384) {
        for (int t = 0; t < 8192; ++t) {
            // This checking in each loop may be slow. It depends on XOR cost.
            //if (j + 8192 + t >= output_limit)
            //    break; // Exit loop not to beyond limit.
            xorv_7(a[j + 8192 + t],
                a[j + t +    1],
                a[j + t +    2],
                a[j + t +   16],
                a[j + t +   32],
                a[j + t +  256],
                a[j + t +  512],
                a[j + t + 4096],
                width);
        }
    }
    // i == 14
    // When "output_limit <= 16384", it can skip whole this step.
    // When "j = 0", dest = a[16384 ~ 32767], src = a[1 ~ 20479].
    // When "j = 32768", dest = a[49152 ~ 65535], src = a[32769 ~ 53247].
    // max dest = a[limit - 1]
    // max src = a[limit - 1 - 16384 + 4096] = a[limit - 1 - 12288]
    for (int j = 0; j < output_limit - 16384; j += 32768) {
        for (int t = 0; t < 16384; ++t) {
            // This checking in each loop may be slow. It depends on XOR cost.
            //if (j + 16384 + t >= output_limit)
            //    break; // Exit loop not to beyond limit.
            xorv_7(a[j + 16384 + t],
                a[j + t +    1],
                a[j + t +    4],
                a[j + t +   16],
                a[j + t +   64],
                a[j + t +  256],
                a[j + t + 1024],
                a[j + t + 4096],
                width);
        }
    }
    // i == 15
    // When "output_limit <= 32768", it can skip whole this step.
    // dest = a[32768 ~ 65535], src = a[1 ~ 49151].
    // max dest = a[limit - 1]
    // max src = a[limit - 1 - 32768 + 16384] = a[limit - 1 - 16384]
    if (output_limit > 32768){
        for (int t = 0; t < 32768; ++t) {
            // This checking in each loop may be slow. It depends on XOR cost.
            //if (32768 + t >= output_limit)
            //    break; // Exit loop not to beyond limit.
            xorv_15(a[32768 + t],
                a[t +     1],
                a[t +     2],
                a[t +     4],
                a[t +     8],
                a[t +    16],
                a[t +    32],
                a[t +    64],
                a[t +   128],
                a[t +   256],
                a[t +   512],
                a[t +  1024],
                a[t +  2048],
                a[t +  4096],
                a[t +  8192],
                a[t + 16384],
                width);
        }
    }
}

#ifdef _MSC_VER // MSVC uses __forceinline instead of always_inline.
static __forceinline void gf16_vt_batch_afft_level_i_skip(
#else
__attribute__((always_inline))
static inline void gf16_vt_batch_afft_level_i_skip(
#endif
        const int i, gf16_t *const restrict a[GF16_ORDER], size_t width, char empty[GF16_ORDER]) {

    const int h           = 1 << i;
    const int block_size  = 1 << (i + 1);
    const int block_count = 1 << (GF16_BITS - 1 - i);

#ifdef TEST_SKIP
    if (batch_count < 16){
        empty_count = 0;
        for (int t = 0; t < GF16_ORDER; t++){
            if (empty[t])
                empty_count++;
        }
        printf("before gf16_vt_batch_afft_level%d, empty = %d\n", i, empty_count);
    }
#endif

    // This can also be inlined in the loop below, but doing it separately can
    // be optimized better by the compiler.
    for (int i = 0; i < block_count; ++i) {
        gf16_t *const restrict *block = a + i * block_size;
        char *empty_flag = empty + i * block_size;
        for (int t = 0; t < h; ++t){
            if (empty_flag[h + t] == 0){ // src isn't empty.
                if (empty_flag[t] == 0){ // Only when both src & dest aren't empty, XOR them.
                    xorv_1(block[t], block[h + t], width);
                } else { // If dest is empty, just copy src.
                    memcpy(block[t], block[h + t], width * sizeof(gf16_t));
                    empty_flag[t] = 0; // dest becomes non-empty.
#ifdef TEST_SKIP
                    empty_count--;
#endif
                }
#ifdef TEST_SKIP
            } else {
                skip_count++;
#endif
            }
#ifdef TEST_SKIP
            task_count++;
#endif
        }
    }

#ifdef TEST_SKIP
    if (batch_count < 16)
        printf("after 1st step, empty = %d, skip = %d/%d\n", empty_count, skip_count, task_count);
#endif

#if !AVX2_MUL_AND_XOR
    // Multiply using GF16 lookup tables
    for (int i = 1; i < block_count; ++i) {
        const gf16_t log_c = subspace_poly_for_afft[i];
        gf16_t *const restrict *block = a + i * block_size;
        char *empty_flag = empty + i * block_size;
        for (int t = 0; t < h; ++t) {
            if (empty_flag[t] == 0){ // src isn't empty.
                const gf16_t *src = block[t];
                gf16_t *dest = block[h + t];
                if (empty_flag[h + t] == 0){ // Only when both src & dest are not empty, XOR result to dest.
                    for (int i = 0; i < width; ++i) {
                        dest[i] ^= gf16_mul_log(src[i], log_c);
                    }
                } else { // If dest is empty, just put result on dest.
                    for (int i = 0; i < width; ++i) {
                        dest[i] = gf16_mul_log(src[i], log_c);
                    }
                    empty_flag[h + t] = 0; // dest becomes non-empty.
#ifdef TEST_SKIP
                    empty_count--;
#endif
                }
#ifdef TEST_SKIP
            } else {
                skip_count++;
#endif
            }
#ifdef TEST_SKIP
            task_count++;
#endif
        }
    }
#else  // AVX2_MUL_AND_XOR
    // Multiply using AVX2 nibble tables
    for (int i = 1; i < block_count; ++i) {
        const gf16_nibtab_avx2_t *tab = &nibtab_for_afft[i];
        gf16_t *const restrict *block = a + i * block_size;
        char *empty_flag = empty + i * block_size;
        for (int t = 0; t < h; ++t) {
            if (empty_flag[t] == 0){ // src isn't empty.
                if (empty_flag[h + t] == 0){ // Only when both src & dest are not empty, XOR result to dest.
                    gf16_avx2_nibmul_and_xor(block[h + t], block[t], tab, width);
                    //gf16_avx2_nibmul32_and_xor(block[h + t], block[t], tab, width); // process 32 words per loop
                } else { // If dest is empty, just put result on dest.
                    gf16_avx2_nibmul(block[h + t], block[t], tab, width);
                    empty_flag[h + t] = 0; // dest becomes non-empty.
#ifdef TEST_SKIP
                    empty_count--;
#endif
                }
#ifdef TEST_SKIP
            } else {
                skip_count++;
#endif
            }
#ifdef TEST_SKIP
            task_count++;
#endif
        }
    }
#endif // AVX2_MUL_AND_XOR

#ifdef TEST_SKIP
    if (batch_count < 16){
        printf("after 2nd step, empty = %d, skip = %d/%d\n", empty_count, skip_count, task_count);
        batch_count++;
    }
#endif
}

void gf16_vt_batch_afft_skip(
#ifdef _MSC_VER
    gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width, char empty[GF16_ORDER], int valid_count)
{
    gf16_vt_batch_afft_level_i_skip(  0, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  1, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  2, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  3, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  4, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  5, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  6, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  7, a, width, empty);
    gf16_vt_batch_afft_level_i_skip(  8, a, width, empty);
#ifdef TEST_SKIP
    gf16_vt_batch_afft_level_i_skip(  9, a, width, empty);
    gf16_vt_batch_afft_level_i_skip( 10, a, width, empty);
    gf16_vt_batch_afft_level_i_skip( 11, a, width, empty);
    gf16_vt_batch_afft_level_i_skip( 12, a, width, empty);
    gf16_vt_batch_afft_level_i_skip( 13, a, width, empty);
    gf16_vt_batch_afft_level_i_skip( 14, a, width, empty);
    gf16_vt_batch_afft_level_i_skip( 15, a, width, empty);
#else
    if (valid_count >= 676){ // When data isn't empty, no need to check.
        gf16_vt_batch_afft_level_i(  9, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip(  9, a, width, empty);
    }
    if (valid_count >= 549){
        gf16_vt_batch_afft_level_i( 10, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip( 10, a, width, empty);
    }
    if (valid_count >= 97){
        gf16_vt_batch_afft_level_i( 11, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip( 11, a, width, empty);
    }
    if (valid_count >= 68){
        gf16_vt_batch_afft_level_i( 12, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip( 12, a, width, empty);
    }
    if (valid_count >= 29){
        gf16_vt_batch_afft_level_i( 13, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip( 13, a, width, empty);
    }
    if (valid_count >= 15){
        gf16_vt_batch_afft_level_i( 14, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip( 14, a, width, empty);
    }
    if (valid_count >= 7){
        gf16_vt_batch_afft_level_i( 15, a, width);
    } else {
        gf16_vt_batch_afft_level_i_skip( 15, a, width, empty);
    }
#endif
}

void gf16_vt_batch_mul_skip(
#ifdef _MSC_VER
    gf16_t *const restrict y[GF16_ORDER],
    const gf16_t *const restrict a[GF16_ORDER],
#else
    gf16_t *const restrict y[restrict static GF16_ORDER],
    const gf16_t *const restrict a[restrict static GF16_ORDER],
#endif
    size_t width,
    int input_num,    // number of source blocks
    int output_limit) // require output below this limit
{
    char empty[GF16_ORDER]; // empty flag of y[]

    for (uint32_t j = 0; j < GF16_ORDER; ++j) {
        const gf16_t *src = a[cantor_permutation[j]];
        if (src == NULL) {
            //memset(y[j], 0, sizeof(gf16_t) * width);
            empty[j] = 1; // Set empty flag, instead of zero fill data.
        } else {
            memcpy(y[j], src, sizeof(gf16_t) * width);
            empty[j] = 0;
        }
    }
    gf16_vt_batch_afft_skip(y, width, empty, input_num);
    gf16_vt_batch_monomial_to_novel_skip(y, width, output_limit);
}
