#include "gf16.h"
#include "gf16_vt_batch_mul.h"
#include "gf16_vt_mul.h"
#include "w1rand.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W (16*3)

// Declared static because otherwise these overflow the stack.
static gf16_t a[GF16_ORDER][W];
static gf16_t b[GF16_ORDER][W];
static gf16_t c[GF16_ORDER][W];

static void compare_results(const char *name) {
    for (int i = 0; i < GF16_ORDER; ++i) {
        for (int j = 0; j < W; ++j) {
            if (b[i][j] != c[i][j]) {
                fprintf(stderr, "%s failed!\n", name);
                fprintf(stderr, "i=%d j=%d a[i][j]=%04x b[i][j]=%04x c[i][j]=%04x\n",
                        i, j, a[i][j], b[i][j], c[i][j]);
                exit(1);
            }
        }
    }
}

static void monomial_to_novel_test() {
    // Initialize b and c to different contents to avoid false positives.
    memset(b, -1, sizeof(b));
    memset(c, -2, sizeof(c));

    // First, use the vector function to fill b, the expected output.
    for (int j = 0; j < W; ++j) {
        gf16_t v[GF16_ORDER];
        for (int i = 0; i < GF16_ORDER; ++i) v[i] = a[i][j];
        monomial_to_novel_transpose(v);
        for (int i = 0; i < GF16_ORDER; ++i) b[i][j] = v[i];
    }

    // Second, use the batch function to fill c, the tested output.
    memcpy(c, a, sizeof(c));
    gf16_t *p[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) p[i] = &c[i][0];
    gf16_vt_batch_monomial_to_novel(p, W);

    // Compare the results.
    compare_results("monomial_to_novel_test");
}

static void afft_test() {
    // Initialize b and c to different contents to avoid false positives.
    memset(b, -1, sizeof(b));
    memset(c, -2, sizeof(c));

    // First, use the vector function to fill b, the expected output.
    for (int j = 0; j < W; ++j) {
        gf16_t v[GF16_ORDER];
        for (int i = 0; i < GF16_ORDER; ++i) v[i] = a[i][j];
        additive_fft_transpose(v);
        for (int i = 0; i < GF16_ORDER; ++i) b[i][j] = v[i];
    }

    // Second, use the batch function to fill c, the tested output.
    memcpy(c, a, sizeof(c));
    gf16_t *p[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) p[i] = &c[i][0];
    gf16_vt_batch_afft(p, W);

    // Compare the results.
    compare_results("afft_test");
}

static void vt_mul_test() {
    // Initialize b and c to different contents to avoid false positives.
    memset(b, -1, sizeof(b));
    memset(c, -2, sizeof(c));

    // First, use the vector function to fill b, the expected output.
    for (int j = 0; j < W; ++j) {
        gf16_t src[GF16_ORDER];
        gf16_t dst[GF16_ORDER];
        for (int i = 0; i < GF16_ORDER; ++i) src[i] = a[i][j];
        gf16_vt_mul(dst, src);
        for (int i = 0; i < GF16_ORDER; ++i) b[i][j] = dst[i];
    }

    // Second, use the batch function to fill c, the tested output.
    const gf16_t *src[GF16_ORDER];
    gf16_t *dst[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) {
        src[i] = &a[i][0];
        dst[i] = &c[i][0];
    }
    gf16_vt_batch_mul(dst, src, W);

    // Compare the results.
    compare_results("vt_mul_test");
}

int main() {

    uint64_t rng_state = 0x123456789abcdef;
    w1rand_fill(a, sizeof(gf16_t) * GF16_ORDER * W, &rng_state);

    gf16_init();
    gf16_vt_batch_mul_init();

    monomial_to_novel_test();
    afft_test();
    vt_mul_test();
}
