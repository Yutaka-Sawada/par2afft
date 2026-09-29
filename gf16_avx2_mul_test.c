#include "gf16.h"
#include "gf16_avx2_mul.h"
#include "w1rand.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void test_multiplier(gf16_t x) {
    //fprintf(stderr, "x=%04x\n", x);
    gf16_t a[GF16_ORDER];
    gf16_t b[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) {
        a[i] = i;
        b[i] = 0;
    }
    gf16_avx2_mul_and_xor(b, a, x, GF16_ORDER);
    for (int i = 0; i < GF16_ORDER; ++i) {
        gf16_t c = gf16_mul(a[i], x);
        if (b[i] != c) {
            fprintf(stderr, "Error at index %d: mul(%04x, %04x) expected %04x, received %04x!",
                i, a[i], x, c, b[i]);
            exit(1);
        }
    }
}

static void test_mul() {
    // Test the first 100 multipliers (importantly, this includes 0 and 1)
    for (int x = 0; x < 100; ++x) test_multiplier(x);

    // Test the last 100 multipliers
    for (int x = 0; x < 100; ++x) test_multiplier(GF16_ORDER - 1 - x);

    // Test 1000 random multipliers
    const int n = 1000;
    gf16_t multipliers[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) multipliers[i] = i;
    uint64_t rng_state = 0x123456789abcdef;
    w1rand_shuffle_prefix(n, GF16_ORDER, &rng_state, gf16_swap, multipliers);
    for (int i = 0; i < n; ++i) {
        test_multiplier(multipliers[i]);
    }
}

static void test_xor() {
    gf16_t a[GF16_ORDER];
    gf16_t b[GF16_ORDER];
    gf16_t c[GF16_ORDER];

    for (int i = 0; i < GF16_ORDER; ++i) {
        a[i] = i;
        b[i] = i;
    }

    uint64_t rng_state = 0x123456789abcdef;
    w1rand_shuffle(GF16_ORDER, &rng_state, gf16_swap, a);
    w1rand_shuffle(GF16_ORDER, &rng_state, gf16_swap, b);
    memcpy(c, b, sizeof(gf16_t) * GF16_ORDER);
    const int batch_size = 16 * 123;
    for (int i = 0; i < GF16_ORDER; ++i) {
        b[i] ^= gf16_mul(a[i], i / batch_size);
    }
    for (int i = 0; i < GF16_ORDER; i += batch_size) {
        int n = GF16_ORDER - i < batch_size ? GF16_ORDER - i : batch_size;
        gf16_avx2_mul_and_xor(c + i, a + i, i/batch_size, n);
    }

    for (int i = 0; i < GF16_ORDER; ++i) {
        if (b[i] != c[i]) {
            fprintf(stderr, "Difference at index %d: expected %04x, received %04x!\n",
                i, b[i], c[i]);
            exit(1);
        }
    }
}

int main() {
    gf16_init();
    gf16_avx2_mul_init();

    test_mul();
    test_xor();
}
