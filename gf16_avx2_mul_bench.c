#include "gf16.h"
#include "w1rand.h"
#include "gf16_avx2_mul.h"

#include <immintrin.h>
#include <inttypes.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// Multiply a vector by a constant
typedef void (f_t)(const gf16_t, const gf16_t *, gf16_t *, size_t);

static void *xalloc(size_t n) {
    void *p = malloc(n);
    if (p == NULL) {
        perror("malloc");
        exit(1);
    }
    return p;
}

static void mul_lut(gf16_t c, const gf16_t *src, gf16_t *dst, size_t n) {
    if (c == 0) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = 0;
        }
    } else {
        gf16_t log_c = gf16_log(c);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = gf16_mul_log(src[i], log_c);
        }
    }
}

static void mul_avx(gf16_t c, const gf16_t *src, gf16_t *dst, size_t n) {
    memset(dst, 0, sizeof(gf16_t) * n);
    gf16_avx2_mul_and_xor(dst, src, c, n);
}

double benchmark_mul(const char *name, f_t *func, int width, int niter) {
    // I use this property below to randomly fill the buffer.
    assert(sizeof(uint64_t) % sizeof(gf16_t) == 0);

    assert(0 < width);
    assert(width % 16 == 0);

    printf("Benchmarking %s (width %d) (%d iterations)...\n", name, width, niter);

    clock_t elapsed_clock = 0;
    uint64_t input_bytes = 0;
    uint64_t checksum = 0;
    uint64_t rng_state = 0x123456789abcdef;

    gf16_t a[GF16_ORDER];
    for (int i = 0; i < GF16_ORDER; ++i) a[i] = i;
    w1rand_shuffle(GF16_ORDER, &rng_state, gf16_swap, a);

    size_t len = width < GF16_ORDER ? GF16_ORDER : width;
    gf16_t *b = xalloc(sizeof(gf16_t) * len);
    for (int i = 0; i < len; ++i) b[i] = i % GF16_ORDER;
    w1rand_shuffle_prefix(width, len, &rng_state, gf16_swap, b);

    gf16_t *c = xalloc(sizeof(gf16_t) * len);

    for (int iter = 0; iter < niter; ++iter) {
        gf16_t x = a[iter % GF16_ORDER];
        clock_t clock_begin = clock();
        func(x, b, c, width);
        elapsed_clock += clock() - clock_begin;

        // Use the result to make sure nothing gets optimized away
        // (also useful for verifying different functions return the same result)
        for (int i = 0; i < GF16_ORDER; ++i) checksum = checksum * 13u + c[i];

        input_bytes += width * sizeof(gf16_t);
    }
    free(b);
    free(c);
    double elapsed_secs = (double) elapsed_clock / CLOCKS_PER_SEC;
    double throughput = (double) input_bytes / (1 << 20) / elapsed_secs;
    printf("Time elapsed: %.6f s\n", elapsed_secs);
    printf("Throughput: %.3f MiB/s\n", throughput);
    printf("Checksum: %" PRIx64 "\n", checksum);
    return throughput;
}

int main() {
    gf16_init();
    gf16_avx2_mul_init();

    gf16_t a[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255 };
    for (int x = 0; x < GF16_ORDER; ++x) {
        gf16_t b[16] = {};
        gf16_t c[16] = {};
        mul_lut(42, a, b, 16);
        mul_avx(42, a, c, 16);
        // printf("a:"); for (int i = 0; i < 16; ++i) printf(" %04x", a[i]); printf("\n");
        // printf("b:"); for (int i = 0; i < 16; ++i) printf(" %04x", b[i]); printf("\n");
        // printf("c:"); for (int i = 0; i < 16; ++i) printf(" %04x", c[i]); printf("\n");
        for (int i = 0; i < 16; ++i) assert(b[i] == c[i]);
    }

    //const int W = 20;
    #define W 20

    double throughput[W + 1][2];

    for (int w = 4; w <= W; ++w) {
        int width = 1 << w;
        throughput[w][0] = benchmark_mul("lut", mul_lut, width, GF16_ORDER);
        throughput[w][1] = benchmark_mul("avx", mul_avx, width, GF16_ORDER);
    }

    printf("Elements:     ");
    for (int w = 4; w <= W; ++w) printf("%8d", (1<<w));
    printf("\n");
    printf("Lookup table: ");
    for (int w = 4; w <= W; ++w) printf("%8.2f", throughput[w][0]);
    printf(" MiB/s\n");
    printf("Nibble AVX2:  ");
    for (int w = 4; w <= W; ++w) printf("%8.2f", throughput[w][1]);
    printf(" MiB/s\n");
}
