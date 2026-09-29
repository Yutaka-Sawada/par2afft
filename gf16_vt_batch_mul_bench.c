#include "gf16.h"
#include "gf16_vt_batch_mul.h"
#include "w1rand.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>

typedef void (func_t)(
    gf16_t *const restrict y[restrict static GF16_ORDER],
    const gf16_t *const restrict a[restrict static GF16_ORDER],
    size_t width);

static void *xalloc(size_t n) {
    void *p = malloc(n);
    if (p == NULL) {
        perror("malloc");
        exit(1);
    }
    return p;
}

static void benchmark(func_t func, size_t width) {
    uint64_t rng_state = 0x123456789abcdef;

    gf16_t **a = xalloc(sizeof(gf16_t*) * GF16_ORDER);
    gf16_t **b = xalloc(sizeof(gf16_t*) * GF16_ORDER);
    for (int i = 0; i < GF16_ORDER; ++i) {
        a[i] = xalloc(sizeof(gf16_t) * width);
        b[i] = xalloc(sizeof(gf16_t) * width);
        w1rand_fill(a[i], width * sizeof(gf16_t), &rng_state);
        w1rand_fill(b[i], width * sizeof(gf16_t), &rng_state);
    }
    const gf16_t *const *src = (const gf16_t *const*)a;
    gf16_t *const *dst = (gf16_t *const *)b;

    const int niter = 10;
    printf("Running benchmark (width %zd) (niter=%d)...\n", width, niter);
    const uint64_t input_bytes = sizeof(gf16_t) * GF16_ORDER * width * niter;
    uint64_t checksum = 0;
    double time_elapsed = 0;
    for (int iter = 0; iter < niter; ++iter) {
        clock_t clock_begin = clock();
        func(dst, src, width);
        time_elapsed += (double) (clock() - clock_begin) / CLOCKS_PER_SEC;

        // Use the result to make sure nothing gets optimized away
        for (int i = 0; i < GF16_ORDER; ++i) {
            for (int j = 0; j < width; ++j) {
                checksum = (13 * checksum + dst[i][j]);
            }
        }
    }

    for (int i = 0; i < GF16_ORDER; ++i) {
        free(a[i]);
        free(b[i]);
    }
    free(a);
    free(b);

    double throughput = (double) input_bytes / (1 << 20) / time_elapsed;
    printf("Time elapsed: %.6f s\n", time_elapsed);
    printf("Throughput: %.3f MiB/s\n", throughput);
    printf("Checksum: %" PRIx64 "\n", checksum);
}

int main() {
    gf16_init();
    gf16_vt_batch_mul_init();
    benchmark(gf16_vt_batch_mul, 1024);
    benchmark(gf16_vt_batch_mul, 2048);
    benchmark(gf16_vt_batch_mul, 4096);
    benchmark(gf16_vt_batch_mul, 8192);
}
