#include "w1rand.h"

#include "vendor/md5/md5.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static const unsigned char expected_digest[MD5_DIGEST_LENGTH] = {
    0xa5, 0xd8, 0x1d, 0xad, 0x62, 0xad, 0xad, 0xf7,
    0x62, 0xad, 0x2d, 0x7d, 0x67, 0x90, 0x18, 0xbc };

int benchmark(const unsigned char *buf, size_t len, size_t chunk_len) {
    clock_t clock_begin = clock();
    MD5_CTX md5_ctx;
    MD5_Init(&md5_ctx);
    for (size_t i = 0; i < len; ) {
        size_t n = len - i < chunk_len ? len - i : chunk_len;
        MD5_Update(&md5_ctx, buf + i, n);
        i += n;
    }
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5_Final(digest, &md5_ctx);
    double time_elapsed = (double) (clock() - clock_begin) / CLOCKS_PER_SEC;
    double throughput = (double) len / (1 << 20) / time_elapsed;
    printf(" hashed %zd bytes in %9zd byte chunks", len, chunk_len);
    printf(" in %.6f seconds; throughput: %.3f MiB/s\n", time_elapsed, throughput);

    if (memcmp(digest, expected_digest, MD5_DIGEST_LENGTH) == 0) {
        return 0;
    } else {
        fprintf(stderr, "Digest verification failed!\nExpected: ");
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
            fprintf(stderr, "%02x", expected_digest[i]);
        }
        fprintf(stderr, "\nReceived: ");
        for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
            fprintf(stderr, "%02x", digest[i]);
        }
        fprintf(stderr, "\n");
        return 1;
    }
}

int main() {
    size_t len = 512 << 20;  // 512 MiB
    unsigned char *buf = malloc(len);
    if (buf == NULL) {
        perror("malloc");
        return 1;
    }

    uint64_t rng_state = 0x0123456789abcdef;
    w1rand_fill(buf, len, &rng_state);

    int failures = 0;
    failures += benchmark(buf, len, 16);
    failures += benchmark(buf, len, 123);
    failures += benchmark(buf, len, 1000);
    failures += benchmark(buf, len, 1000000);
    failures += benchmark(buf, len, 10000000);
    failures += benchmark(buf, len, len);
    if (failures == 0) {
        printf("All tests passed.\n");
        return 0;
    } else {
        fprintf(stderr, "%d tests failed!\n", failures);
        return 1;
    }
}
