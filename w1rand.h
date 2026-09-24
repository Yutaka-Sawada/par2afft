#ifndef W1RAND_H_INCLUDED
#define W1RAND_H_INCLUDED

#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Generates a pseudo-random 64-bit number really quickly.
//
// If the initial state is 0, then the first number returned is also 0,
// so it's best to use any other number for the initial state.
//
// From: https://github.com/wangyi-fudan/wyhash
static inline uint64_t w1rand(uint64_t *s) {
    const uint64_t c = 0xd07ebc63274654c7ull;
    *s += c;
    __uint128_t t = (__uint128_t) *s * (*s ^ c);
    return (t >> 64) ^ t;
}

// Fills `buf` with `len` random bytes.
//
// This copies results from w1rand() in the native byte order, so the results
// are not portable between different architectures.
static inline void *w1rand_fill(void *buf, size_t len, uint64_t *s) {
    char *ptr = (char*) buf;
    while (len >= sizeof(uint64_t)) {
        uint64_t val = w1rand(s);
        memcpy(ptr, &val, sizeof(uint64_t));
        ptr += sizeof(uint64_t);
        len -= sizeof(uint64_t);
    }
    if (len > 0) {
        uint64_t val = w1rand(s);
        memcpy(ptr, &val, len);
    }
    return buf;
}

typedef void (*w1rand_swap_fn)(void *ctx, size_t i, size_t j);

// Uses w1rand() to shuffle the first `m` of a total of `n` elements.
//
// `m` must be less than or equal to `n`.
//
// This function will call swap(i, j, ctx) up to `m` times to indicate `i` and
// `j` should be swapped, where 0 <= i < j < n.
//
// Afterwards, the first `m` elements are a random selection of all `n` elements
// in a random order.
//
// To shuffle the entire array, call w1rand_shuffle() defined below.
static inline void w1rand_shuffle_prefix(
        size_t m, size_t n, uint64_t *rng_state,
        w1rand_swap_fn swap, void *ctx)
{
    for (size_t i = 0; i < m; ++i) {
        size_t k = w1rand(rng_state) % (n - i);
        if (k != 0) swap(ctx, i, i + k);
    }
}

// Shuffles all `n` elements. See w1rand_shuffle_prefix() for details.
static inline void w1rand_shuffle(size_t n, uint64_t *rng_state, w1rand_swap_fn swap, void *ctx) {
    w1rand_shuffle_prefix(n - 1, n, rng_state, swap, ctx);
}

#endif // ndef W1RAND_H_INCLUDED
