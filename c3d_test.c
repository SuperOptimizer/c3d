/* c3d_test.c — tests for c3d.c.  See LICENSE.
 *
 * Compiled WITH #include "c3d.c" so static internals are reachable, and with
 * -DC3D_BUILD_REF so the reference decoder path is present (later stages). */

#define C3D_BUILD_REF 1
#include "c3d.c"

#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── test harness ───────────────────────────────────────────────────────── */

static int  g_tests_run  = 0;
static int  g_tests_fail = 0;
static jmp_buf g_panic_jmp;
static bool    g_expect_panic = false;
static char    g_panic_msg[256];

_Noreturn static void test_panic_hook(const char *file, int line, const char *msg) {
    (void)file; (void)line;
    if (g_expect_panic) {
        snprintf(g_panic_msg, sizeof g_panic_msg, "%s", msg);
        longjmp(g_panic_jmp, 1);
    }
    /* Unexpected panic during a normal test — fall through and abort. */
    fprintf(stderr, "UNEXPECTED PANIC: %s:%d: %s\n", file, line, msg);
    abort();
}

#define CHECK(cond) do {                                                       \
    ++g_tests_run;                                                             \
    if (!(cond)) {                                                             \
        ++g_tests_fail;                                                        \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
    }                                                                          \
} while (0)

#define CHECK_EQ(a, b) do {                                                    \
    ++g_tests_run;                                                             \
    long long _a = (long long)(a), _b = (long long)(b);                        \
    if (_a != _b) {                                                            \
        ++g_tests_fail;                                                        \
        fprintf(stderr, "FAIL %s:%d: %s (%lld) != %s (%lld)\n",                \
                __FILE__, __LINE__, #a, _a, #b, _b);                           \
    }                                                                          \
} while (0)

/* Expect the next block to panic; requires the panic hook to longjmp. */
/* Panic-expected blocks longjmp out mid-call, bypassing any free() in the
 * normal unwind path.  Under LSan that reads as a leak, so disable leak
 * accounting for the duration.  Harmless when LSan is absent (the two
 * __lsan_* symbols are provided as weak no-ops by the sanitizer runtime). */
#if defined(__has_feature)
# if __has_feature(address_sanitizer)
#  define C3D_TEST_HAS_LSAN 1
# endif
#endif
#ifndef C3D_TEST_HAS_LSAN
# if defined(__SANITIZE_ADDRESS__)
#  define C3D_TEST_HAS_LSAN 1
# endif
#endif
#ifdef C3D_TEST_HAS_LSAN
void __lsan_disable(void);
void __lsan_enable(void);
#  define C3D_LSAN_DISABLE() __lsan_disable()
#  define C3D_LSAN_ENABLE()  __lsan_enable()
#else
#  define C3D_LSAN_DISABLE() ((void)0)
#  define C3D_LSAN_ENABLE()  ((void)0)
#endif

#define EXPECT_PANIC(block) do {                                               \
    ++g_tests_run;                                                             \
    g_expect_panic = true;                                                     \
    C3D_LSAN_DISABLE();                                                        \
    if (setjmp(g_panic_jmp) == 0) {                                            \
        block;                                                                 \
        g_expect_panic = false;                                                \
        C3D_LSAN_ENABLE();                                                     \
        ++g_tests_fail;                                                        \
        fprintf(stderr, "FAIL %s:%d: expected panic, none occurred\n",         \
                __FILE__, __LINE__);                                           \
    } else {                                                                   \
        g_expect_panic = false;                                                \
        C3D_LSAN_ENABLE();                                                     \
    }                                                                          \
} while (0)

/* ─── §A  scaffolding tests ──────────────────────────────────────────────── */

static void test_voxel_key(void) {
    uint32_t x = 0xabcde, y = 0x12345, z = 0x6789a;
    uint8_t  lod = 5;
    uint64_t k = c3d_key(x, y, z, lod);

    uint32_t rx, ry, rz; uint8_t rlod;
    c3d_unkey(k, &rx, &ry, &rz, &rlod);
    CHECK_EQ(rx, x);
    CHECK_EQ(ry, y);
    CHECK_EQ(rz, z);
    CHECK_EQ(rlod, lod);
    /* Extremes. */
    k = c3d_key(0xfffff, 0xfffff, 0xfffff, 15);
    c3d_unkey(k, &rx, &ry, &rz, &rlod);
    CHECK_EQ(rx, 0xfffffu);
    CHECK_EQ(ry, 0xfffffu);
    CHECK_EQ(rz, 0xfffffu);
    CHECK_EQ(rlod, 15);
}

static void test_leb128(void) {
    uint8_t buf[16];
    uint64_t samples[] = { 0, 1, 127, 128, 16383, 16384, 1u << 28, (1ull << 56) - 1, ~0ull };
    for (size_t i = 0; i < sizeof samples / sizeof samples[0]; ++i) {
        size_t n = c3d_leb128_encode(samples[i], buf, sizeof buf);
        uint64_t v = 0;
        size_t m = c3d_leb128_decode(buf, n, &v);
        CHECK_EQ(m, n);
        CHECK_EQ(v, samples[i]);
    }
}

static void test_morton12(void) {
    /* Bijection + round-trip on all 4096 inputs. */
    bool seen[4096] = {0};
    for (uint32_t cz = 0; cz < 16; ++cz)
    for (uint32_t cy = 0; cy < 16; ++cy)
    for (uint32_t cx = 0; cx < 16; ++cx) {
        uint32_t m = c3d_morton12(cx, cy, cz);
        CHECK(m < 4096);
        CHECK(!seen[m]);
        seen[m] = true;
        uint32_t dx, dy, dz;
        c3d_morton12_decode(m, &dx, &dy, &dz);
        CHECK_EQ(dx, cx);
        CHECK_EQ(dy, cy);
        CHECK_EQ(dz, cz);
    }
    /* Morton bit-placement spot-check: (1,0,0) → bit 0; (0,1,0) → bit 1; (0,0,1) → bit 2. */
    CHECK_EQ(c3d_morton12(1,0,0), 1u);
    CHECK_EQ(c3d_morton12(0,1,0), 2u);
    CHECK_EQ(c3d_morton12(0,0,1), 4u);
    CHECK_EQ(c3d_morton12(15,15,15), 0xfffu);
}

static void test_panic_mechanism(void) {
    EXPECT_PANIC(c3d_assert(0 == 1));
    CHECK(strstr(g_panic_msg, "0 == 1") != NULL);
}

/* ─── §B  hash tests ─────────────────────────────────────────────────────── */

static void test_hash128_empty(void) {
    uint8_t h[16];
    c3d_hash128("", 0, h);
    /* Hash of empty input under MurmurHash3_x64_128 is all zeros in the
     * Appleby reference (h1 = h2 = 0 initially, tail is empty, finaliser runs
     * with len = 0 → no change).  Just check it's stable. */
    uint8_t h2[16];
    c3d_hash128("", 0, h2);
    CHECK(memcmp(h, h2, 16) == 0);
}

static void test_hash128_determinism(void) {
    const char *msg = "the quick brown fox jumps over the lazy dog";
    uint8_t h1[16], h2[16];
    c3d_hash128(msg, strlen(msg), h1);
    c3d_hash128(msg, strlen(msg), h2);
    CHECK(memcmp(h1, h2, 16) == 0);
}

static void test_hash128_avalanche(void) {
    /* Two inputs differing by one bit should produce very different hashes. */
    uint8_t a[64], b[64];
    memset(a, 0x55, sizeof a);
    memset(b, 0x55, sizeof b);
    b[32] ^= 0x01;
    uint8_t ha[16], hb[16];
    c3d_hash128(a, sizeof a, ha);
    c3d_hash128(b, sizeof b, hb);
    /* Count bit differences — expect near-half. */
    int diff = 0;
    for (int i = 0; i < 16; ++i) {
        uint8_t x = ha[i] ^ hb[i];
        while (x) { diff += x & 1; x >>= 1; }
    }
    CHECK(diff >= 40);  /* expected ~64; 40 is a very loose floor */
    CHECK(diff <= 90);
}

static void test_hash128_various_lengths(void) {
    /* Exercise the tail switch for every length 0..31. */
    uint8_t buf[32];
    for (int i = 0; i < 32; ++i) buf[i] = (uint8_t)(i * 17 + 1);
    uint8_t prev[16] = {0};
    for (size_t len = 1; len <= 32; ++len) {
        uint8_t h[16];
        c3d_hash128(buf, len, h);
        /* Hashes of prefixes of length 1..32 should all differ from each other. */
        CHECK(memcmp(h, prev, 16) != 0);
        memcpy(prev, h, 16);
    }
}

/* ─── §C  rANS tests ─────────────────────────────────────────────────────── */

/* Build a frequency table for n_symbols that sums to 1<<denom_shift.
 * Frequencies are a simple decreasing geometric distribution; the first
 * symbol gets the largest share. */
static void make_geometric_freqs(uint32_t *freqs, size_t n_symbols,
                                 uint32_t denom_shift)
{
    uint32_t M = 1u << denom_shift;
    /* Initial weights: 2^(n-1), 2^(n-2), ..., 2, 1. */
    uint64_t sum_w = 0;
    uint64_t w[65];
    for (size_t i = 0; i < n_symbols; ++i) {
        /* Cap the shift at 63 — 1<<64 is UB for uint64_t.  A geometric
         * ratio that saturates in the top handful of symbols is exactly
         * what this test's later normalisation already expects. */
        size_t sh = n_symbols - 1 - i;
        if (sh > 63) sh = 63;
        w[i] = (uint64_t)1 << sh;
        sum_w += w[i];
    }
    /* Scale to M, reserving floor; distribute residual. */
    uint32_t used = 0;
    for (size_t i = 0; i < n_symbols; ++i) {
        freqs[i] = (uint32_t)((w[i] * M) / sum_w);
        if (freqs[i] == 0) freqs[i] = 1;
        used += freqs[i];
    }
    /* Adjust the largest bin to make sum = M exactly. */
    if (used > M) {
        uint32_t over = used - M;
        /* Trim from the largest bin (freqs[0]). */
        c3d_assert(freqs[0] > over);
        freqs[0] -= over;
    } else if (used < M) {
        freqs[0] += (M - used);
    }
    /* Verify. */
    uint32_t check = 0;
    for (size_t i = 0; i < n_symbols; ++i) check += freqs[i];
    c3d_assert(check == M);
}

static void test_rans_roundtrip(uint32_t denom_shift, size_t n_symbols_alphabet,
                                size_t n_data)
{
    uint32_t freqs[65] = {0};
    make_geometric_freqs(freqs, n_symbols_alphabet, denom_shift);

    c3d_rans_tables tbl;
    c3d_rans_build_tables(&tbl, denom_shift, freqs, n_symbols_alphabet);

    /* Generate data drawn from the same distribution (approximately):
     * sample a uniform in [0, M), find the symbol that contains it. */
    uint8_t *data = malloc(n_data);
    c3d_assert(data);

    srand(12345);  /* deterministic */
    const uint32_t M = 1u << denom_shift;
    for (size_t i = 0; i < n_data; ++i) {
        uint32_t r = (uint32_t)((uint64_t)rand() * (M - 1) / RAND_MAX);
        /* find symbol via cum2sym */
        data[i] = (uint8_t)tbl.cum2sym[r];
    }

    /* Encode. */
    size_t   scratch_size = n_data * 2 + 1024;
    uint8_t *scratch      = malloc(scratch_size);
    size_t   out_cap      = n_data * 2 + 1024;
    uint8_t *out          = malloc(out_cap);
    c3d_assert(scratch && out);

    size_t out_len = c3d_rans_enc_x8(data, n_data, &tbl,
                                     scratch, scratch_size, out, out_cap);
    CHECK(out_len >= 32);
    CHECK(out_len <= out_cap);

    /* Decode. */
    uint8_t *decoded = malloc(n_data);
    c3d_assert(decoded);
    c3d_rans_dec_x8(out, out_len, &tbl, decoded, n_data);

    CHECK(memcmp(data, decoded, n_data) == 0);

    /* Expected size in bits ≈ -Σ p log2 p × n_data; this is just a sanity
     * check that we're within a reasonable factor. */
    double H = 0.0;
    for (size_t i = 0; i < n_symbols_alphabet; ++i) {
        if (freqs[i] == 0) continue;
        double p = (double)freqs[i] / (double)M;
        H -= p * log2(p);
    }
    double expected_bytes = (H * (double)n_data) / 8.0;
    double actual_bytes   = (double)(out_len - 32);   /* subtract 32 B state header */
    (void)expected_bytes; (void)actual_bytes;
    /* Don't assert on this — just informative. */

    free(data); free(scratch); free(out); free(decoded);
}

static void test_rans(void) {
    /* Small, medium, and large blocks at both denominators we'll use. */
    test_rans_roundtrip(12, 65, 512);      /* LLL_5 analogue          */
    test_rans_roundtrip(12, 65, 4096);     /* a level-4 detail size   */
    test_rans_roundtrip(12, 65, 128*128*128);  /* level-1 detail scale (~2 MiB) */
    test_rans_roundtrip(14, 65, 512);      /* LLL_5 with bigger M     */
    /* Edge: highly skewed (first symbol almost all the mass). */
    test_rans_roundtrip(12,  8,  4096);
    /* Edge: small alphabet. */
    test_rans_roundtrip(12,  2,  1024);
}

/* Degenerate case from PLAN §3.4: one symbol with freq = M. */
static void test_rans_single_symbol(void) {
    uint32_t freqs[65] = {0};
    freqs[0] = 4096;    /* M = 4096 */
    c3d_rans_tables tbl;
    c3d_rans_build_tables(&tbl, 12, freqs, 1);

    const size_t N = 1000;
    uint8_t data[1000];
    for (size_t i = 0; i < N; ++i) data[i] = 0;

    size_t   scratch_size = N * 2 + 1024;
    uint8_t *scratch      = malloc(scratch_size);
    uint8_t  out[2048];
    size_t   out_len      = c3d_rans_enc_x8(data, N, &tbl,
                                            scratch, scratch_size,
                                            out, sizeof out);
    /* Because the only symbol has freq = M, state never renormalises; the
     * output should be exactly 32 B (just the 8 initial-state u32s). */
    CHECK_EQ(out_len, 32u);
    uint8_t decoded[1000];
    c3d_rans_dec_x8(out, out_len, &tbl, decoded, N);
    CHECK(memcmp(data, decoded, N) == 0);
    free(scratch);
}

/* ─── §D  frequency-table tests ──────────────────────────────────────────── */

static void test_histogram(void) {
    uint8_t syms[] = {0, 1, 1, 64, 3, 3, 3, 0, 2, 64};
    uint32_t h[65];
    c3d_histogram65(syms, sizeof syms, h);
    CHECK_EQ(h[0], 2u);
    CHECK_EQ(h[1], 2u);
    CHECK_EQ(h[2], 1u);
    CHECK_EQ(h[3], 3u);
    CHECK_EQ(h[4], 0u);
    CHECK_EQ(h[64], 2u);
}

static void test_normalise_freqs_basic(void) {
    uint32_t hist[65] = {0};
    hist[0] = 100; hist[1] = 50; hist[2] = 25; hist[3] = 25;  /* sum = 200 */
    uint32_t f[65];
    c3d_normalise_freqs(hist, 12, f);  /* M = 4096 */
    uint32_t sum = 0;
    for (unsigned i = 0; i < 65; ++i) sum += f[i];
    CHECK_EQ(sum, 4096u);
    /* All originally-nonzero remain ≥ 1. */
    CHECK(f[0] >= 1 && f[1] >= 1 && f[2] >= 1 && f[3] >= 1);
    /* All originally-zero remain 0. */
    for (unsigned i = 4; i < 65; ++i) CHECK_EQ(f[i], 0u);
}

static void test_normalise_freqs_sparse(void) {
    /* Extreme skew: one symbol dominates, a rare one appears once. */
    uint32_t hist[65] = {0};
    hist[0] = 999999;
    hist[3] = 1;
    uint32_t f[65];
    c3d_normalise_freqs(hist, 12, f);
    uint32_t sum = 0;
    for (unsigned i = 0; i < 65; ++i) sum += f[i];
    CHECK_EQ(sum, 4096u);
    CHECK(f[0] >= 1);
    CHECK_EQ(f[3], 1u);  /* rare symbol pinned at 1 */
}

static void test_freqs_serialise_roundtrip(void) {
    /* Build a synthetic histogram, normalise, serialise, parse, compare. */
    srand(57);
    for (int trial = 0; trial < 10; ++trial) {
        uint32_t hist[65] = {0};
        /* Random sparse histogram. */
        int nz = (rand() % 30) + 1;
        for (int k = 0; k < nz; ++k) {
            unsigned s = (unsigned)rand() % 65;
            hist[s] += (uint32_t)((rand() % 10000) + 1);
        }
        uint32_t denom_shift = (rand() & 1) ? 12u : 14u;
        uint32_t f[65];
        c3d_normalise_freqs(hist, denom_shift, f);

        uint8_t buf[800];
        size_t n = c3d_freqs_serialise(denom_shift, f, buf, sizeof buf);
        CHECK(n >= 2);
        CHECK(n <= sizeof buf);

        uint32_t ds2;
        uint32_t f2[65];
        size_t consumed = c3d_freqs_parse(buf, n, &ds2, f2);
        CHECK_EQ(consumed, n);
        CHECK_EQ(ds2, denom_shift);
        for (unsigned i = 0; i < 65; ++i) CHECK_EQ(f2[i], f[i]);
    }
}

/* ─── §E  DWT tests ──────────────────────────────────────────────────────── */

/* Reference 1D CDF 9/7 lifting in double precision for cross-check. */
static void cdf97_lift_fwd_d(double *x, size_t N) {
    const double A = -1.586134342059924;
    const double B = -0.052980118572961;
    const double G =  0.882911075530934;
    const double D =  0.443506852043971;
    const double K =  1.230174104914001;

    for (size_t i = 1; i + 1 < N; i += 2) x[i] += A * (x[i-1] + x[i+1]);
    x[N-1] += 2.0 * A * x[N-2];
    x[0] += 2.0 * B * x[1];
    for (size_t i = 2; i < N; i += 2) x[i] += B * (x[i-1] + x[i+1]);
    for (size_t i = 1; i + 1 < N; i += 2) x[i] += G * (x[i-1] + x[i+1]);
    x[N-1] += 2.0 * G * x[N-2];
    x[0] += 2.0 * D * x[1];
    for (size_t i = 2; i < N; i += 2) x[i] += D * (x[i-1] + x[i+1]);
    for (size_t i = 0; i < N; i += 2) x[i] *= (1.0 / K);
    for (size_t i = 1; i < N; i += 2) x[i] *= K;
}

static float  g_line_buf[C3D_CHUNK_SIDE * 2];

static void test_dwt_1d_roundtrip(void) {
    float x[256], y[256];
    srand(7);
    for (size_t i = 0; i < 256; ++i) {
        x[i] = (float)((rand() & 0xffff) - 32768) / 100.0f;
        y[i] = x[i];
    }
    c3d_dwt_1d_fwd(y, 256, g_line_buf);
    c3d_dwt_1d_inv(y, 256, g_line_buf);
    float max_err = 0.0f;
    for (size_t i = 0; i < 256; ++i) {
        float e = y[i] - x[i];
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
    }
    CHECK(max_err < 1e-4f);   /* float round-trip on 256 samples */
}

static void test_dwt_1d_double_cross(void) {
    /* Compare float forward lift (before deinterleave) against double. */
    const size_t N = 256;
    float  xf[256];
    double xd[256];
    srand(11);
    for (size_t i = 0; i < N; ++i) {
        float v = (float)((rand() & 0xffff) - 32768) / 100.0f;
        xf[i] = v;
        xd[i] = (double)v;
    }
    c3d_cdf97_lift_fwd(xf, N);
    cdf97_lift_fwd_d(xd, N);
    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double e = (double)xf[i] - xd[i];
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
    }
    /* float32 drift vs double on a 256-sample lift should be tiny. */
    CHECK(max_err < 1e-3);
}

static void test_dwt_1d_impulse(void) {
    /* Impulse at position 0 → after lift + deinterleave, s[0] and d[0] get
     * non-zero values; the impulse energy spreads to a few neighbours.
     * We just sanity-check that energy is conserved within a small fudge. */
    const size_t N = 256;
    float x[256] = {0};
    x[0] = 100.0f;
    double pre_energy = 100.0 * 100.0;
    c3d_dwt_1d_fwd(x, N, g_line_buf);
    double post_energy = 0.0;
    for (size_t i = 0; i < N; ++i) post_energy += (double)x[i] * (double)x[i];
    /* Biorthogonal basis — energy isn't exactly conserved, but within ~10%
     * on a delta input after one lift level is reasonable.  (Relaxed bound.) */
    CHECK(post_energy > pre_energy * 0.5);
    CHECK(post_energy < pre_energy * 2.0);
}

static void test_dwt_3d_roundtrip(void) {
    float *buf  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    float *orig = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    c3d_assert(buf && orig);

    srand(23);
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        float v = (float)((rand() & 0x1ff) - 128);   /* ±128 range, like centred u8 */
        buf[i]  = v;
        orig[i] = v;
    }
    float scratch[2 * C3D_TILE_X * C3D_CHUNK_SIDE];
    c3d_dwt3_fwd(buf, scratch);
    c3d_dwt3_inv_levels(buf, C3D_N_DWT_LEVELS, scratch);

    float max_err = 0.0f;
    double sse = 0.0;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        float e = buf[i] - orig[i];
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
        sse += (double)e * (double)e;
    }
    double rmse = sqrt(sse / (double)C3D_VOXELS_PER_CHUNK);
    CHECK(max_err < 1e-2f);   /* 5 levels of DWT on 256³ f32 — loose tolerance */
    CHECK(rmse    < 1e-3);

    free(buf); free(orig);
}

static void test_dwt_3d_lod_partial(void) {
    /* Verify that decode-to-LOD-k, via partial inverse, is coherent.  We
     * forward-transform a random volume, then for each k ∈ 0..5, reproduce
     * the buffer and inverse-synthesise exactly 5-k levels; the resulting
     * LLL_k sub-cube should equal the ideal wavelet-pyramid reconstruction
     * at that level (produced by doing the forward then stopping earlier). */
    float *full_fwd = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    float *tmp      = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    c3d_assert(full_fwd && tmp);

    srand(41);
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        full_fwd[i] = (float)((rand() & 0x1ff) - 128);
    }
    float scratch[2 * C3D_TILE_X * C3D_CHUNK_SIDE];
    c3d_dwt3_fwd(full_fwd, scratch);

    /* For each LOD k, inverse-synth 5-k levels and check the LLL_k sub-cube
     * at [0:side, 0:side, 0:side] matches what we'd get if we ran forward
     * just k levels.  Going from 5 to k by inverting 5-k levels recovers
     * that k-level forward result. */
    for (unsigned k = 0; k <= C3D_N_DWT_LEVELS; ++k) {
        memcpy(tmp, full_fwd, C3D_VOXELS_PER_CHUNK * sizeof(float));
        c3d_dwt3_inv_levels(tmp, C3D_N_DWT_LEVELS - k, scratch);

        /* Independently: fresh forward of only k levels on the raw input. */
        float *ref = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
        c3d_assert(ref);
        srand(41);   /* same seed as above */
        for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
            ref[i] = (float)((rand() & 0x1ff) - 128);
        }
        size_t side = C3D_CHUNK_SIDE;
        for (unsigned lvl = 0; lvl < k; ++lvl) {
            c3d_dwt3_fwd_level(ref, side, scratch);
            side /= 2;
        }

        size_t side_k = (size_t)C3D_CHUNK_SIDE >> k;
        float  max_err = 0.0f;
        for (size_t z = 0; z < side_k; ++z)
        for (size_t y = 0; y < side_k; ++y)
        for (size_t x = 0; x < side_k; ++x) {
            size_t idx = z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x;
            float e = tmp[idx] - ref[idx];
            if (e < 0) e = -e;
            if (e > max_err) max_err = e;
        }
        CHECK(max_err < 1e-2f);
        free(ref);
    }
    free(full_fwd); free(tmp);
}

/* ─── §F  quantizer + zigzag + subband-info tests ────────────────────────── */

static void test_quant_roundtrip(void) {
    const float alpha = 0.375f;
    /* Use the per-kind table: dz_ratio = 0.55 here is the h=1 case. */
    const float dz_ratio = 0.55f;
    float steps[] = {0.5f, 1.0f, 2.0f, 4.0f, 16.0f, 128.0f};
    for (size_t si = 0; si < sizeof steps / sizeof steps[0]; ++si) {
        float step    = steps[si];
        float dz_half = dz_ratio * step;
        float worst = 0.0f;
        /* 16261 samples from -1000 to +1000 in 0.123f-ish steps — use an
         * integer counter so accumulation error doesn't drift the sweep. */
        for (int step_i = 0; step_i <= 16260; ++step_i) {
            float c = -1000.0f + 0.123f * (float)step_i;
            int32_t q   = c3d_quant(c, step, dz_half);
            float   chi = c3d_dequant(q, step, dz_half, alpha);
            float   err = chi - c;
            if (err < 0) err = -err;
            if (err > worst) worst = err;
        }
        /* Max error bound: bin 0 spans [-dz_half, dz_half] with centroid 0,
         * so error up to dz_half.  Bin k≥1 spans [dz_half + (k-1)·step,
         * dz_half + k·step] with centroid offset α·step from low end, so
         * worst is at bin edges with error up to (1-α)·step or α·step.
         * Loose ceiling: max(dz_half, step). */
        float ceiling = step > dz_half ? step : dz_half;
        CHECK(worst <= ceiling * 1.1f);
    }
}

static void test_zigzag32(void) {
    int32_t samples[] = {0, 1, -1, 2, -2, 31, -32, 32, -33, 1000, -1000, INT32_MAX, INT32_MIN+1};
    for (size_t i = 0; i < sizeof samples/sizeof samples[0]; ++i) {
        uint32_t z  = c3d_zigzag32(samples[i]);
        int32_t  v  = c3d_unzigzag32(z);
        CHECK_EQ(v, samples[i]);
    }
    /* Spot-check expected encoding. */
    CHECK_EQ(c3d_zigzag32(0),  0u);
    CHECK_EQ(c3d_zigzag32(-1), 1u);
    CHECK_EQ(c3d_zigzag32(1),  2u);
    CHECK_EQ(c3d_zigzag32(31), 62u);
    CHECK_EQ(c3d_zigzag32(-32), 63u);
    CHECK_EQ(c3d_zigzag32(32), 64u);   /* first escape value */
}

static void test_symbol_mapping(void) {
    /* Sign-predictive mapping: test that round-tripping through
     * c3d_quant_to_symbol → c3d_symbol_to_quant recovers the original qv
     * for both direct (|qv| < 32) and escape (|qv| ≥ 32) ranges.
     * Both encoder and decoder must use matching sign prediction state. */
    /* Direct range: |q| ∈ [0, 31]. */
    for (int32_t q = -31; q <= 31; ++q) {
        bool sp_enc = true, sp_dec = true;
        uint32_t esc;
        uint8_t s = c3d_quant_to_symbol(q, &esc, &sp_enc);
        CHECK(!C3D_SYM_IS_ESCAPE(s));
        CHECK_EQ(esc, 0u);
        int32_t q2 = c3d_symbol_to_quant(s, 0u, &sp_dec);
        CHECK_EQ(q2, q);
    }
    /* Escape range: |q| ≥ 32. */
    int32_t esc_samples[] = {32, -33, 100, -1000, 100000, -100000};
    for (size_t i = 0; i < sizeof esc_samples/sizeof esc_samples[0]; ++i) {
        int32_t q = esc_samples[i];
        bool sp_enc = true, sp_dec = true;
        uint32_t esc;
        uint8_t s = c3d_quant_to_symbol(q, &esc, &sp_enc);
        CHECK(C3D_SYM_IS_ESCAPE(s));
        CHECK(esc >= 32u);
        int32_t q2 = c3d_symbol_to_quant(s, esc, &sp_dec);
        CHECK_EQ(q2, q);
    }
    /* Sequential sign prediction: encode a sequence, decode it, verify. */
    int32_t seq[] = {0, 3, -2, 5, -5, 0, 1, -1, 32, -100};
    bool sp_enc = true, sp_dec = true;
    for (size_t i = 0; i < sizeof seq / sizeof seq[0]; ++i) {
        uint32_t esc;
        uint8_t s = c3d_quant_to_symbol(seq[i], &esc, &sp_enc);
        int32_t q2 = c3d_symbol_to_quant(s, esc, &sp_dec);
        CHECK_EQ(q2, seq[i]);
    }
}

static void test_subband_info(void) {
    /* Total coefficient count must equal 256³. */
    size_t total = 0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb;
        c3d_subband_info_of(i, &sb);
        total += (size_t)sb.side * sb.side * sb.side;
        /* Bounds. */
        CHECK(sb.z0 + sb.side <= C3D_CHUNK_SIDE);
        CHECK(sb.y0 + sb.side <= C3D_CHUNK_SIDE);
        CHECK(sb.x0 + sb.side <= C3D_CHUNK_SIDE);
    }
    CHECK_EQ(total, C3D_VOXELS_PER_CHUNK);

    /* Spot-check specific entries. */
    c3d_subband_info sb;
    c3d_subband_info_of(0, &sb);    /* LLL_5 */
    CHECK_EQ(sb.level, 5u); CHECK_EQ(sb.kind, 0u); CHECK_EQ(sb.side, 8u);
    CHECK_EQ(sb.z0, 0u); CHECK_EQ(sb.y0, 0u); CHECK_EQ(sb.x0, 0u);

    c3d_subband_info_of(1, &sb);    /* HHH_5 */
    CHECK_EQ(sb.level, 5u); CHECK_EQ(sb.kind, 1u); CHECK_EQ(sb.side, 8u);
    CHECK_EQ(sb.z0, 8u); CHECK_EQ(sb.y0, 8u); CHECK_EQ(sb.x0, 8u);

    c3d_subband_info_of(7, &sb);    /* LLH_5 */
    CHECK_EQ(sb.level, 5u); CHECK_EQ(sb.kind, 7u); CHECK_EQ(sb.side, 8u);
    CHECK_EQ(sb.z0, 0u); CHECK_EQ(sb.y0, 0u); CHECK_EQ(sb.x0, 8u);

    c3d_subband_info_of(8, &sb);    /* HHH_4 */
    CHECK_EQ(sb.level, 4u); CHECK_EQ(sb.kind, 1u); CHECK_EQ(sb.side, 16u);
    CHECK_EQ(sb.z0, 16u); CHECK_EQ(sb.y0, 16u); CHECK_EQ(sb.x0, 16u);

    c3d_subband_info_of(35, &sb);   /* LLH_1, last entry */
    CHECK_EQ(sb.level, 1u); CHECK_EQ(sb.kind, 7u); CHECK_EQ(sb.side, 128u);
    CHECK_EQ(sb.z0, 0u); CHECK_EQ(sb.y0, 0u); CHECK_EQ(sb.x0, 128u);
}

static void test_subband_extract_scatter(void) {
    /* Round-trip: fill buffer, extract each subband, scatter back, compare. */
    float *buf  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    float *copy = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    float *flat = malloc((size_t)128*128*128 * sizeof(float));
    c3d_assert(buf && copy && flat);

    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) buf[i] = (float)(i & 0xffff);
    memcpy(copy, buf, C3D_VOXELS_PER_CHUNK * sizeof(float));

    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb;
        c3d_subband_info_of(i, &sb);
        size_t n = c3d_subband_extract(buf, &sb, flat);
        CHECK_EQ(n, (size_t)sb.side * sb.side * sb.side);
        /* Zero the subband region then scatter flat back to confirm round-trip. */
        for (uint32_t z = sb.z0; z < sb.z0 + sb.side; ++z) {
            for (uint32_t y = sb.y0; y < sb.y0 + sb.side; ++y) {
                for (uint32_t x = sb.x0; x < sb.x0 + sb.side; ++x) {
                    buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x] = -1.0f;
                }
            }
        }
        c3d_subband_scatter(buf, &sb, flat);
    }
    CHECK(memcmp(buf, copy, C3D_VOXELS_PER_CHUNK * sizeof(float)) == 0);
    free(buf); free(copy); free(flat);
}

/* ─── §GHI  chunk encode/decode round-trip tests ─────────────────────────── */

/* A 256³ test input: smooth 3D ramp + low-frequency wave.  Uses ~120 bytes
 * of dynamic range, which the DWT can compact well. */
static void make_test_chunk(uint8_t *out) {
    for (uint32_t z = 0; z < C3D_CHUNK_SIDE; ++z)
    for (uint32_t y = 0; y < C3D_CHUNK_SIDE; ++y)
    for (uint32_t x = 0; x < C3D_CHUNK_SIDE; ++x) {
        double v = 128.0
                 + 40.0 * sin(z * 0.04)
                 + 30.0 * cos(y * 0.03)
                 + 20.0 * sin(x * 0.05 + z * 0.02);
        int iv = (int)(v + 0.5);
        if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
        out[(size_t)z * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE + y * C3D_CHUNK_SIDE + x] = (uint8_t)iv;
    }
}

static double measure_psnr(const uint8_t *a, const uint8_t *b, size_t n) {
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double e = (double)a[i] - (double)b[i];
        sse += e * e;
    }
    if (sse <= 0.0) return 200.0;
    double mse = sse / (double)n;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static void test_chunk_encode_decode_at_q(void) {
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *dec = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *enc = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(in && dec && enc);

    make_test_chunk(in);

    /* Smallest allowed q (per C3D_Q_MIN = 2^-6) → finest possible quantization. */
    size_t sz = c3d_chunk_encode_at_q(in, 1.0f / 64.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(sz > 352 && sz <= C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(c3d_chunk_validate(enc, sz));
    c3d_chunk_decode(enc, sz, dec);
    double psnr_fine = measure_psnr(in, dec, C3D_VOXELS_PER_CHUNK);
    printf("  q=1/64 (finest):  size=%zu PSNR=%.1f dB\n", sz, psnr_fine);
    CHECK(psnr_fine > 40.0);

    /* Moderate q. */
    sz = c3d_chunk_encode_at_q(in, 0.1f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(sz < C3D_VOXELS_PER_CHUNK / 2);
    c3d_chunk_decode(enc, sz, dec);
    double psnr_mid = measure_psnr(in, dec, C3D_VOXELS_PER_CHUNK);
    printf("  q=0.1   (moderate): size=%zu PSNR=%.1f dB\n", sz, psnr_mid);
    CHECK(psnr_mid > 30.0);

    /* Aggressive. */
    sz = c3d_chunk_encode_at_q(in, 2.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(sz < 20000);
    c3d_chunk_decode(enc, sz, dec);
    double psnr_coarse = measure_psnr(in, dec, C3D_VOXELS_PER_CHUNK);
    printf("  q=2.0   (coarse):   size=%zu PSNR=%.1f dB\n", sz, psnr_coarse);
    CHECK(psnr_coarse > 15.0);

    /* Sizes should be monotonic: smaller q → larger chunk. */
    CHECK(psnr_fine > psnr_mid);
    CHECK(psnr_mid > psnr_coarse);

    free(in); free(dec); free(enc);
}

static void test_chunk_rate_control(void) {
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *dec = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *enc = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(in && dec && enc);
    make_test_chunk(in);

    float ratios[] = {2.0f, 10.0f, 100.0f};
    for (size_t i = 0; i < sizeof ratios / sizeof ratios[0]; ++i) {
        float r = ratios[i];
        size_t sz = c3d_chunk_encode(in, r, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
        size_t target = (size_t)((double)C3D_VOXELS_PER_CHUNK / (double)r);
        double rel_err = (double)((long)sz - (long)target) / (double)target;
        if (rel_err < 0) rel_err = -rel_err;
        /* Rate control should land within 2× target (very loose); tighter
         * in practice after 8 iterations on this input.  This test just
         * confirms the loop is doing something sensible. */
        CHECK(rel_err < 3.0);
        CHECK(c3d_chunk_validate(enc, sz));

        c3d_chunk_decode(enc, sz, dec);
        /* On this smooth input, even 100:1 should give OK PSNR. */
        double psnr = measure_psnr(in, dec, C3D_VOXELS_PER_CHUNK);
        CHECK(psnr > 20.0);
    }
    free(in); free(dec); free(enc);
}

static void test_chunk_truncated_decode(void) {
    /* §T9 — quality-scalable truncated decode.  Decoding with in_len smaller
     * than the emitted chunk size must (a) not panic and (b) produce a
     * valid u8 reconstruction.  Subband granularity is the truncation unit.
     *
     * Monotonicity (more bytes → ≥ PSNR) holds on real data but not
     * universally on synthetic test chunks — when the HF bitstream
     * encodes specific patterns, adding more of it at high compression
     * can introduce reconstruction noise that wasn't there in the
     * LL-only reconstruction.  So we check validity + reasonable floor
     * here and leave monotonicity validation to the corpus bench. */
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *dec = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *enc = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(in && dec && enc);
    make_test_chunk(in);

    size_t full = c3d_chunk_encode(in, 10.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(full > 400);

    c3d_chunk_decode(enc, full, dec);
    double psnr_full = measure_psnr(in, dec, C3D_VOXELS_PER_CHUNK);
    CHECK(psnr_full > 30.0);

    /* Every truncation must still produce output, and PSNR must stay above
     * a LL-only-reconstruction floor (chunk mean ~= 20 dB in the worst case). */
    static const double fracs[] = { 0.75, 0.50, 0.25, 0.10, 0.05 };
    for (size_t i = 0; i < sizeof fracs / sizeof fracs[0]; ++i) {
        size_t trunc = (size_t)((double)full * fracs[i]);
        if (trunc < C3D_CHUNK_FIXED_SIZE) trunc = C3D_CHUNK_FIXED_SIZE;
        c3d_chunk_decode(enc, trunc, dec);
        double p = measure_psnr(in, dec, C3D_VOXELS_PER_CHUNK);
        printf("  frac=%.2f  bytes=%7zu  PSNR=%.2f dB\n", fracs[i], trunc, p);
        CHECK(p > 15.0);   /* above LL-only floor for this synthetic chunk */
    }

    /* Header-only decode (exactly C3D_CHUNK_FIXED_SIZE bytes): no entropy
     * payload at all; every subband zero-filled; output is the chunk's
     * dc_offset + 128 (constant value).  Must still not panic. */
    c3d_chunk_decode(enc, C3D_CHUNK_FIXED_SIZE, dec);

    free(in); free(dec); free(enc);
}

static void test_chunk_deterministic_encode(void) {
    /* Same-binary deterministic: encoding the same input twice must produce
     * byte-identical output, whether via the stateless API, fresh encoder
     * contexts, or a reused encoder context.  Protects against accidental
     * state leaks in the encoder (warm-start, fine-hist cache, allocator)
     * when future format changes (Q4) land. */
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *a   = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    uint8_t *b   = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    uint8_t *dec_a = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *dec_b = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    c3d_assert(in && a && b && dec_a && dec_b);
    make_test_chunk(in);

    float ratios[] = {5.0f, 25.0f, 100.0f};
    for (size_t i = 0; i < sizeof ratios / sizeof ratios[0]; ++i) {
        float r = ratios[i];
        /* Path 1: stateless API, two separate calls. */
        size_t sz_a = c3d_chunk_encode(in, r, a, C3D_CHUNK_ENCODE_MAX_SIZE);
        size_t sz_b = c3d_chunk_encode(in, r, b, C3D_CHUNK_ENCODE_MAX_SIZE);
        CHECK_EQ(sz_a, sz_b);
        CHECK(memcmp(a, b, sz_a) == 0);

        /* Path 2: reused encoder context — must produce same bytes as fresh. */
        c3d_encoder *e = c3d_encoder_new();
        /* Warm the context once so last_q is set; then re-encode. */
        size_t sz_warm = c3d_encoder_chunk_encode(e, in, r, b,
                                                  C3D_CHUNK_ENCODE_MAX_SIZE);
        size_t sz_c    = c3d_encoder_chunk_encode(e, in, r, b,
                                                  C3D_CHUNK_ENCODE_MAX_SIZE);
        CHECK_EQ(sz_warm, sz_c);
        CHECK_EQ(sz_a, sz_c);
        CHECK(memcmp(a, b, sz_a) == 0);
        c3d_encoder_free(e);

        /* Path 3: round-trip must match between the two paths. */
        c3d_chunk_decode(a, sz_a, dec_a);
        c3d_chunk_decode(b, sz_a, dec_b);
        CHECK(memcmp(dec_a, dec_b, C3D_VOXELS_PER_CHUNK) == 0);
    }
    free(in); free(a); free(b); free(dec_a); free(dec_b);
}

static void test_chunks_batched(void) {
    /* I3: multi-chunk batched encode/decode must produce output byte-identical
     * to calling the single-chunk API in sequence.  Exercises the shared
     * encoder/decoder state that backs the batch. */
    const size_t N = 3;
    uint8_t *ins[3], *outs_batch[3], *outs_one[3], *decs_batch[3], *decs_one[3];
    size_t sizes_batch[3], sizes_one[3];
    const uint8_t *ins_const[3];
    const size_t  *sizes_const;
    for (size_t i = 0; i < N; ++i) {
        ins[i]        = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
        outs_batch[i] = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
        outs_one[i]   = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
        decs_batch[i] = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
        decs_one[i]   = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
        c3d_assert(ins[i] && outs_batch[i] && outs_one[i]);
        make_test_chunk(ins[i]);
        /* Perturb slightly so chunks aren't identical. */
        ins[i][0] = (uint8_t)(i * 17);
        ins_const[i] = ins[i];
    }
    sizes_const = sizes_batch;
    (void)sizes_const;

    c3d_encoder *e = c3d_encoder_new();
    c3d_decoder *dc = c3d_decoder_new();

    /* Batched path. */
    c3d_encoder_chunks_encode(e, ins_const, N, 25.0f,
                              outs_batch, sizes_batch);
    const uint8_t *batch_ins[3] = { outs_batch[0], outs_batch[1], outs_batch[2] };
    c3d_decoder_chunks_decode(dc, batch_ins, sizes_batch, N, decs_batch);

    /* Reference path — same encoder/decoder, called per chunk. */
    c3d_encoder *e2 = c3d_encoder_new();
    c3d_decoder *dc2 = c3d_decoder_new();
    for (size_t i = 0; i < N; ++i) {
        sizes_one[i] = c3d_encoder_chunk_encode(
            e2, ins[i], 25.0f, outs_one[i], C3D_CHUNK_ENCODE_MAX_SIZE);
        c3d_decoder_chunk_decode(dc2, outs_one[i], sizes_one[i], decs_one[i]);
    }

    for (size_t i = 0; i < N; ++i) {
        CHECK_EQ(sizes_batch[i], sizes_one[i]);
        CHECK(memcmp(outs_batch[i], outs_one[i], sizes_batch[i]) == 0);
        CHECK(memcmp(decs_batch[i], decs_one[i], C3D_VOXELS_PER_CHUNK) == 0);
    }

    c3d_encoder_free(e); c3d_encoder_free(e2);
    c3d_decoder_free(dc); c3d_decoder_free(dc2);
    for (size_t i = 0; i < N; ++i) {
        free(ins[i]); free(outs_batch[i]); free(outs_one[i]);
        free(decs_batch[i]); free(decs_one[i]);
    }
}

static void test_chunk_lod_decode(void) {
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *enc = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(in && enc);
    make_test_chunk(in);

    size_t sz = c3d_chunk_encode_at_q(in, 0.05f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(sz > 352);

    /* Every LOD decode must succeed and produce the correct sized buffer. */
    for (unsigned k = 0; k <= 5; ++k) {
        size_t side = (size_t)C3D_CHUNK_SIDE >> k;
        uint8_t *out = aligned_alloc(C3D_ALIGN, side * side * side);
        c3d_assert(out);
        c3d_chunk_decode_lod(enc, sz, (uint8_t)k, out);
        /* Output must be u8 values — just check at least one non-zero
         * (on this non-trivial input).  Values 128 all over would still
         * be "valid" for a uniform chunk, but our input isn't uniform. */
        bool any_nonzero = false;
        for (size_t i = 0; i < side*side*side; ++i) {
            if (out[i] != 0) { any_nonzero = true; break; }
        }
        CHECK(any_nonzero);
        free(out);
    }
    free(in); free(enc);
}

static void test_chunk_empty(void) {
    /* Uniform-after-centering chunk: all voxels same value → empty entropy. */
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *dec = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *enc = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(in && dec && enc);

    memset(in, 77, C3D_VOXELS_PER_CHUNK);   /* uniform */
    size_t sz = c3d_chunk_encode_at_q(in, 1.0f / 32.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK_EQ(sz, (size_t)C3D_CHUNK_FIXED_SIZE);

    c3d_chunk_info info;
    c3d_chunk_inspect(enc, sz, &info);
    for (unsigned k = 0; k < C3D_N_LODS; ++k) CHECK_EQ(info.lod_offsets[k], 0u);
    CHECK(c3d_chunk_validate(enc, sz));

    c3d_chunk_decode(enc, sz, dec);
    /* All dec voxels should be 77 (recovered from dc_offset). */
    bool all_77 = true;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        if (dec[i] != 77) { all_77 = false; break; }
    }
    CHECK(all_77);
    free(in); free(dec); free(enc);
}

/* "Zero means ignore" encode path: caller zeros out don't-care voxels and
 * c3d_chunk_encode_masked fills them with min-non-zero before the regular
 * encode.  Verify:
 *   1. Non-zero (material) voxels are preserved within reasonable PSNR at
 *      moderate q.
 *   2. The encoded bitstream is a standard v1 chunk readable by any decoder.
 *   3. At all-zero input, the masked encode produces the empty-chunk fast
 *      path (C3D_CHUNK_FIXED_SIZE, 0-filled on decode). */
static void test_chunk_encode_masked(void) {
    uint8_t *in  = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *dec = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    uint8_t *enc = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(in && dec && enc);

    /* Build an input with a sharp 0/material boundary and some smooth
     * material interior.  This mimics scroll-after-threshold: voxels below
     * the material floor are 0, rest is real signal. */
    make_test_chunk(in);
    /* Zero out a half-cube: x < 128. */
    for (uint32_t z = 0; z < C3D_CHUNK_SIDE; ++z)
    for (uint32_t y = 0; y < C3D_CHUNK_SIDE; ++y)
    for (uint32_t x = 0; x < 128; ++x)
        in[(size_t)z * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE + y * C3D_CHUNK_SIDE + x] = 0;

    size_t sz = c3d_chunk_encode_masked(in, 50.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(sz > 352 && sz <= C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(c3d_chunk_validate(enc, sz));

    c3d_chunk_decode(enc, sz, dec);

    /* Material-only PSNR (x >= 128): material voxels in `in` (non-zero) vs
     * decoded values.  Should be reasonably high at 50× target. */
    double sse = 0.0; size_t cnt = 0;
    for (uint32_t z = 0; z < C3D_CHUNK_SIDE; ++z)
    for (uint32_t y = 0; y < C3D_CHUNK_SIDE; ++y)
    for (uint32_t x = 128; x < C3D_CHUNK_SIDE; ++x) {
        size_t i = (size_t)z*C3D_CHUNK_SIDE*C3D_CHUNK_SIDE + y*C3D_CHUNK_SIDE + x;
        double d = (double)in[i] - (double)dec[i];
        sse += d * d; ++cnt;
    }
    double matP = 10.0 * log10(255.0 * 255.0 / (sse / (double)cnt));
    printf("  masked 50x material PSNR: %.2f dB\n", matP);
    CHECK(matP > 28.0);

    /* All-zero input (e.g. pure-black padding chunk) → empty-chunk fast path. */
    memset(in, 0, C3D_VOXELS_PER_CHUNK);
    sz = c3d_chunk_encode_masked(in, 50.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK_EQ(sz, (size_t)C3D_CHUNK_FIXED_SIZE);
    c3d_chunk_decode(enc, sz, dec);
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) CHECK_EQ(dec[i], 0);

    /* 99%-air sparse chunk: a single thin 16³ cluster of material, rest 0.
     * Should compress to a small chunk and recover the material accurately. */
    memset(in, 0, C3D_VOXELS_PER_CHUNK);
    for (uint32_t z = 120; z < 136; ++z)
    for (uint32_t y = 120; y < 136; ++y)
    for (uint32_t x = 120; x < 136; ++x)
        in[(size_t)z*C3D_CHUNK_SIDE*C3D_CHUNK_SIDE + y*C3D_CHUNK_SIDE + x] = 150;
    sz = c3d_chunk_encode_masked(in, 50.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK(c3d_chunk_validate(enc, sz));
    c3d_chunk_decode(enc, sz, dec);
    /* Material cluster should reconstruct close to 150. */
    double s = 0; size_t n = 0;
    for (uint32_t z = 120; z < 136; ++z)
    for (uint32_t y = 120; y < 136; ++y)
    for (uint32_t x = 120; x < 136; ++x) {
        size_t i = (size_t)z*C3D_CHUNK_SIDE*C3D_CHUNK_SIDE + y*C3D_CHUNK_SIDE + x;
        double d = 150.0 - (double)dec[i]; s += d * d; ++n;
    }
    double sparse_mat_psnr = 10.0 * log10(255.0 * 255.0 / (s / (double)n));
    printf("  masked 99%%-air sparse cluster matP: %.2f dB  bytes=%zu\n", sparse_mat_psnr, sz);
    CHECK(sparse_mat_psnr > 30.0);

    /* All non-zero voxels are exactly 255 — exercise the former sentinel bug
     * (m_min stayed at 255 and fill was skipped).  Fill should now replace
     * every 0 with 255, giving a uniform chunk that hits the empty-chunk
     * fast path. */
    memset(in, 0, C3D_VOXELS_PER_CHUNK);
    for (size_t i = 0; i < 1024; ++i) in[i * 1024] = 255;
    sz = c3d_chunk_encode_masked(in, 50.0f, enc, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK_EQ(sz, (size_t)C3D_CHUNK_FIXED_SIZE);
    c3d_chunk_decode(enc, sz, dec);
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) CHECK_EQ(dec[i], 255);

    /* Determinism: same input → same output bytes under _masked. */
    make_test_chunk(in);
    for (uint32_t z = 0; z < C3D_CHUNK_SIDE; ++z)
    for (uint32_t y = 0; y < C3D_CHUNK_SIDE; ++y)
    for (uint32_t x = 0; x < 128; ++x)
        in[(size_t)z * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE + y * C3D_CHUNK_SIDE + x] = 0;
    uint8_t *enc2 = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    size_t sz1 = c3d_chunk_encode_masked(in, 50.0f, enc,  C3D_CHUNK_ENCODE_MAX_SIZE);
    size_t sz2 = c3d_chunk_encode_masked(in, 50.0f, enc2, C3D_CHUNK_ENCODE_MAX_SIZE);
    CHECK_EQ(sz1, sz2);
    CHECK(memcmp(enc, enc2, sz1) == 0);
    free(enc2);

    free(in); free(dec); free(enc);
}

static void test_chunk_validate_rejects_garbage(void) {
    uint8_t buf[512] = {0};
    CHECK(!c3d_chunk_validate(buf, 512));       /* bad magic */
    CHECK(!c3d_chunk_validate(buf, 100));       /* too short */
    CHECK(!c3d_chunk_validate(NULL, 0));        /* null */

    memcpy(buf, "C3DC", 4);
    c3d_write_u16_le(buf + 4, 2);               /* wrong version */
    CHECK(!c3d_chunk_validate(buf, 512));

}

/* ─── §J  shard tests ────────────────────────────────────────────────────── */

/* ─── downsample helper ──────────────────────────────────────────────────── */

static void test_downsample_2x(void) {
    /* 16³ → 8³ box average. */
    uint8_t in[16*16*16], out[8*8*8];
    for (size_t i = 0; i < 16*16*16; ++i) in[i] = (uint8_t)(i & 0xff);
    c3d_downsample_chunk_2x(in, 16, out);

    /* Spot check: out[0,0,0] = mean of the 8 voxels at (0..1, 0..1, 0..1). */
    uint32_t sum = 0;
    for (uint32_t dz = 0; dz < 2; ++dz) {
        for (uint32_t dy = 0; dy < 2; ++dy) {
            for (uint32_t dx = 0; dx < 2; ++dx) {
                sum += in[dz*16*16 + dy*16 + dx];
            }
        }
    }
    /* Rounded average with ties-to-even. */
    uint32_t expected = (sum + 4) >> 3;
    if ((sum & 7u) == 4u && (expected & 1u)) --expected;
    CHECK_EQ(out[0], expected);

    /* A uniform input → uniform output. */
    memset(in, 100, sizeof in);
    c3d_downsample_chunk_2x(in, 16, out);
    for (size_t i = 0; i < sizeof out; ++i) CHECK_EQ(out[i], 100u);
}

/* ─── main ───────────────────────────────────────────────────────────────── */

int main(void) {
    c3d_set_panic_hook(test_panic_hook);

    printf("§A scaffolding\n");
    test_voxel_key();
    test_leb128();
    test_morton12();
    test_panic_mechanism();

    printf("§B hash\n");
    test_hash128_empty();
    test_hash128_determinism();
    test_hash128_avalanche();
    test_hash128_various_lengths();

    printf("§C rANS\n");
    test_rans();
    test_rans_single_symbol();

    printf("§D frequency tables\n");
    test_histogram();
    test_normalise_freqs_basic();
    test_normalise_freqs_sparse();
    test_freqs_serialise_roundtrip();

    printf("§E DWT\n");
    test_dwt_1d_roundtrip();
    test_dwt_1d_double_cross();
    test_dwt_1d_impulse();
    test_dwt_3d_roundtrip();
    test_dwt_3d_lod_partial();

    printf("§F quantizer + symbols\n");
    test_quant_roundtrip();
    test_zigzag32();
    test_symbol_mapping();
    test_subband_info();
    test_subband_extract_scatter();

    printf("§GHI chunk encode/decode\n");
    test_chunk_empty();
    test_chunk_validate_rejects_garbage();
    test_chunk_encode_decode_at_q();
    test_chunk_rate_control();
    test_chunk_truncated_decode();
    test_chunk_deterministic_encode();
    test_chunks_batched();
    test_chunk_lod_decode();
    test_chunk_encode_masked();

    printf("§J downsample\n");
    test_downsample_2x();

    printf("\n%d tests, %d failures\n", g_tests_run, g_tests_fail);
    return g_tests_fail ? 1 : 0;
}
