/* c3d — 3D volumetric u8 compression codec.  See LICENSE, PLAN.md, CLAUDE.md.
 *
 * This TU is organised into sections:
 *     §A  scaffolding (panic/assert, bit-io, alignment, Morton-12)
 *     §B  c3d_hash128 (MurmurHash3_x64_128)
 *     §C  rANS engine (scalar + 8-way interleaved)
 *     §D..§L  not yet implemented (see PLAN.md §6)
 */

#include "c3d.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Endianness gate — LE only. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#  error "c3d requires a little-endian target"
#endif

/* ========================================================================= *
 *  §A  Scaffolding                                                          *
 * ========================================================================= */

static c3d_panic_fn s_panic_hook = NULL;

void c3d_set_panic_hook(c3d_panic_fn hook) {
    s_panic_hook = hook;
}

_Noreturn void c3d_panic(const char *file, int line, const char *msg) {
    if (s_panic_hook) {
        s_panic_hook(file, line, msg);
        /* Hook must not return; if it does, fall through to abort. */
    }
    fprintf(stderr, "c3d_panic: %s:%d: %s\n", file, line, msg ? msg : "(null)");
    fflush(stderr);
    abort();
}

/* ----- bit-io / integer read-write (memcpy-based for unaligned safety) ---- */

static inline uint16_t c3d_read_u16_le(const uint8_t *p) {
    uint16_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint32_t c3d_read_u32_le(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, sizeof v); return v;
}
static inline uint64_t c3d_read_u64_le(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, sizeof v); return v;
}
static inline float c3d_read_f32_le(const uint8_t *p) {
    float v; memcpy(&v, p, sizeof v); return v;
}
static inline void c3d_write_u16_le(uint8_t *p, uint16_t v) { memcpy(p, &v, sizeof v); }
static inline void c3d_write_u32_le(uint8_t *p, uint32_t v) { memcpy(p, &v, sizeof v); }
static inline void c3d_write_u64_le(uint8_t *p, uint64_t v) { memcpy(p, &v, sizeof v); }
static inline void c3d_write_f32_le(uint8_t *p, float v)    { memcpy(p, &v, sizeof v); }

/* ----- LEB128 unsigned varints (7-bit groups, high bit = continuation) ----
 * Caller-provided bounds to detect overrun without a separate state struct. */

static size_t c3d_leb128_encode(uint64_t v, uint8_t *out, size_t out_cap) {
    size_t n = 0;
    do {
        c3d_assert(n < out_cap);
        uint8_t b = (uint8_t)(v & 0x7f);
        v >>= 7;
        if (v) b |= 0x80;
        out[n++] = b;
    } while (v);
    return n;
}

/* Decode one LEB128 varint from `in[0..in_len)`; returns consumed bytes,
 * writes value to *out.  Panics on truncation or >10-byte encoding. */
static size_t c3d_leb128_decode(const uint8_t *in, size_t in_len, uint64_t *out) {
    uint64_t v = 0;
    unsigned shift = 0;
    size_t n = 0;
    for (;;) {
        c3d_assert(n < in_len);
        c3d_assert(shift < 64);
        uint8_t b = in[n++];
        v |= (uint64_t)(b & 0x7f) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    *out = v;
    return n;
}

/* ----- alignment ---------------------------------------------------------- */

static inline void c3d_check_voxel_alignment(const void *p) {
    c3d_assert(((uintptr_t)p & (C3D_ALIGN - 1)) == 0);
}

/* ----- 12-bit Morton (shard chunk-index ordering) ------------------------- *
 * Chunk coords are 4 bits each; Morton interleaves as z3 y3 x3 ... z0 y0 x0. */

static inline uint32_t c3d_morton12(uint32_t cx, uint32_t cy, uint32_t cz) {
    c3d_assert(cx < 16 && cy < 16 && cz < 16);
    /* Spread 4 bits of each to every third bit, then combine. */
    static const uint32_t spread4[16] = {
        0x000, 0x001, 0x008, 0x009, 0x040, 0x041, 0x048, 0x049,
        0x200, 0x201, 0x208, 0x209, 0x240, 0x241, 0x248, 0x249,
    };
    return spread4[cx] | (spread4[cy] << 1) | (spread4[cz] << 2);
}

static inline void c3d_morton12_decode(uint32_t m,
                                       uint32_t *cx, uint32_t *cy, uint32_t *cz) {
    /* Compact every third bit back to a nibble. */
    uint32_t x = 0, y = 0, z = 0;
    for (unsigned i = 0; i < 4; ++i) {
        x |= ((m >> (3*i + 0)) & 1u) << i;
        y |= ((m >> (3*i + 1)) & 1u) << i;
        z |= ((m >> (3*i + 2)) & 1u) << i;
    }
    *cx = x; *cy = y; *cz = z;
}

/* ========================================================================= *
 *  §B  c3d_hash128  (MurmurHash3_x64_128)                                   *
 * ========================================================================= */

/* MurmurHash3_x64_128 by Austin Appleby (public domain).  Rewritten here for
 * 64-byte-per-block accumulation, adapted for little-endian direct reads.
 * Not cryptographically secure; intended only for content addressing. */

static inline uint64_t c3d_rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}
static inline uint64_t c3d_fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

void c3d_hash128(const void *data, size_t len, uint8_t out[16]) {
    const uint8_t *p = (const uint8_t *)data;
    const size_t   nblocks = len / 16;

    uint64_t h1 = 0;
    uint64_t h2 = 0;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    for (size_t i = 0; i < nblocks; ++i) {
        uint64_t k1 = c3d_read_u64_le(p + 16*i + 0);
        uint64_t k2 = c3d_read_u64_le(p + 16*i + 8);

        k1 *= c1; k1 = c3d_rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = c3d_rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

        k2 *= c2; k2 = c3d_rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = c3d_rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t *tail = p + nblocks * 16;
    uint64_t k1 = 0, k2 = 0;
    switch (len & 15) {
    case 15: k2 ^= (uint64_t)tail[14] << 48;  /* fallthrough */
    case 14: k2 ^= (uint64_t)tail[13] << 40;  /* fallthrough */
    case 13: k2 ^= (uint64_t)tail[12] << 32;  /* fallthrough */
    case 12: k2 ^= (uint64_t)tail[11] << 24;  /* fallthrough */
    case 11: k2 ^= (uint64_t)tail[10] << 16;  /* fallthrough */
    case 10: k2 ^= (uint64_t)tail[ 9] <<  8;  /* fallthrough */
    case  9: k2 ^= (uint64_t)tail[ 8] <<  0;
             k2 *= c2; k2 = c3d_rotl64(k2, 33); k2 *= c1; h2 ^= k2;
             /* fallthrough */
    case  8: k1 ^= (uint64_t)tail[ 7] << 56;  /* fallthrough */
    case  7: k1 ^= (uint64_t)tail[ 6] << 48;  /* fallthrough */
    case  6: k1 ^= (uint64_t)tail[ 5] << 40;  /* fallthrough */
    case  5: k1 ^= (uint64_t)tail[ 4] << 32;  /* fallthrough */
    case  4: k1 ^= (uint64_t)tail[ 3] << 24;  /* fallthrough */
    case  3: k1 ^= (uint64_t)tail[ 2] << 16;  /* fallthrough */
    case  2: k1 ^= (uint64_t)tail[ 1] <<  8;  /* fallthrough */
    case  1: k1 ^= (uint64_t)tail[ 0] <<  0;
             k1 *= c1; k1 = c3d_rotl64(k1, 31); k1 *= c2; h1 ^= k1;
             /* fallthrough */
    case  0: break;
    }

    h1 ^= (uint64_t)len;
    h2 ^= (uint64_t)len;
    h1 += h2;
    h2 += h1;
    h1 = c3d_fmix64(h1);
    h2 = c3d_fmix64(h2);
    h1 += h2;
    h2 += h1;

    c3d_write_u64_le(out + 0, h1);
    c3d_write_u64_le(out + 8, h2);
}

/* ========================================================================= *
 *  §C  rANS engine (ryg_rans_byte-style)                                    *
 * ========================================================================= *
 *
 * 32-bit state, byte-at-a-time renormalisation, lower bound RANS_BYTE_L.
 * Encoder writes bytes backward (from buf end toward buf start); decoder
 * reads bytes forward from where the encoder finished.
 *
 * 8-way interleaved variant: 8 independent states encoding symbols dealt
 * round-robin.  Wire format per PLAN §3.4:
 *   rans_header    = 32 B = 8 × u32 final states (little-endian)
 *   rans_renorm    = the forward-read renorm byte stream
 * The decoder loads the 8 states from the header, then consumes renorm
 * bytes on demand as each state's value drops below RANS_BYTE_L.
 */

#define C3D_RANS_BYTE_L    (1u << 23)   /* lower bound of normal state range */
#define C3D_RANS_N_STATES  8u           /* interleaving factor               */

typedef struct {
    uint32_t start;   /* cumulative probability of symbol */
    uint32_t freq;    /* probability of symbol            */
} c3d_rans_sym;

typedef struct {
    uint32_t cum2sym[1u << 14];  /* map cumulative prob → symbol (max M=16384) */
    c3d_rans_sym syms[65];        /* 65 symbols in the c3d alphabet            */
    uint32_t denom_shift;         /* log2(M); M = 1<<denom_shift               */
} c3d_rans_tables;

/* Build fast tables from a (symbol,freq) list that sums to M = 1<<denom_shift.
 * Panics if frequencies don't sum to M. */
static void c3d_rans_build_tables(c3d_rans_tables *t,
                                  uint32_t denom_shift,
                                  const uint32_t *freqs,      /* freqs[0..n_symbols) */
                                  size_t n_symbols)
{
    c3d_assert(denom_shift <= 14);
    const uint32_t M = 1u << denom_shift;
    t->denom_shift = denom_shift;

    uint32_t cum = 0;
    for (size_t s = 0; s < n_symbols; ++s) {
        t->syms[s].start = cum;
        t->syms[s].freq  = freqs[s];
        for (uint32_t i = 0; i < freqs[s]; ++i) {
            t->cum2sym[cum + i] = (uint32_t)s;
        }
        cum += freqs[s];
    }
    c3d_assert(cum == M);
    /* Fill unused alphabet entries (freq=0) with defined start/freq. */
    for (size_t s = n_symbols; s < 65; ++s) {
        t->syms[s].start = cum;
        t->syms[s].freq  = 0;
    }
}

/* ----- scalar rANS ------------------------------------------------------- */

/* Encoder state; bytes are written at *out_p, stepping backward.
 * Initial state = RANS_BYTE_L.  Call rans_flush at end to dump final state. */

static inline void c3d_rans_enc_init(uint32_t *state) {
    *state = C3D_RANS_BYTE_L;
}

static inline void c3d_rans_enc_put(uint32_t *state,
                                    uint8_t **out_p,
                                    const uint8_t *out_begin,
                                    uint32_t start, uint32_t freq,
                                    uint32_t denom_shift)
{
    /* Renormalise: while state * M / freq >= 2^32 — equivalently, while
     * state >= freq * (RANS_BYTE_L >> (denom_shift)) << 8. */
    uint32_t x = *state;
    uint32_t x_max = ((C3D_RANS_BYTE_L >> denom_shift) << 8) * freq;
    while (x >= x_max) {
        c3d_assert(*out_p > out_begin);
        *--(*out_p) = (uint8_t)(x & 0xff);
        x >>= 8;
    }
    /* Mix in the symbol. */
    *state = ((x / freq) << denom_shift) + (x % freq) + start;
}

static inline void c3d_rans_enc_flush(uint32_t state,
                                      uint8_t **out_p, const uint8_t *out_begin)
{
    c3d_assert(*out_p - out_begin >= 4);
    *--(*out_p) = (uint8_t)((state >> 24) & 0xff);
    *--(*out_p) = (uint8_t)((state >> 16) & 0xff);
    *--(*out_p) = (uint8_t)((state >>  8) & 0xff);
    *--(*out_p) = (uint8_t)((state >>  0) & 0xff);
}

static inline void c3d_rans_dec_init(uint32_t *state,
                                     const uint8_t **in_p, const uint8_t *in_end)
{
    c3d_assert(in_end - *in_p >= 4);
    uint32_t x = (*in_p)[0]
              | ((uint32_t)(*in_p)[1] <<  8)
              | ((uint32_t)(*in_p)[2] << 16)
              | ((uint32_t)(*in_p)[3] << 24);
    *in_p += 4;
    *state = x;
}

/* Decode one symbol using the cum2sym lookup. */
static inline uint32_t c3d_rans_dec_get(uint32_t *state,
                                        const c3d_rans_tables *t)
{
    const uint32_t M_mask = (1u << t->denom_shift) - 1u;
    uint32_t slot = *state & M_mask;
    uint32_t sym  = t->cum2sym[slot];
    /* advance state */
    *state = t->syms[sym].freq * (*state >> t->denom_shift)
           + slot - t->syms[sym].start;
    return sym;
}

/* After decoding a symbol, the decoder may need to consume renorm bytes. */
static inline void c3d_rans_dec_renorm(uint32_t *state,
                                       const uint8_t **in_p, const uint8_t *in_end)
{
    while (*state < C3D_RANS_BYTE_L) {
        c3d_assert(*in_p < in_end);
        *state = (*state << 8) | **in_p;
        ++*in_p;
    }
}

/* ----- 8-way interleaved rANS -------------------------------------------- *
 *
 * Encoder:  8 independent states, symbols dispatched round-robin by index.
 * Output layout in the per-subband bitstream:
 *     [ 8 × u32 final states (32 B) ][ renorm bytes, forward-read ]
 *
 * During encoding we don't know where the renorm byte stream will end up
 * relative to the 32 B state header until the encode completes, because bytes
 * are written backward.  We allocate a working buffer, run the encoder
 * writing backward from the end, then:
 *     1. Copy the resulting renorm bytes into the final output in FORWARD
 *        order (because the decoder reads forward).
 *     2. Write the 8 final states as 32 B little-endian u32s at the
 *        start.
 *
 * So the encoder needs O(max_output_size) scratch memory per subband. */

typedef struct {
    /* Scratch layout used during encode: bytes are written at scratch+scratch_end
     * growing downward to scratch+scratch_head. */
    uint8_t *scratch;
    size_t   scratch_size;
} c3d_rans_enc_scratch;

/* Encode `symbols[0..n_symbols)` interleaved over 8 states.
 * `symbol_of` is a function that maps an alphabet index to (start, freq).
 * Output format: out[0..32) = 8 u32 states; out[32..out_len) = renorm bytes.
 * Returns the total number of bytes written to out; panics on out_cap exceed. */
static size_t c3d_rans_enc_x8(const uint8_t *symbols,     /* each < 65 */
                              size_t n_symbols,
                              const c3d_rans_tables *t,
                              uint8_t *scratch, size_t scratch_size,
                              uint8_t *out, size_t out_cap)
{
    /* We need scratch_size large enough to hold all renorm bytes.  Worst case:
     * each symbol triggers up to 2 renorm bytes (rare at denom_shift ≤ 14 with
     * start state = RANS_BYTE_L).  We demand scratch_size ≥ n_symbols * 2 + 32. */
    c3d_assert(scratch_size >= n_symbols * 2u + 32u);
    (void)scratch_size;

    uint8_t       *buf_end = scratch + scratch_size;
    uint8_t       *buf_ptr = buf_end;
    const uint8_t *buf_beg = scratch;

    uint32_t states[C3D_RANS_N_STATES];
    for (unsigned i = 0; i < C3D_RANS_N_STATES; ++i) {
        c3d_rans_enc_init(&states[i]);
    }

    /* Encode from back to front so that decode-forward reads renorm bytes in
     * the right order.  Each symbol i is assigned to stream (i % 8).
     * We reverse that: iterate i from n_symbols-1 down to 0.
     *
     * Fast path for symbol 0: hoist its (freq, start=0, x_max) so the common
     * case avoids the t->syms[s] dependent load that feeds the renorm check. */
    const uint32_t ds       = t->denom_shift;
    const uint32_t freq0    = t->syms[0].freq;
    c3d_assert(t->syms[0].start == 0);
    const uint32_t x_max_0  = freq0 ? ((C3D_RANS_BYTE_L >> ds) << 8) * freq0
                                    : UINT32_MAX;
    for (size_t i = n_symbols; i > 0; --i) {
        size_t idx = i - 1;
        unsigned lane = (unsigned)(idx & (C3D_RANS_N_STATES - 1));
        uint8_t  s    = symbols[idx];
        if (__builtin_expect(s == 0, 1)) {
            uint32_t x = states[lane];
            while (x >= x_max_0) {
                c3d_assert(buf_ptr > buf_beg);
                *--buf_ptr = (uint8_t)(x & 0xff);
                x >>= 8;
            }
            states[lane] = ((x / freq0) << ds) + (x % freq0);
        } else {
            const c3d_rans_sym *sym = &t->syms[s];
            c3d_assert(sym->freq > 0);
            c3d_rans_enc_put(&states[lane], &buf_ptr, buf_beg,
                             sym->start, sym->freq, ds);
        }
    }

    size_t renorm_bytes = (size_t)(buf_end - buf_ptr);
    size_t total = 32u + renorm_bytes;
    c3d_assert(total <= out_cap);

    /* Write the 8 final states (u32 LE each) to out[0..32). */
    for (unsigned i = 0; i < C3D_RANS_N_STATES; ++i) {
        c3d_write_u32_le(out + 4u * i, states[i]);
    }
    /* Copy renorm bytes; in our scratch they run [buf_ptr .. buf_end), and the
     * decoder reads them forward starting at out+32. */
    memcpy(out + 32, buf_ptr, renorm_bytes);
    return total;
}

/* Decode n_symbols from the packed rans block at `in[0..in_len)`; writes
 * symbols[0..n_symbols).  Panics on truncation.
 *
 * 8 lanes are unrolled into independent local state variables so the compiler
 * can schedule their dec_get / renorm operations as parallel ILP.  This is
 * pure C — no intrinsics — and gets a meaningful speedup on out-of-order
 * cores by exposing inter-lane independence to the scheduler. */
static void c3d_rans_dec_x8(const uint8_t *in, size_t in_len,
                            const c3d_rans_tables *t,
                            uint8_t *symbols, size_t n_symbols)
{
    c3d_assert(in_len >= 32);
    uint32_t s0 = c3d_read_u32_le(in +  0);
    uint32_t s1 = c3d_read_u32_le(in +  4);
    uint32_t s2 = c3d_read_u32_le(in +  8);
    uint32_t s3 = c3d_read_u32_le(in + 12);
    uint32_t s4 = c3d_read_u32_le(in + 16);
    uint32_t s5 = c3d_read_u32_le(in + 20);
    uint32_t s6 = c3d_read_u32_le(in + 24);
    uint32_t s7 = c3d_read_u32_le(in + 28);
    const uint8_t *r   = in + 32;
    const uint8_t *r_e = in + in_len;
    const uint32_t M_mask  = (1u << t->denom_shift) - 1u;
    const uint32_t ds      = t->denom_shift;
    const uint32_t *cum2sym = t->cum2sym;
    const c3d_rans_sym *syms = t->syms;
    /* Symbol 0 occupies cumulative slots [0, freq0) — typically 70-90 % of M
     * in quantized wavelet subbands.  Hoist it so the hot branch avoids the
     * cum2sym and syms[] dependent loads. */
    const uint32_t freq0 = syms[0].freq;

    /* dec_get inlined: sym = cum2sym[state & M], state = freq*(state>>ds) + (state&M) - start.
     * Fast path: if slot < freq0 then sym=0, start=0, freq=freq0 (all loads avoidable). */
    #define DEC_LANE(SREF, OUT) do {                                          \
        uint32_t st = (SREF);                                                 \
        uint32_t slot = st & M_mask;                                          \
        if (__builtin_expect(slot < freq0, 1)) {                              \
            (OUT) = 0;                                                        \
            (SREF) = freq0 * (st >> ds) + slot;                               \
        } else {                                                              \
            uint32_t sym = cum2sym[slot];                                     \
            c3d_assert(sym < 65);                                             \
            (OUT) = (uint8_t)sym;                                             \
            (SREF) = syms[sym].freq * (st >> ds) + slot - syms[sym].start;    \
        }                                                                     \
    } while (0)
    #define RENORM(SREF) do {                                                 \
        while ((SREF) < C3D_RANS_BYTE_L) {                                    \
            c3d_assert(r < r_e);                                              \
            __builtin_prefetch(r + 64, 0, 0);                                 \
            (SREF) = ((SREF) << 8) | *r++;                                    \
        }                                                                     \
    } while (0)

    /* Process 8 lanes per iteration: lane k handles symbols i where i % 8 == k. */
    size_t full = n_symbols & ~(size_t)7u;
    for (size_t i = 0; i < full; i += 8) {
        DEC_LANE(s0, symbols[i + 0]);  RENORM(s0);
        DEC_LANE(s1, symbols[i + 1]);  RENORM(s1);
        DEC_LANE(s2, symbols[i + 2]);  RENORM(s2);
        DEC_LANE(s3, symbols[i + 3]);  RENORM(s3);
        DEC_LANE(s4, symbols[i + 4]);  RENORM(s4);
        DEC_LANE(s5, symbols[i + 5]);  RENORM(s5);
        DEC_LANE(s6, symbols[i + 6]);  RENORM(s6);
        DEC_LANE(s7, symbols[i + 7]);  RENORM(s7);
    }
    /* Tail: at most 7 symbols left.  Walk lane 0..min(7, remaining-1). */
    size_t left = n_symbols - full;
    uint32_t *tail_states[8] = {&s0,&s1,&s2,&s3,&s4,&s5,&s6,&s7};
    for (size_t k = 0; k < left; ++k) {
        DEC_LANE(*tail_states[k], symbols[full + k]);
        RENORM(*tail_states[k]);
    }
    #undef DEC_LANE
    #undef RENORM

    c3d_assert(r == r_e);
}

/* ========================================================================= *
 *  §D  Per-subband frequency tables (build + serialise + parse)             *
 * ========================================================================= *
 *
 * Wire format (PLAN §3.4):
 *     u8  denom_shift        log2(M); M is the cumulative-frequency denominator
 *     u8  n_nonzero          1..65
 *     n_nonzero × { u8 symbol_index, LEB128 freq }
 * Invariant: Σ freq[i] == M exactly.  Parser verifies and panics on mismatch.
 *
 * For encoding, we build a histogram from the symbol buffer, normalise so the
 * sum is exactly M, then serialise.  Normalisation preserves the "present →
 * freq ≥ 1" invariant required by rANS (a zero-prob symbol can't be encoded).
 */

/* Count each symbol value 0..64 into hist[65]. */
static void c3d_histogram65(const uint8_t *symbols, size_t n, uint32_t hist[65]) {
    memset(hist, 0, 65 * sizeof(uint32_t));
    for (size_t i = 0; i < n; ++i) {
        c3d_assert(symbols[i] < 65u);
        hist[symbols[i]]++;
    }
}

/* Normalise a 65-entry histogram so the nonzero entries sum to M = 1<<denom_shift.
 * Every originally-nonzero entry ends ≥ 1.  Writes freqs[65].
 *
 * Algorithm (after ryg_rans):
 *   1. alloc[i] = (hist[i] * M) / T       (where T = Σ hist)
 *   2. if hist[i] > 0 and alloc[i] == 0: bump alloc[i] to 1
 *   3. adjust total up or down by shaving from / adding to the largest entry
 *      until sum == M. */
static void c3d_normalise_freqs(const uint32_t hist[65], uint32_t denom_shift,
                                uint32_t freqs[65])
{
    c3d_assert(denom_shift >= 1 && denom_shift <= 14);
    const uint32_t M = 1u << denom_shift;

    uint64_t T = 0;
    for (unsigned i = 0; i < 65; ++i) T += hist[i];
    c3d_assert(T > 0);

    /* Initial floor-scale allocation. */
    uint64_t used = 0;
    for (unsigned i = 0; i < 65; ++i) {
        if (hist[i] == 0) { freqs[i] = 0; continue; }
        uint64_t f = ((uint64_t)hist[i] * M) / T;
        if (f == 0) f = 1;
        freqs[i] = (uint32_t)f;
        used += f;
    }

    /* Adjust to hit M exactly.  Iteratively trim/give at the largest entry;
     * a single pass usually suffices, but we loop to be safe. */
    while (used != M) {
        unsigned best = 0;
        uint32_t best_f = freqs[0];
        for (unsigned i = 1; i < 65; ++i) {
            if (freqs[i] > best_f) { best_f = freqs[i]; best = i; }
        }
        if (used > M) {
            uint64_t over = used - M;
            /* Trim from `best`, but never drop to 0 if it was nonzero. */
            uint32_t keep_min = (hist[best] > 0) ? 1u : 0u;
            uint32_t max_trim = freqs[best] - keep_min;
            uint32_t trim = (over < max_trim) ? (uint32_t)over : max_trim;
            freqs[best] -= trim;
            used -= trim;
            c3d_assert(trim > 0);   /* forward progress */
        } else {
            uint64_t under = M - used;
            freqs[best] += (uint32_t)under;
            used += under;
        }
    }

    /* Sanity: every originally-nonzero symbol still has freq ≥ 1; sum == M. */
    uint64_t check = 0;
    for (unsigned i = 0; i < 65; ++i) {
        if (hist[i] > 0) c3d_assert(freqs[i] >= 1u);
        check += freqs[i];
    }
    c3d_assert(check == M);
}

/* Serialise: writes denom_shift, n_nonzero, then per-symbol (sym, LEB128 freq).
 * Returns bytes written.  out_cap must be large enough (worst case: 2 + 65*(1+10) = 717 B). */
static size_t c3d_freqs_serialise(uint32_t denom_shift, const uint32_t freqs[65],
                                  uint8_t *out, size_t out_cap)
{
    c3d_assert(out_cap >= 2);
    unsigned n_nonzero = 0;
    for (unsigned i = 0; i < 65; ++i) if (freqs[i] > 0) n_nonzero++;
    c3d_assert(n_nonzero >= 1 && n_nonzero <= 65);

    size_t w = 0;
    out[w++] = (uint8_t)denom_shift;
    out[w++] = (uint8_t)n_nonzero;
    for (unsigned i = 0; i < 65; ++i) {
        if (freqs[i] == 0) continue;
        c3d_assert(w < out_cap);
        out[w++] = (uint8_t)i;
        w += c3d_leb128_encode(freqs[i], out + w, out_cap - w);
    }
    return w;
}

/* Parse the reverse.  Writes denom_shift, freqs[65] (zero-filled first).
 * Returns bytes consumed.  Panics on sum != M or malformed input. */
static size_t c3d_freqs_parse(const uint8_t *in, size_t in_len,
                              uint32_t *denom_shift, uint32_t freqs[65])
{
    c3d_assert(in_len >= 2);
    uint32_t ds = in[0];
    unsigned n_nonzero = in[1];
    c3d_assert(ds >= 1 && ds <= 14);
    c3d_assert(n_nonzero >= 1 && n_nonzero <= 65);

    size_t r = 2;
    memset(freqs, 0, 65 * sizeof(uint32_t));

    uint64_t sum = 0;
    int last_sym = -1;
    for (unsigned k = 0; k < n_nonzero; ++k) {
        c3d_assert(r < in_len);
        uint8_t sym = in[r++];
        c3d_assert(sym < 65);
        c3d_assert((int)sym > last_sym);   /* symbols must be ascending */
        last_sym = (int)sym;

        uint64_t f = 0;
        r += c3d_leb128_decode(in + r, in_len - r, &f);
        c3d_assert(f >= 1 && f <= (1u << ds));
        freqs[sym] = (uint32_t)f;
        sum += f;
    }
    c3d_assert(sum == (uint64_t)(1u << ds));
    *denom_shift = ds;
    return r;
}

/* ========================================================================= *
 *  §E  CDF 9/7 lifting DWT (1D and separable 3D)                            *
 * ========================================================================= *
 *
 * Lifting cascade per JPEG 2000 Part 1, Annex H (informative):
 *     1. d[i] += α (s[i-1] + s[i+1])    (predict 1)
 *     2. s[i] += β (d[i-1] + d[i])      (update  1)
 *     3. d[i] += γ (s[i-1] + s[i+1])    (predict 2)
 *     4. s[i] += δ (d[i-1] + d[i])      (update  2)
 *     5. s *= 1/K,  d *= K              (scaling)
 *
 * Whole-sample symmetric boundary extension at both ends:
 *     x[-1]       = x[1]
 *     x[N]        = x[N-2]
 *
 * After the 1D lift, we deinterleave: s in x[0..N/2), d in x[N/2..N).
 * Inverse 1D: reinterleave, then run the cascade in reverse.
 *
 * 3D: apply 1D along X, Y, Z (in that order), then recurse on the LLL octant.
 * 5 levels on 256³ → LLL_5 at [0:8, 0:8, 0:8].
 */

#define C3D_CDF97_ALPHA (-1.586134342059924f)
#define C3D_CDF97_BETA  (-0.052980118572961f)
#define C3D_CDF97_GAMMA ( 0.882911075530934f)
#define C3D_CDF97_DELTA ( 0.443506852043971f)
#define C3D_CDF97_K     ( 1.230174104914001f)
#define C3D_CDF97_INV_K ( 0.812893066115961f)

/* In-place 1D lift on interleaved samples x[0..N), N even. */
static void c3d_cdf97_lift_fwd(float *x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);

    /* Predict 1: odd += α (even_L + even_R).  WSS at right: x[N] = x[N-2]. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] += C3D_CDF97_ALPHA * (x[i-1] + x[i+1]);
    x[N-1] += 2.0f * C3D_CDF97_ALPHA * x[N-2];

    /* Update 1: even += β (odd_L + odd_R).  WSS at left: x[-1] = x[1]. */
    x[0] += 2.0f * C3D_CDF97_BETA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] += C3D_CDF97_BETA * (x[i-1] + x[i+1]);

    /* Predict 2: γ. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] += C3D_CDF97_GAMMA * (x[i-1] + x[i+1]);
    x[N-1] += 2.0f * C3D_CDF97_GAMMA * x[N-2];

    /* Update 2: δ. */
    x[0] += 2.0f * C3D_CDF97_DELTA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] += C3D_CDF97_DELTA * (x[i-1] + x[i+1]);

    /* Scale: even (s) *= 1/K, odd (d) *= K. */
    for (size_t i = 0; i < N; i += 2) x[i]   *= C3D_CDF97_INV_K;
    for (size_t i = 1; i < N; i += 2) x[i]   *= C3D_CDF97_K;
}

static void c3d_cdf97_lift_inv(float *x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);

    /* Undo scaling. */
    for (size_t i = 0; i < N; i += 2) x[i] *= C3D_CDF97_K;
    for (size_t i = 1; i < N; i += 2) x[i] *= C3D_CDF97_INV_K;

    /* Undo update 2: even -= δ (odd_L + odd_R). */
    x[0] -= 2.0f * C3D_CDF97_DELTA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] -= C3D_CDF97_DELTA * (x[i-1] + x[i+1]);

    /* Undo predict 2. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] -= C3D_CDF97_GAMMA * (x[i-1] + x[i+1]);
    x[N-1] -= 2.0f * C3D_CDF97_GAMMA * x[N-2];

    /* Undo update 1. */
    x[0] -= 2.0f * C3D_CDF97_BETA * x[1];
    for (size_t i = 2; i < N; i += 2)
        x[i] -= C3D_CDF97_BETA * (x[i-1] + x[i+1]);

    /* Undo predict 1. */
    for (size_t i = 1; i + 1 < N; i += 2)
        x[i] -= C3D_CDF97_ALPHA * (x[i-1] + x[i+1]);
    x[N-1] -= 2.0f * C3D_CDF97_ALPHA * x[N-2];
}

/* Deinterleave x[0..N) → [evens | odds] using aux[0..N) as scratch. */
static void c3d_deinterleave(float *x, size_t N, float *aux) {
    size_t half = N / 2;
    for (size_t i = 0; i < half; ++i) aux[i]        = x[2*i];
    for (size_t i = 0; i < half; ++i) aux[half + i] = x[2*i + 1];
    memcpy(x, aux, N * sizeof(float));
}
static void c3d_interleave(float *x, size_t N, float *aux) {
    size_t half = N / 2;
    for (size_t i = 0; i < half; ++i) aux[2*i]     = x[i];
    for (size_t i = 0; i < half; ++i) aux[2*i + 1] = x[half + i];
    memcpy(x, aux, N * sizeof(float));
}

/* 1D DWT: lift + deinterleave.  aux must be N floats of scratch. */
static void c3d_dwt_1d_fwd(float *x, size_t N, float *aux) {
    c3d_cdf97_lift_fwd(x, N);
    c3d_deinterleave(x, N, aux);
}
static void c3d_dwt_1d_inv(float *x, size_t N, float *aux) {
    c3d_interleave(x, N, aux);
    c3d_cdf97_lift_inv(x, N);
}

/* 4-column-parallel 1D lift + deinterleave, used for Y/Z axis passes.
 *
 * `x` holds four columns interleaved: x[i*4 + c] is column c at index i.
 * Each `#pragma GCC ivdep` / unrolled `for (c)` loop is 4 parallel FMAs that
 * the compiler trivially vectorises to one 128-bit NEON op.  No intrinsics. */
/* Tile width for the Y/Z passes.  8 columns at a time fits two 128-bit NEON
 * FMAs per step without reloading the lane-pair; the compiler autovectorises
 * both halves cleanly.  Keep this a compile-time constant so loops can unroll. */
#define C3D_TILE_X 8

static void c3d_cdf97_lift_fwd_x4(float *restrict x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);
    /* Predict 1. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_ALPHA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] += 2.0f * C3D_CDF97_ALPHA * x[(N-2)*C3D_TILE_X + c];
    /* Update 1. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] += 2.0f * C3D_CDF97_BETA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_BETA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Predict 2. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_GAMMA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] += 2.0f * C3D_CDF97_GAMMA * x[(N-2)*C3D_TILE_X + c];
    /* Update 2. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] += 2.0f * C3D_CDF97_DELTA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] += C3D_CDF97_DELTA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Scale. */
    for (size_t i = 0; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_INV_K;
    for (size_t i = 1; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_K;
}

static void c3d_cdf97_lift_inv_x4(float *restrict x, size_t N) {
    c3d_assert(N >= 4 && (N & 1) == 0);
    /* Undo scale. */
    for (size_t i = 0; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_K;
    for (size_t i = 1; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) x[i*C3D_TILE_X + c] *= C3D_CDF97_INV_K;
    /* Undo update 2. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] -= 2.0f * C3D_CDF97_DELTA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_DELTA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Undo predict 2. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_GAMMA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] -= 2.0f * C3D_CDF97_GAMMA * x[(N-2)*C3D_TILE_X + c];
    /* Undo update 1. */
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[0 + c] -= 2.0f * C3D_CDF97_BETA * x[1*C3D_TILE_X + c];
    for (size_t i = 2; i < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_BETA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    /* Undo predict 1. */
    for (size_t i = 1; i + 1 < N; i += 2)
        for (unsigned c = 0; c < C3D_TILE_X; ++c)
            x[i*C3D_TILE_X + c] -= C3D_CDF97_ALPHA * (x[(i-1)*C3D_TILE_X + c] + x[(i+1)*C3D_TILE_X + c]);
    for (unsigned c = 0; c < C3D_TILE_X; ++c)
        x[(N-1)*C3D_TILE_X + c] -= 2.0f * C3D_CDF97_ALPHA * x[(N-2)*C3D_TILE_X + c];
}

/* Deinterleave TILE_X interleaved columns in-place: per column, evens go to
 * first half, odds to second half.  aux must be N*TILE_X floats of scratch. */
static void c3d_deinterleave_x4(float *restrict x, size_t N, float *restrict aux) {
    size_t half = N / 2;
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[i*C3D_TILE_X + c]        = x[(2*i)*C3D_TILE_X + c];
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[(half+i)*C3D_TILE_X + c] = x[(2*i+1)*C3D_TILE_X + c];
    memcpy(x, aux, N * C3D_TILE_X * sizeof(float));
}
static void c3d_interleave_x4(float *restrict x, size_t N, float *restrict aux) {
    size_t half = N / 2;
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[(2*i)*C3D_TILE_X + c]   = x[i*C3D_TILE_X + c];
    for (size_t i = 0; i < half; ++i)
        for (unsigned c = 0; c < C3D_TILE_X; ++c) aux[(2*i+1)*C3D_TILE_X + c] = x[(half+i)*C3D_TILE_X + c];
    memcpy(x, aux, N * C3D_TILE_X * sizeof(float));
}

static void c3d_dwt_1d_fwd_x4(float *restrict x, size_t N, float *restrict aux) {
    c3d_cdf97_lift_fwd_x4(x, N);
    c3d_deinterleave_x4(x, N, aux);
}
static void c3d_dwt_1d_inv_x4(float *restrict x, size_t N, float *restrict aux) {
    c3d_interleave_x4(x, N, aux);
    c3d_cdf97_lift_inv_x4(x, N);
}

/* 3D single-level forward on the [0:side, 0:side, 0:side] sub-cube of a 256³
 * volume.  scratch must be at least 8 * C3D_CHUNK_SIDE floats. */
#define C3D_STRIDE_Y ((size_t)C3D_CHUNK_SIDE)
#define C3D_STRIDE_Z ((size_t)C3D_CHUNK_SIDE * C3D_CHUNK_SIDE)

/* For Y/Z tiled passes we use TILE_X contiguous X columns at a time.  Scratch
 * layout: tile[N*TILE_X] + aux[N*TILE_X] = 2 * TILE_X * side floats. */
#define C3D_Y_TILE  C3D_TILE_X
#define C3D_Z_TILE  C3D_TILE_X

static void c3d_dwt3_fwd_level(float *buf, size_t side, float *scratch) {
    float *tile = scratch;
    float *aux  = scratch + C3D_TILE_X * C3D_CHUNK_SIDE;

    /* X pass — row stride 1, contiguous.  Unchanged (already vectorised). */
    for (size_t z = 0; z < side; ++z) {
        for (size_t y = 0; y < side; ++y) {
            float *row = &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y];
            c3d_dwt_1d_fwd(row, side, aux);
        }
    }
    /* Y pass — 4 adjacent X-columns at a time (cache-line-sized load/store). */
    c3d_assert((side & 3u) == 0);
    for (size_t z = 0; z < side; ++z) {
        for (size_t xb = 0; xb < side; xb += C3D_Y_TILE) {
            for (size_t y = 0; y < side; ++y)
                memcpy(&tile[y * C3D_TILE_X], &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                       C3D_TILE_X * sizeof(float));
            c3d_dwt_1d_fwd_x4(tile, side, aux);
            for (size_t y = 0; y < side; ++y)
                memcpy(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb], &tile[y * C3D_TILE_X],
                       C3D_TILE_X * sizeof(float));
        }
    }
    /* Z pass — same tiling. */
    for (size_t y = 0; y < side; ++y) {
        for (size_t xb = 0; xb < side; xb += C3D_Z_TILE) {
            for (size_t z = 0; z < side; ++z)
                memcpy(&tile[z * C3D_TILE_X], &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                       C3D_TILE_X * sizeof(float));
            c3d_dwt_1d_fwd_x4(tile, side, aux);
            for (size_t z = 0; z < side; ++z)
                memcpy(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb], &tile[z * C3D_TILE_X],
                       C3D_TILE_X * sizeof(float));
        }
    }
}

static void c3d_dwt3_inv_level(float *buf, size_t side, float *scratch) {
    float *tile = scratch;
    float *aux  = scratch + C3D_TILE_X * C3D_CHUNK_SIDE;

    /* Inverse order: Z, Y, X. */
    c3d_assert((side & 3u) == 0);
    for (size_t y = 0; y < side; ++y) {
        for (size_t xb = 0; xb < side; xb += C3D_Z_TILE) {
            for (size_t z = 0; z < side; ++z)
                memcpy(&tile[z * C3D_TILE_X], &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                       C3D_TILE_X * sizeof(float));
            c3d_dwt_1d_inv_x4(tile, side, aux);
            for (size_t z = 0; z < side; ++z)
                memcpy(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb], &tile[z * C3D_TILE_X],
                       C3D_TILE_X * sizeof(float));
        }
    }
    for (size_t z = 0; z < side; ++z) {
        for (size_t xb = 0; xb < side; xb += C3D_Y_TILE) {
            for (size_t y = 0; y < side; ++y)
                memcpy(&tile[y * C3D_TILE_X], &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb],
                       C3D_TILE_X * sizeof(float));
            c3d_dwt_1d_inv_x4(tile, side, aux);
            for (size_t y = 0; y < side; ++y)
                memcpy(&buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + xb], &tile[y * C3D_TILE_X],
                       C3D_TILE_X * sizeof(float));
        }
    }
    for (size_t z = 0; z < side; ++z) {
        for (size_t y = 0; y < side; ++y) {
            float *row = &buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y];
            c3d_dwt_1d_inv(row, side, aux);
        }
    }
}

/* Full 5-level forward DWT on a 256³ f32 buffer.
 * scratch must be ≥ 8 * C3D_CHUNK_SIDE floats (2 KiB). */
static void c3d_dwt3_fwd(float *buf, float *scratch) {
    size_t side = C3D_CHUNK_SIDE;
    for (unsigned lvl = 0; lvl < C3D_N_DWT_LEVELS; ++lvl) {
        c3d_dwt3_fwd_level(buf, side, scratch);
        side /= 2;
    }
}

/* Inverse `n_synth_levels` levels, 0 ≤ n ≤ 5.
 *   n=0 → no inverse (output is LLL_5 at [0:8, 0:8, 0:8]).
 *   n=k → synthesise levels 5, 4, ..., 6-k; output is LLL_{5-k} at [0:(8<<k), ...].
 *   n=5 → full inverse, output at [0:256, 0:256, 0:256]. */
static void c3d_dwt3_inv_levels(float *buf, unsigned n_synth_levels, float *scratch) {
    c3d_assert(n_synth_levels <= C3D_N_DWT_LEVELS);
    for (unsigned i = 0; i < n_synth_levels; ++i) {
        size_t active_side = (size_t)16u << i;   /* 16, 32, 64, 128, 256 */
        c3d_dwt3_inv_level(buf, active_side, scratch);
    }
}

/* ========================================================================= *
 *  §F  Quantizer, symbol mapping (zigzag + escape), and subband info        *
 * ========================================================================= */

/* Dead-zone uniform quantizer, dead zone = step, step width = step.
 *   |c| < step/2      → 0
 *   step/2 ≤ |c|      → sign(c) * (floor((|c| - step/2) / step) + 1) */
static inline int32_t c3d_quant(float c, float step) {
    float ac = (c < 0.0f) ? -c : c;
    if (ac < 0.5f * step) return 0;
    int32_t q = (int32_t)((ac - 0.5f * step) / step) + 1;
    return (c < 0.0f) ? -q : q;
}

/* Laplacian-optimal mid-tread dequantizer.
 *   q = 0 → 0
 *   q ≠ 0 → sign(q) * (|q| - 0.5 + alpha) * step */
static inline float c3d_dequant(int32_t q, float step, float alpha) {
    if (q == 0) return 0.0f;
    float aq  = (float)((q < 0) ? -q : q);
    float mag = (aq - 0.5f + alpha) * step;
    return (q < 0) ? -mag : mag;
}

/* Standard 32-bit zigzag: signed ↔ unsigned bijection.
 *   0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...  */
static inline uint32_t c3d_zigzag32(int32_t v) {
    return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31);
}
static inline int32_t c3d_unzigzag32(uint32_t z) {
    return (int32_t)((z >> 1) ^ -(int32_t)(z & 1u));
}

/* Map a quantized integer to a c3d alphabet symbol.
 *   If |q| ≤ 31 (i.e. zigzag(q) ≤ 62) and zigzag(q) = 63 is -32 — the direct
 *   range is zigzag ∈ [0, 63], covering signed q ∈ [-32, 31].
 *   Escape symbol is 64; escape payload (LEB128 varint of the zigzag value)
 *   is written by the caller to the subband's escape stream. */
#define C3D_SYM_ESCAPE 64u
#define C3D_N_SYMBOLS  65u   /* 0..63 direct + 1 escape                        */

/* Returns symbol; if the symbol is C3D_SYM_ESCAPE, writes the full zigzag
 * value to *escape_zigzag_out so the caller can emit it as a LEB128 varint. */
static inline uint8_t c3d_quant_to_symbol(int32_t q, uint32_t *escape_zigzag_out) {
    uint32_t z = c3d_zigzag32(q);
    if (z < 64u) {
        *escape_zigzag_out = 0;
        return (uint8_t)z;
    }
    *escape_zigzag_out = z;
    return (uint8_t)C3D_SYM_ESCAPE;
}

/* Inverse: given a symbol and (when escape) a zigzag payload, recover q. */
static inline int32_t c3d_symbol_to_quant(uint8_t sym, uint32_t escape_zigzag) {
    uint32_t z = (sym == C3D_SYM_ESCAPE) ? escape_zigzag : (uint32_t)sym;
    return c3d_unzigzag32(z);
}

/* --- Subband descriptor --------------------------------------------------- *
 *
 * Canonical indexing (PLAN §2.3):
 *   index 0            = LLL_5   at (0,0,0), side 8
 *   index 1..7         = level-5 details, side 8, in 7 non-LLL octants of [0..16]³
 *   index 8..14        = level-4 details, side 16, in [0..32]³
 *   index 15..21       = level-3 details, side 32, in [0..64]³
 *   index 22..28       = level-2 details, side 64, in [0..128]³
 *   index 29..35       = level-1 details, side 128, in [0..256]³
 *
 * Within each level the 7-detail ordering is
 *     HHH, HHL, HLH, LHH, HLL, LHL, LLH
 * i.e. kind indices 1..7 respectively.  The letter order is ZYX (first = Z).
 * Octant offset: each H adds +side on its axis.
 *
 *   kind 1 HHH → (+s, +s, +s)
 *   kind 2 HHL → (+s, +s,  0 )
 *   kind 3 HLH → (+s,  0 , +s)
 *   kind 4 LHH → ( 0 , +s, +s)
 *   kind 5 HLL → (+s,  0 ,  0 )
 *   kind 6 LHL → ( 0 , +s,  0 )
 *   kind 7 LLH → ( 0 ,  0 , +s)
 */

typedef struct {
    unsigned level;    /* 1..5                                               */
    unsigned kind;     /* 0 = LLL (only for LLL_5); else 1..7 per table above */
    uint32_t side;     /* coefficient count per axis                         */
    uint32_t z0, y0, x0; /* origin inside the 256³ coefficient buffer        */
} c3d_subband_info;

static void c3d_subband_info_of(unsigned idx, c3d_subband_info *info) {
    c3d_assert(idx < C3D_N_SUBBANDS);
    if (idx == 0) {
        info->level = 5;
        info->kind  = 0;
        info->side  = 8;
        info->z0 = info->y0 = info->x0 = 0;
        return;
    }
    unsigned i = idx - 1;           /* 0..34 over detail subbands      */
    unsigned level_from_deep = i / 7;  /* 0 = level 5, 4 = level 1        */
    unsigned kind_minus_1    = i % 7;
    info->level = 5u - level_from_deep;
    info->kind  = kind_minus_1 + 1u;
    info->side  = 8u << level_from_deep;   /* 8, 16, 32, 64, 128          */

    uint32_t s = info->side;
    uint32_t z_hi = 0, y_hi = 0, x_hi = 0;
    switch (info->kind) {
    case 1: z_hi = 1; y_hi = 1; x_hi = 1; break; /* HHH */
    case 2: z_hi = 1; y_hi = 1;            break; /* HHL */
    case 3: z_hi = 1;            x_hi = 1; break; /* HLH */
    case 4:            y_hi = 1; x_hi = 1; break; /* LHH */
    case 5: z_hi = 1;                      break; /* HLL */
    case 6:            y_hi = 1;           break; /* LHL */
    case 7:                       x_hi = 1; break; /* LLH */
    default: c3d_panic(__FILE__, __LINE__, "bad subband kind");
    }
    info->z0 = z_hi * s;
    info->y0 = y_hi * s;
    info->x0 = x_hi * s;
}

/* Extract a subband's coefficients from the 3D buffer into a flat array,
 * row-major (z outermost).  Returns count = side^3. */
static size_t c3d_subband_extract(const float *buf,
                                  const c3d_subband_info *sb,
                                  float *out_flat)
{
    size_t count = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        out_flat[count++] = buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x];
    }
    return count;
}
static void c3d_subband_scatter(float *buf,
                                const c3d_subband_info *sb,
                                const float *in_flat)
{
    size_t count = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x] = in_flat[count++];
    }
}

/* ========================================================================= *
 *  §G/§H/§I  Chunk encoder + decoder (SELF mode only for now)               *
 * ========================================================================= *
 *
 * Chunk layout (PLAN §2.2):
 *    0     chunk header (40 B)
 *   40     qmul[36]      (144 B)
 *  184     subband_offset[36] (144 B)
 *  328     lod_offset[6]   (24 B)
 *  352     entropy payload (variable, resolution-first)
 *
 * Per-subband bitstream (PLAN §3.4):
 *    0   u16 freq_table_size
 *    2   freq_table (denom_shift u8, n_nonzero u8, n_nonzero × {u8 sym, LEB128 freq})
 *    ?   u32 n_symbols
 *    ?   u32 rans_block_size
 *    ?   rans_header (32 B) + rans_renorm (variable)
 *    ?   escape_stream (variable)
 */

#define C3D_CHUNK_FIXED_SIZE 352u
#define C3D_Q_MIN            (1.0f / 4096.0f) /* 2^-12 (was 2^-6 — wider range
                                                  needed to reach big target
                                                  budgets under perceptual
                                                  weighting which compresses
                                                  HF subbands aggressively). */
#define C3D_Q_MAX            4096.0f          /* 2^12 */

/* --- Perceptual per-subband quantizer weights --------------------------- *
 *
 * CDF 9/7 synthesis gains squared per axis (||synthesis basis||² from JPEG
 * 2000 Part 1 Annex F.2).  An error ε in a subband with L_count low-pass
 * axes and H_count high-pass axes contributes to reconstruction MSE by
 *    ε² · (G_L²)^L_count · (G_H²)^H_count
 * For R-D-optimal bit allocation, subband step ∝ 1/sqrt(weight), so deep
 * low-frequency bands get fine quantization and high-frequency bands get
 * coarse.  Normalised so geomean across 36 subbands == 1.0, preserving the
 * chunk_scalar q range semantics.  Computed lazily on first use. */

#define C3D_CDF97_GAIN_L_SQ 2.08f
#define C3D_CDF97_GAIN_H_SQ 0.48f

static float c3d_subband_baseline_table[C3D_N_SUBBANDS];
static bool  c3d_subband_baseline_init = false;

static unsigned c3d_kind_h_count(unsigned kind) {
    switch (kind) {
    case 0: return 0;                       /* LLL_5               */
    case 1: return 3;                       /* HHH                 */
    case 2: case 3: case 4: return 2;       /* HHL, HLH, LHH       */
    case 5: case 6: case 7: return 1;       /* HLL, LHL, LLH       */
    default: return 0;
    }
}

/* Fill baselines[36] using 1/w^softness weighting.  w = product of axis
 * CDF97 synthesis gains² per subband; strict R-D-optimal is softness=0.5 but
 * that collapses step dynamic range so much the rate-control loop saturates
 * at q_min.  Default 0.25 keeps control responsive; adaptive path varies
 * softness mildly with target_ratio (§G).  Normalised to geomean 1.0 so the
 * chunk_scalar q semantics are preserved across softness values. */
static void c3d_fill_subband_baselines(float softness, float baselines[C3D_N_SUBBANDS]) {
    float b[C3D_N_SUBBANDS];
    double log_sum = 0.0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb; c3d_subband_info_of(i, &sb);
        unsigned h = c3d_kind_h_count(sb.kind);
        float log_w = (float)(3u * (sb.level - 1u) + (3u - h)) * logf(C3D_CDF97_GAIN_L_SQ)
                    + (float)h * logf(C3D_CDF97_GAIN_H_SQ);
        b[i] = expf(-softness * log_w);
        log_sum += -(double)softness * (double)log_w;
    }
    float scale = expf(-(float)(log_sum / (double)C3D_N_SUBBANDS));
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i)
        baselines[i] = b[i] * scale;
}

static void c3d_compute_subband_baselines(void) {
    c3d_fill_subband_baselines(0.25f, c3d_subband_baseline_table);
    c3d_subband_baseline_init = true;
}

static inline float c3d_subband_baseline(unsigned i) {
    if (!c3d_subband_baseline_init) c3d_compute_subband_baselines();
    return c3d_subband_baseline_table[i];
}

/* Default rANS denom_shift per subband.  LLL_5 needs M=16384 (only 512 coeffs
 * need a tight histogram); every other subband uses M=4096.  Tried pushing
 * finest HF bands to M=1024: LUT shrinks from 4 KiB to 1 KiB but each symbol
 * consumes fewer fractional bits of state → renorm reads ~12 % more bytes,
 * net decode regressed 15-20 % at fine q.  The LUT was already L1-resident. */
static inline uint32_t c3d_default_denom_shift(unsigned i) {
    return (i == 0) ? 14u : 12u;
}

/* Per-subband default Laplacian-optimal dead-zone offset (dequant α).  The
 * reconstruction is (|q| - 0.5 + α) * step; smaller α biases toward zero
 * (matches heavier-tailed distributions in pure-HF subbands), larger α
 * toward bin midpoint (better for the near-uniform LLL_5 DC residual).
 * 0.375 was the previous global default — retained for mid-frequency bands.
 * Ctx overrides still win; this only changes the fall-through default. */
static inline float c3d_default_alpha(unsigned s) {
    if (s == 0) return 0.45f;  /* LLL_5: broad distribution, closer to midpoint */
    c3d_subband_info sb; c3d_subband_info_of(s, &sb);
    unsigned h = c3d_kind_h_count(sb.kind);
    /* h=1 mixed-LF: 0.40;  h=2 mixed: 0.375;  h=3 pure HF (HHH): 0.33 */
    switch (h) {
    case 1: return 0.40f;
    case 2: return 0.375f;
    case 3: return 0.33f;
    default: return 0.375f;
    }
}

/* Per-subband softness (1 / w^softness perceptual weighting).  A calibration
 * sweep on real scroll CT data (s ∈ {0.15..0.55} × r ∈ {5..200}) showed
 * monotonically rising PSNR up to s≈0.50 at every ratio, with a plateau
 * past 0.50 (gains drop below 0.05 dB).  s=0.50 is also the strict R-D-
 * optimal exponent (1/sqrt(w) ≡ step ∝ 1/sqrt(synthesis_gain²)).
 * We used to clamp at 0.32 with q_min=2^-6 because rate control saturated;
 * with q_min=2^-12 the full range is usable and 0.50 wins by +0.5 to +1.2
 * dB over the old adaptive curve.  Env override kept for future sweeps. */
static float c3d_adaptive_softness(float target_ratio) {
    (void)target_ratio;
    const char *env = getenv("C3D_SOFTNESS");
    if (env) {
        float v = (float)atof(env);
        if (v >= 0.05f && v <= 0.6f) return v;
    }
    return 0.50f;
}

/* c3d_ctx struct definition — used by §G/§H/§I encode/decode paths.
 * Full parse/serialise/builder implementation lives in §K. */
struct c3d_ctx {
    uint8_t  self_hash[16];
    uint16_t block_size;

    bool     has_laplacian_alpha;
    float    laplacian_alpha[C3D_N_SUBBANDS];

    bool     has_quantizer_baseline;
    float    quantizer_baseline[C3D_N_SUBBANDS];

    bool     has_freq_tables;
    uint32_t denom_shifts[C3D_N_SUBBANDS];
    uint32_t freqs[C3D_N_SUBBANDS][65];
};

/* Number of subbands required to decode each LOD (prefix of canonical order). */
static const unsigned c3d_n_subbands_for_lod[C3D_N_LODS] = {
    C3D_N_SUBBANDS,  /* LOD 0: all 36                          */
    29,              /* LOD 1: excludes level-1 details (29..35) */
    22,              /* LOD 2: excludes levels 1 and 2           */
    15,              /* LOD 3                                    */
    8,               /* LOD 4: LLL_5 + level-5 details           */
    1,               /* LOD 5: LLL_5 only                        */
};

/* -- Reusable encoder / decoder scratch contexts -------------------------- *
 *
 * c3d_encoder owns ~115 MiB:  coeff_buf 64M + sub_symbols 2M + sub_escapes 2M
 *                              + rans_scratch 8M + small DWT scratch.
 * c3d_decoder owns ~80 MiB:   coeff_buf 64M + sub_symbols 2M + small.
 *
 * Both are exposed to callers as opaque handles via c3d_{encoder,decoder}_new.
 * The stateless c3d_chunk_encode/decode functions are now thin wrappers that
 * allocate a temporary context per call. */

struct c3d_encoder {
    float   *coeff_buf;
    uint8_t *sub_symbols;
    uint8_t *sub_escapes;
    uint8_t *rans_scratch;
    float    dwt_scratch[2 * C3D_TILE_X * C3D_CHUNK_SIDE];
    /* Dynamic per-subband baselines (adaptive perceptual softness).  Populated
     * by c3d_encoder_chunk_encode from target_ratio; unused by encode_at_q
     * (which falls back to the cached default-softness table). */
    float    dyn_baselines[C3D_N_SUBBANDS];
    bool     has_dyn_baselines;
    /* Per-subband max |coeff| after prepare_chunk.  Lets the estimator skip
     * the full quant scan on subbands that are definitely all-zero at the
     * trial step — matches the empty-subband fast path in the real emit. */
    float    max_abs_per_subband[C3D_N_SUBBANDS];
    bool     has_max_abs;
    /* Raw post-DWT max|coeff| (= coeff_scale).  Absorbed into per-subband
     * step at emit/estimate time so the normalise-to-[-1,1] scan can be
     * skipped — see c3d_prepare_chunk. */
    float    coeff_scale;
    /* Warm start for rate-control bisection.  Populated from the previous
     * call at the same target_ratio; successive chunks in a shard usually
     * converge to a very similar q, so this cuts bisection from ~8 iters
     * to ~3-4. */
    float    last_q;
    float    last_target_ratio;
};

struct c3d_decoder {
    float   *coeff_buf;
    uint8_t *sub_symbols;
    float    dwt_scratch[2 * C3D_TILE_X * C3D_CHUNK_SIDE];
};

c3d_encoder *c3d_encoder_new(void) {
    c3d_encoder *e = malloc(sizeof *e);
    c3d_assert(e);
    e->coeff_buf    = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    e->sub_symbols  = malloc((size_t)128 * 128 * 128);
    e->sub_escapes  = malloc((size_t)128 * 128 * 128 / 4 + 1024);
    e->rans_scratch = malloc((size_t)128 * 128 * 128 * 2 + 1024);
    c3d_assert(e->coeff_buf && e->sub_symbols && e->sub_escapes && e->rans_scratch);
    e->has_dyn_baselines = false;
    e->has_max_abs = false;
    e->last_q = 0.0f;
    e->last_target_ratio = 0.0f;
    return e;
}
void c3d_encoder_free(c3d_encoder *e) {
    if (!e) return;
    free(e->coeff_buf);   free(e->sub_symbols);
    free(e->sub_escapes); free(e->rans_scratch);
    free(e);
}

c3d_decoder *c3d_decoder_new(void) {
    c3d_decoder *d = malloc(sizeof *d);
    c3d_assert(d);
    d->coeff_buf   = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK * sizeof(float));
    d->sub_symbols = malloc((size_t)128 * 128 * 128);
    c3d_assert(d->coeff_buf && d->sub_symbols);
    return d;
}
void c3d_decoder_free(c3d_decoder *d) {
    if (!d) return;
    free(d->coeff_buf); free(d->sub_symbols); free(d);
}

/* -- Stage 1: ingest + DWT + compute coeff_scale, normalise. -------------- *
 * Writes the 40 B chunk header and zero-fills table regions to byte 352.
 * Returns true if the chunk is nonempty (needs entropy payload); false if
 * uniform-after-centering (just emit the 352 B header with all-zero tables). */
static bool c3d_prepare_chunk(const uint8_t *in, uint8_t *out,
                              c3d_encoder *s,
                              float *out_dc_offset, float *out_coeff_scale)
{
    /* Header skeleton. */
    memcpy(out + 0, "C3DC", 4);
    c3d_write_u16_le(out + 4, 1u);
    out[6] = 0; out[7] = 0;  /* context_mode = SELF, reserved */
    /* dc_offset, coeff_scale filled later. */
    memset(out + 16, 0, 8 + 16);  /* reserved2 + context_id */
    /* Zero-fill tables (will be overwritten). */
    memset(out + 40, 0, C3D_CHUNK_FIXED_SIZE - 40);

    /* Ingest: u8 → f32 − 128 − dc_offset.  Two passes over `in` (integer
     * accumulator on the first, no coeff_buf write) + one pass that writes
     * coeff_buf exactly once.  Saves one 64 MiB coeff_buf round-trip vs the
     * naive (write f32-128 then subtract dc) ordering. */
    uint64_t u8_sum = 0;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) u8_sum += in[i];
    float dc_offset = (float)u8_sum / (float)C3D_VOXELS_PER_CHUNK - 128.0f;
    for (size_t i = 0; i < C3D_VOXELS_PER_CHUNK; ++i) {
        s->coeff_buf[i] = (float)in[i] - 128.0f - dc_offset;
    }

    /* Forward 3D DWT in place. */
    c3d_dwt3_fwd(s->coeff_buf, s->dwt_scratch);

    /* Single fused pass: per-subband max |coeff| + overall max.  Covers all
     * 36 subbands (= every coefficient in the 256³ buffer exactly once) so
     * the global max is just the max over the per-subband table — no extra
     * 64 MiB scan.  Saves ~12 ms/chunk vs separate loops. */
    float max_abs = 0.0f;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb; c3d_subband_info_of(i, &sb);
        float mx = 0.0f;
        for (uint32_t z = sb.z0; z < sb.z0 + sb.side; ++z)
        for (uint32_t y = sb.y0; y < sb.y0 + sb.side; ++y)
        for (uint32_t x = sb.x0; x < sb.x0 + sb.side; ++x) {
            float a = fabsf(s->coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x]);
            if (a > mx) mx = a;
        }
        s->max_abs_per_subband[i] = mx;
        if (mx > max_abs) max_abs = mx;
    }

    c3d_write_f32_le(out + 8, dc_offset);
    *out_dc_offset = dc_offset;

    if (max_abs == 0.0f) {
        c3d_write_f32_le(out + 12, 1.0f);
        *out_coeff_scale = 1.0f;
        return false;   /* empty path: return 352 B chunk with all-zero tables */
    }

    /* Raw-coefficient format (post-DC, post-DWT, un-normalised).  Per-subband
     * step values absorb coeff_scale at emit time (step = q*baseline*coeff_scale)
     * so quant sees matching units; the decoder dequantizes directly into the
     * raw range and skips the old post-IDWT *coeff_scale multiply.  Skipping
     * the normalise-to-[-1,1] scan saves ~12 ms/chunk.  coeff_scale is still
     * written to the header for inspection but is no longer used on decode —
     * preserved so c3d_inspect / downstream tools see the pre-encode magnitude. */
    float coeff_scale = max_abs;
    c3d_write_f32_le(out + 12, coeff_scale);
    *out_coeff_scale = coeff_scale;

    s->coeff_scale = coeff_scale;
    s->has_max_abs = true;
    return true;
}

/* -- Stage 2: per-subband encode (quantize → symbols + escapes → freq table
 *             → rANS → pack).  Writes bytes to `out`.  Returns bytes written.
 * If `external_freqs` is non-NULL, use those frequencies (EXTERNAL mode):
 * all chunk symbols must have freq ≥ 1 in the provided table, otherwise rANS
 * encoding panics.  freq_table_size is written as 0 in this mode. */
static size_t c3d_encode_one_subband(
    const float *coeff_buf, const c3d_subband_info *sb,
    float step, uint32_t denom_shift,
    const uint32_t *external_freqs,
    uint8_t *sub_symbols, uint8_t *sub_escapes,
    uint8_t *rans_scratch, size_t rans_scratch_size,
    uint8_t *out, size_t out_cap)
{
    size_t n = (size_t)sb->side * sb->side * sb->side;

    /* Pass 1: quantize + symbol + escape + histogram. */
    uint32_t hist[65] = {0};
    size_t escape_pos = 0;
    size_t idx = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        float c = coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x];
        int32_t qv = c3d_quant(c, step);
        uint32_t zigzag;
        uint8_t sym = c3d_quant_to_symbol(qv, &zigzag);
        sub_symbols[idx++] = sym;
        hist[sym]++;
        if (sym == C3D_SYM_ESCAPE) {
            /* LEB128: u32 zigzag needs at most 5 bytes. */
            escape_pos += c3d_leb128_encode(zigzag, sub_escapes + escape_pos, 5);
        }
    }
    c3d_assert(idx == n);

    /* All-zero fast path: every coefficient quantizes to symbol 0.  Emit a
     * 2-byte sentinel (freq_table_size = 0xFFFF) and stop — decoder zero-fills
     * the subband.  Saves ~40 bytes/subband on sparse chunks (rANS state
     * header alone is 32 B).  Common case for HF subbands at moderate ratios. */
    if (hist[0] == n) {
        c3d_assert(out_cap >= 2);
        c3d_write_u16_le(out, 0xFFFFu);
        return 2;
    }

    /* Pick the frequency source per-subband.  Prefer external (saves the
     * in-band freq table bytes) when it covers every symbol in this chunk;
     * otherwise fall back to a locally-built table. */
    uint32_t local_freqs[65];
    const uint32_t *freqs_to_use;
    bool use_external = false;
    if (external_freqs) {
        use_external = true;
        for (unsigned k = 0; k < 65; ++k) {
            if (hist[k] > 0 && external_freqs[k] == 0) { use_external = false; break; }
        }
    }
    if (use_external) {
        freqs_to_use = external_freqs;
    } else {
        c3d_normalise_freqs(hist, denom_shift, local_freqs);
        freqs_to_use = local_freqs;
    }
    c3d_rans_tables tbl;
    c3d_rans_build_tables(&tbl, denom_shift, freqs_to_use, 65);

    /* Emit per-subband bitstream layout. */
    size_t w = 0;

    /* [freq_table_size u16] — 0 for EXTERNAL (table lives in ctx). */
    c3d_assert(w + 2 <= out_cap);
    size_t ftable_size_pos = w;
    w += 2;

    /* [freq_table bytes] — only when falling back to in-band. */
    size_t ftable_bytes;
    if (use_external) {
        ftable_bytes = 0;
    } else {
        ftable_bytes = c3d_freqs_serialise(denom_shift, local_freqs, out + w, out_cap - w);
    }
    c3d_assert(ftable_bytes <= 65535);
    c3d_write_u16_le(out + ftable_size_pos, (uint16_t)ftable_bytes);
    w += ftable_bytes;

    /* [n_symbols u32][rans_block_size u32 placeholder] */
    c3d_assert(w + 8 <= out_cap);
    c3d_write_u32_le(out + w, (uint32_t)n); w += 4;
    size_t rans_size_pos = w;
    w += 4;

    /* [rans_header 32 B][rans_renorm variable] — emitted by c3d_rans_enc_x8. */
    size_t rans_bytes = c3d_rans_enc_x8(
        sub_symbols, n, &tbl,
        rans_scratch, rans_scratch_size,
        out + w, out_cap - w);
    c3d_write_u32_le(out + rans_size_pos, (uint32_t)rans_bytes);
    w += rans_bytes;

    /* [escape_stream variable] */
    c3d_assert(w + escape_pos <= out_cap);
    memcpy(out + w, sub_escapes, escape_pos);
    w += escape_pos;

    return w;
}

/* Cheap entropy estimator: quantize + histogram + Shannon (or cross-entropy
 * vs external freqs when applicable) + escape LEB128 size, no rANS encode,
 * no freq-table normalisation, no serialise.  Used by the rate-control
 * bisection so each iteration costs ~1 quantize pass instead of a full emit.
 * Returns estimated subband byte size (double so errors aggregate cleanly).
 * Matches c3d_encode_one_subband's "use external iff external covers every
 * symbol present in this chunk" decision so the estimate tracks reality. */
static double c3d_estimate_one_subband_bytes(
    const float *coeff_buf, const c3d_subband_info *sb,
    float step, uint32_t denom_shift,
    const uint32_t *external_freqs,
    float max_abs)
{
    /* Fast reject: if max |c| in this subband quantizes to 0, the whole band
     * is empty → matches c3d_encode_one_subband's 2-byte sentinel, and we
     * skip the O(N) quant loop entirely.  On sparse chunks (typical at
     * r≥50) this hits for 10-20 of 36 subbands per estimator iteration. */
    if (max_abs < 0.5f * step) return 2.0;

    size_t n = (size_t)sb->side * sb->side * sb->side;
    uint32_t hist[65] = {0};
    size_t escape_bytes = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        float c = coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x];
        int32_t qv = c3d_quant(c, step);
        uint32_t zigzag;
        uint8_t sym = c3d_quant_to_symbol(qv, &zigzag);
        hist[sym]++;
        if (sym == C3D_SYM_ESCAPE) {
            /* LEB128 byte count for zigzag value. */
            uint32_t v = zigzag;
            do { escape_bytes++; v >>= 7; } while (v);
        }
    }

    /* All-zero subband fast path matches c3d_encode_one_subband. */
    if (hist[0] == n) return 2.0;

    bool use_external = (external_freqs != NULL);
    if (use_external) {
        for (unsigned k = 0; k < 65; ++k) {
            if (hist[k] > 0 && external_freqs[k] == 0) { use_external = false; break; }
        }
    }

    double code_bits = 0.0;
    if (use_external) {
        double log2_M = (double)denom_shift;
        for (unsigned k = 0; k < 65; ++k) {
            if (!hist[k]) continue;
            code_bits += (double)hist[k] * (log2_M - log2((double)external_freqs[k]));
        }
    } else {
        /* Shannon H(hist).  rANS renorm adds <1 % in practice. */
        double inv_n = 1.0 / (double)n;
        for (unsigned k = 0; k < 65; ++k) {
            if (!hist[k]) continue;
            double p = (double)hist[k] * inv_n;
            code_bits += -(double)hist[k] * log2(p);
        }
    }

    /* rANS overhead: 32 B header + <1 % renorm slack (rounded up). */
    double rans_bytes = code_bits / 8.0 + 32.0;

    /* Approximate ftable_bytes (SELF only): 2 fixed + per-nonzero (1 sym +
     * LEB128 freq; freq up to 2^denom_shift so 1-3 bytes). */
    double ftable_bytes = 0.0;
    if (!use_external) {
        unsigned nnz = 0;
        for (unsigned k = 0; k < 65; ++k) if (hist[k]) nnz++;
        ftable_bytes = 2.0 + 3.0 * (double)nnz;
    }

    /* Per-subband framing: 2 (ftable_size) + ftable + 4 (n_symbols)
     * + 4 (rans_block_size) + rans + escape. */
    return 2.0 + ftable_bytes + 8.0 + rans_bytes + (double)escape_bytes;
}

/* Cheap whole-chunk estimate: sum of per-subband estimates under the same
 * baseline / denom_shift / ctx-override logic as c3d_emit_entropy_at_q. */
static double c3d_estimate_entropy_at_q(float q, const c3d_encoder *s,
                                        const c3d_ctx *ctx)
{
    double total = 0.0;
    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb;
        c3d_subband_info_of(i, &sb);
        float baseline = (ctx && ctx->has_quantizer_baseline) ? ctx->quantizer_baseline[i]
                       : s->has_dyn_baselines                 ? s->dyn_baselines[i]
                                                              : c3d_subband_baseline(i);
        /* Absorb coeff_scale into step: encoder no longer normalises
         * coeff_buf to [-1,1] so subband coefficients are in raw units. */
        float step = q * baseline * s->coeff_scale;
        uint32_t denom_shift = (ctx && ctx->has_freq_tables) ? ctx->denom_shifts[i]
                                                             : c3d_default_denom_shift(i);
        float max_abs = s->has_max_abs ? s->max_abs_per_subband[i] : s->coeff_scale;
        total += c3d_estimate_one_subband_bytes(
            s->coeff_buf, &sb, step, denom_shift,
            (ctx && ctx->has_freq_tables) ? ctx->freqs[i] : NULL,
            max_abs);
    }
    return total;
}

/* -- Stage 3: emit all subbands given normalised coeff_buf and chunk_scalar q.
 * If `ctx` is non-NULL, applies its overrides:
 *   - has_quantizer_baseline: step = q * baseline[s]
 *   - has_freq_tables: uses ctx's freqs, emits freq_table_size = 0 per subband
 * Writes entropy payload into out[352..], fills qmul/subband_offset/lod_offset
 * tables.  Returns total chunk size (352 + entropy bytes). */
static size_t c3d_emit_entropy_at_q(float q, const c3d_encoder *s,
                                    const c3d_ctx *ctx,
                                    uint8_t *out, size_t out_cap)
{
    uint8_t *qmul_ptr   = out + 40;
    uint8_t *suboff_ptr = out + 40 + 144;
    uint8_t *lodoff_ptr = out + 40 + 144 + 144;

    size_t entropy_pos = 0;
    const size_t entropy_cap = out_cap - C3D_CHUNK_FIXED_SIZE;
    const size_t rans_scratch_size = (size_t)128 * 128 * 128 * 2 + 1024;

    for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
        c3d_subband_info sb;
        c3d_subband_info_of(i, &sb);
        float baseline = (ctx && ctx->has_quantizer_baseline) ? ctx->quantizer_baseline[i]
                       : s->has_dyn_baselines                 ? s->dyn_baselines[i]
                                                                : c3d_subband_baseline(i);
        /* Absorb coeff_scale into step (see c3d_prepare_chunk). */
        float step = q * baseline * s->coeff_scale;
        uint32_t denom_shift = (ctx && ctx->has_freq_tables) ? ctx->denom_shifts[i]
                                                              : c3d_default_denom_shift(i);

        c3d_write_f32_le(qmul_ptr + 4 * i, step);
        c3d_write_u32_le(suboff_ptr + 4 * i, (uint32_t)entropy_pos);

        size_t bytes = c3d_encode_one_subband(
            s->coeff_buf, &sb, step, denom_shift,
            (ctx && ctx->has_freq_tables) ? ctx->freqs[i] : NULL,
            s->sub_symbols, s->sub_escapes,
            s->rans_scratch, rans_scratch_size,
            out + C3D_CHUNK_FIXED_SIZE + entropy_pos,
            entropy_cap - entropy_pos);
        entropy_pos += bytes;
    }

    /* LOD offsets: the cumulative sizes at resolution boundaries.  Subband
     * indices 1, 8, 15, 22, 29 are the first detail subband of levels 5..1. */
    c3d_write_u32_le(lodoff_ptr + 4 * 5, c3d_read_u32_le(suboff_ptr + 4 * 1));
    c3d_write_u32_le(lodoff_ptr + 4 * 4, c3d_read_u32_le(suboff_ptr + 4 * 8));
    c3d_write_u32_le(lodoff_ptr + 4 * 3, c3d_read_u32_le(suboff_ptr + 4 * 15));
    c3d_write_u32_le(lodoff_ptr + 4 * 2, c3d_read_u32_le(suboff_ptr + 4 * 22));
    c3d_write_u32_le(lodoff_ptr + 4 * 1, c3d_read_u32_le(suboff_ptr + 4 * 29));
    c3d_write_u32_le(lodoff_ptr + 4 * 0, (uint32_t)entropy_pos);

    return C3D_CHUNK_FIXED_SIZE + entropy_pos;
}

/* Sets out[6] and out[24..40] based on ctx.  Called after c3d_prepare_chunk
 * has written placeholder values. */
static void c3d_write_ctx_header_fields(uint8_t *out, const c3d_ctx *ctx) {
    if (ctx) {
        out[6] = 1;   /* EXTERNAL */
        memcpy(out + 24, ctx->self_hash, 16);
    } else {
        out[6] = 0;
        memset(out + 24, 0, 16);
    }
}

/* Public: encode with an explicit q.  One pass, no rate control. */
size_t c3d_chunk_encode_at_q(const uint8_t *in, float q, const c3d_ctx *ctx,
                             uint8_t *out, size_t out_cap)
{
    c3d_assert(in && out);
    c3d_check_voxel_alignment(in);
    c3d_assert(out_cap >= C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(q >= C3D_Q_MIN && q <= C3D_Q_MAX);

    c3d_encoder *e = c3d_encoder_new();
    size_t r = c3d_encoder_chunk_encode_at_q(e, in, q, ctx, out, out_cap);
    c3d_encoder_free(e);
    return r;
}

size_t c3d_encoder_chunk_encode_at_q(c3d_encoder *e, const uint8_t *in, float q,
                                     const c3d_ctx *ctx,
                                     uint8_t *out, size_t out_cap)
{
    c3d_assert(e && in && out);
    c3d_check_voxel_alignment(in);
    c3d_assert(out_cap >= C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(q >= C3D_Q_MIN && q <= C3D_Q_MAX);

    /* Fixed-q path skips adaptive softness — falls back to the cached default. */
    e->has_dyn_baselines = false;

    float dc, cs;
    bool has_entropy = c3d_prepare_chunk(in, out, e, &dc, &cs);
    c3d_write_ctx_header_fields(out, ctx);
    if (!has_entropy) return C3D_CHUNK_FIXED_SIZE;
    return c3d_emit_entropy_at_q(q, e, ctx, out, out_cap);
}

/* Public: rate-controlled encode targeting `target_ratio`.
 * Uses log-space bisection on q, capped at 8 iterations.  Last attempt's
 * output is committed (may not be best if didn't converge). */
size_t c3d_chunk_encode(const uint8_t *in, float target_ratio, const c3d_ctx *ctx,
                        uint8_t *out, size_t out_cap)
{
    c3d_encoder *e = c3d_encoder_new();
    size_t r = c3d_encoder_chunk_encode(e, in, target_ratio, ctx, out, out_cap);
    c3d_encoder_free(e);
    return r;
}

size_t c3d_encoder_chunk_encode(c3d_encoder *e, const uint8_t *in,
                                float target_ratio, const c3d_ctx *ctx,
                                uint8_t *out, size_t out_cap)
{
    c3d_assert(e && in && out);
    c3d_check_voxel_alignment(in);
    c3d_assert(out_cap >= C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(target_ratio > 1.0f);

    /* Adaptive perceptual softness: fill the encoder's per-subband dynamic
     * baselines so emit / estimate use target-ratio-aware weighting.  ctx
     * overrides take precedence at the use site, so this is harmless when
     * the caller supplies a full quantizer_baseline. */
    float softness = c3d_adaptive_softness(target_ratio);
    c3d_fill_subband_baselines(softness, e->dyn_baselines);
    e->has_dyn_baselines = true;

    float dc, cs;
    bool has_entropy = c3d_prepare_chunk(in, out, e, &dc, &cs);
    c3d_write_ctx_header_fields(out, ctx);
    size_t total;
    if (!has_entropy) {
        return C3D_CHUNK_FIXED_SIZE;
    }

    double target_bytes_d = (double)C3D_VOXELS_PER_CHUNK / (double)target_ratio
                          - (double)C3D_CHUNK_FIXED_SIZE;
    if (target_bytes_d < 64.0) target_bytes_d = 64.0;

    /* Warm-start q from the previous chunk when target_ratio hasn't changed.
     * Bracket narrows to [q/4, q*4] — still wide enough to converge even if
     * chunk content varies sharply, but cuts typical iteration count in half. */
    float q, q_lo, q_hi;
    if (e->last_q > 0.0f && e->last_target_ratio == target_ratio) {
        q = e->last_q;
        q_lo = q * 0.25f;
        q_hi = q * 4.0f;
        if (q_lo < C3D_Q_MIN) q_lo = C3D_Q_MIN;
        if (q_hi > C3D_Q_MAX) q_hi = C3D_Q_MAX;
    } else {
        q = sqrtf(target_ratio) / 64.0f;
        if (q < C3D_Q_MIN) q = C3D_Q_MIN;
        if (q > C3D_Q_MAX) q = C3D_Q_MAX;
        q_lo = C3D_Q_MIN;
        q_hi = C3D_Q_MAX;
    }

    /* Bisect on the cheap estimator (quantize + Shannon, no rANS) to pick q,
     * then run the true emit exactly once.  ~3-4× encode speedup vs the
     * per-iteration full-emit loop; final output is always the real encode. */
    for (int iter = 0; iter < 10; ++iter) {
        double est_bytes = c3d_estimate_entropy_at_q(q, e, ctx);
        double err = est_bytes - target_bytes_d;
        double rel = (err < 0 ? -err : err) / target_bytes_d;
        if (rel < 0.01) break;

        if (est_bytes > target_bytes_d) {
            q_lo = q;
            float new_q = sqrtf(q_lo * q_hi);
            if (new_q <= q_lo * 1.001f) break;
            q = new_q;
        } else {
            q_hi = q;
            float new_q = sqrtf(q_lo * q_hi);
            if (new_q >= q_hi * 0.999f) break;
            q = new_q;
        }
    }
    total = c3d_emit_entropy_at_q(q, e, ctx, out, out_cap);
    e->last_q = q;
    e->last_target_ratio = target_ratio;
    return total;
}

size_t c3d_chunk_encode_max_size(void) { return C3D_CHUNK_ENCODE_MAX_SIZE; }

/* ------------------------------------------------------------------------- *
 *  §I  Chunk decoder (SELF mode)                                            *
 * ------------------------------------------------------------------------- */

/* Decodes one subband's bitstream (its full byte range), dequantizes, and
 * scatters reconstructed float coefficients into coeff_buf at the subband's
 * spatial position.
 * If `external_freqs` is non-NULL and freq_table_size == 0, uses those
 * frequencies with the provided `external_denom_shift`. */
static void c3d_decode_one_subband(
    const uint8_t *in, size_t in_size,
    float step, float alpha,
    const uint32_t *external_freqs, uint32_t external_denom_shift,
    float *coeff_buf, const c3d_subband_info *sb,
    uint8_t *sub_symbols, c3d_rans_tables *tbl_scratch)
{
    size_t n = (size_t)sb->side * sb->side * sb->side;
    size_t r = 0;

    /* freq_table_size + freq_table */
    c3d_assert(in_size >= 2);
    uint16_t ftable_bytes = c3d_read_u16_le(in + r);
    r += 2;

    /* All-zero subband sentinel (encoder's fast path): zero-fill and return. */
    if (ftable_bytes == 0xFFFFu) {
        c3d_assert(in_size == 2);
        for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
        for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
        for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x)
            coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x] = 0.0f;
        (void)step; (void)alpha;
        (void)sub_symbols; (void)tbl_scratch;
        return;
    }

    uint32_t denom_shift;
    uint32_t local_freqs[65];
    const uint32_t *freqs_to_use;
    if (ftable_bytes == 0) {
        /* EXTERNAL: require ctx to have provided tables. */
        c3d_assert(external_freqs != NULL);
        freqs_to_use = external_freqs;
        denom_shift  = external_denom_shift;
    } else {
        c3d_assert(r + ftable_bytes <= in_size);
        size_t consumed = c3d_freqs_parse(in + r, ftable_bytes, &denom_shift, local_freqs);
        c3d_assert(consumed == ftable_bytes);
        freqs_to_use = local_freqs;
        r += ftable_bytes;
    }

    /* n_symbols + rans_block_size */
    c3d_assert(r + 8 <= in_size);
    uint32_t n_symbols = c3d_read_u32_le(in + r); r += 4;
    uint32_t rans_block_size = c3d_read_u32_le(in + r); r += 4;
    c3d_assert(n_symbols == n);
    c3d_assert(r + rans_block_size <= in_size);

    c3d_rans_build_tables(tbl_scratch, denom_shift, freqs_to_use, 65);
    c3d_rans_dec_x8(in + r, rans_block_size, tbl_scratch, sub_symbols, n);
    r += rans_block_size;

    /* escape_stream spans [r..in_size). */
    const uint8_t *esc_ptr = in + r;
    size_t esc_remaining = in_size - r;

    /* Dequantize + scatter into coeff_buf. */
    size_t idx = 0;
    for (uint32_t z = sb->z0; z < sb->z0 + sb->side; ++z)
    for (uint32_t y = sb->y0; y < sb->y0 + sb->side; ++y)
    for (uint32_t x = sb->x0; x < sb->x0 + sb->side; ++x) {
        uint8_t sym = sub_symbols[idx++];
        uint32_t zigzag;
        if (sym == C3D_SYM_ESCAPE) {
            uint64_t zv = 0;
            size_t c = c3d_leb128_decode(esc_ptr, esc_remaining, &zv);
            c3d_assert(zv <= 0xffffffffull);
            zigzag = (uint32_t)zv;
            esc_ptr += c;
            esc_remaining -= c;
        } else {
            zigzag = sym;
        }
        int32_t qv = c3d_unzigzag32(zigzag);
        coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x]
            = c3d_dequant(qv, step, alpha);
    }
    c3d_assert(esc_remaining == 0);
}

void c3d_decoder_chunk_decode_lod(c3d_decoder *d,
                                  const uint8_t *in, size_t in_len, uint8_t lod,
                                  const c3d_ctx *ctx, uint8_t *out)
{
    c3d_assert(d && in && out);
    c3d_check_voxel_alignment(out);
    c3d_assert(in_len >= C3D_CHUNK_FIXED_SIZE);
    c3d_assert(lod < C3D_N_LODS);
    c3d_assert(memcmp(in, "C3DC", 4) == 0);
    uint16_t version = c3d_read_u16_le(in + 4);
    c3d_assert(version == 1);
    uint8_t context_mode = in[6];
    c3d_assert(context_mode == 0 || context_mode == 1);
    if (context_mode == 1) {
        c3d_assert(ctx != NULL);
        uint8_t id[16];
        c3d_ctx_id(ctx, id);
        c3d_assert(memcmp(id, in + 24, 16) == 0);
    }

    float dc_offset   = c3d_read_f32_le(in + 8);
    float coeff_scale = c3d_read_f32_le(in + 12);

    const uint8_t *qmul_ptr   = in + 40;
    const uint8_t *suboff_ptr = in + 40 + 144;
    const uint8_t *lodoff_ptr = in + 40 + 144 + 144;
    const uint8_t *entropy    = in + C3D_CHUNK_FIXED_SIZE;

    uint32_t lod_end = c3d_read_u32_le(lodoff_ptr + 4 * lod);
    size_t out_side = (size_t)C3D_CHUNK_SIDE >> lod;
    size_t out_vox  = out_side * out_side * out_side;

    if (lod_end == 0) {
        float v = dc_offset + 128.0f;
        int iv = (int)(v + 0.5f);
        if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
        memset(out, (uint8_t)iv, out_vox);
        return;
    }

    c3d_rans_tables tbl;
    unsigned n_sb = c3d_n_subbands_for_lod[lod];

    for (unsigned s = 0; s < n_sb; ++s) {
        c3d_subband_info sb;
        c3d_subband_info_of(s, &sb);
        float step  = c3d_read_f32_le(qmul_ptr + 4 * s);
        float alpha = (ctx && ctx->has_laplacian_alpha) ? ctx->laplacian_alpha[s]
                                                        : c3d_default_alpha(s);
        const uint32_t *ext_freqs = (ctx && ctx->has_freq_tables) ? ctx->freqs[s] : NULL;
        uint32_t ext_ds           = (ctx && ctx->has_freq_tables) ? ctx->denom_shifts[s] : 0u;

        uint32_t sub_start = c3d_read_u32_le(suboff_ptr + 4 * s);
        uint32_t sub_end;
        if (s + 1 < n_sb) {
            sub_end = c3d_read_u32_le(suboff_ptr + 4 * (s + 1));
        } else {
            sub_end = lod_end;
        }
        c3d_assert(sub_end >= sub_start);
        c3d_assert(sub_end <= in_len - C3D_CHUNK_FIXED_SIZE);

        c3d_decode_one_subband(entropy + sub_start, sub_end - sub_start,
                               step, alpha,
                               ext_freqs, ext_ds,
                               d->coeff_buf, &sb, d->sub_symbols, &tbl);
    }

    unsigned n_synth = C3D_N_DWT_LEVELS - lod;
    c3d_dwt3_inv_levels(d->coeff_buf, n_synth, d->dwt_scratch);

    /* Encoder v2: coeff_scale is already absorbed into per-subband step, so
     * dequant produces raw-magnitude coefficients.  coeff_scale in the header
     * is informational only (preserved for c3d_inspect and downstream tools). */
    (void)coeff_scale;
    for (size_t z = 0; z < out_side; ++z)
    for (size_t y = 0; y < out_side; ++y)
    for (size_t x = 0; x < out_side; ++x) {
        float v = d->coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x]
                  + dc_offset + 128.0f;
        int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
        out[z * out_side * out_side + y * out_side + x] = (uint8_t)iv;
    }
}

void c3d_decoder_chunk_decode(c3d_decoder *d, const uint8_t *in, size_t in_len,
                              const c3d_ctx *ctx, uint8_t *out)
{
    c3d_decoder_chunk_decode_lod(d, in, in_len, 0, ctx, out);
}

void c3d_chunk_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod,
                          const c3d_ctx *ctx, uint8_t *out)
{
    c3d_decoder *d = c3d_decoder_new();
    c3d_decoder_chunk_decode_lod(d, in, in_len, lod, ctx, out);
    c3d_decoder_free(d);
}

void c3d_chunk_decode(const uint8_t *in, size_t in_len, const c3d_ctx *ctx, uint8_t *out) {
    c3d_chunk_decode_lod(in, in_len, 0, ctx, out);
}

/* Cheap metadata peek — no entropy decode. */
void c3d_chunk_inspect(const uint8_t *in, size_t in_len, c3d_chunk_info *info) {
    c3d_assert(in && info);
    c3d_assert(in_len >= C3D_CHUNK_FIXED_SIZE);
    c3d_assert(memcmp(in, "C3DC", 4) == 0);
    uint16_t version = c3d_read_u16_le(in + 4);
    c3d_assert(version == 1);
    info->context_mode = in[6];
    memcpy(info->context_id, in + 24, 16);
    info->dc_offset   = c3d_read_f32_le(in + 8);
    info->coeff_scale = c3d_read_f32_le(in + 12);
    const uint8_t *lodoff = in + 40 + 144 + 144;
    for (unsigned k = 0; k < C3D_N_LODS; ++k)
        info->lod_offsets[k] = c3d_read_u32_le(lodoff + 4 * k);
}

/* Non-panicking structural check — does NOT run entropy decode. */
bool c3d_chunk_validate(const uint8_t *in, size_t in_len) {
    if (!in || in_len < C3D_CHUNK_FIXED_SIZE)             return false;
    if (memcmp(in, "C3DC", 4) != 0)                       return false;
    if (c3d_read_u16_le(in + 4) != 1)                     return false;
    if (in[6] > 1)                                         return false;   /* 0=SELF, 1=EXTERNAL */

    const uint8_t *suboff = in + 40 + 144;
    const uint8_t *lodoff = in + 40 + 144 + 144;
    uint32_t lod0 = c3d_read_u32_le(lodoff + 0);

    /* entropy region length = in_len - 352 must ≥ lod0 */
    if (in_len < (size_t)C3D_CHUNK_FIXED_SIZE + lod0)     return false;

    /* Empty chunk: all lod_offsets zero, lod0 == 0. */
    if (lod0 == 0) {
        for (unsigned k = 0; k < C3D_N_LODS; ++k)
            if (c3d_read_u32_le(lodoff + 4 * k) != 0)     return false;
        return true;
    }

    /* Check monotonic lod_offsets (lod_offset[5] ≤ [4] ≤ ... ≤ [0]). */
    uint32_t prev = 0;
    for (unsigned k = C3D_N_LODS; k-- > 0; ) {
        uint32_t v = c3d_read_u32_le(lodoff + 4 * k);
        if (v < prev) return false;
        prev = v;
    }
    if (prev != lod0) return false;

    /* Check monotonic subband_offsets. */
    prev = 0;
    for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
        uint32_t v = c3d_read_u32_le(suboff + 4 * s);
        if (v < prev) return false;
        if (v > lod0) return false;
        prev = v;
    }

    return true;
}

/* ========================================================================= *
 *  §J  Shard container (in-memory, Morton-indexed, no I/O)                  *
 * ========================================================================= *
 *
 * Shard file layout (PLAN §2.1):
 *    0       shard header (64 B)
 *    64      chunk index (4096 × 16 B = 64 KiB)  [Morton-ordered]
 *    65600   embedded .c3dx (0..65535 B, optional; ctx_offset/ctx_size in header)
 *    65600+ctx_size  chunk payloads (packed)
 *
 * In-memory representation tracks per-slot { data ptr, size, owned flag } so
 * the shard can hold a mix of borrowed (from parse) and owned (from put_chunk)
 * chunks. */

#define C3D_SHARD_HEADER_SIZE  64u
#define C3D_SHARD_INDEX_SIZE   (4096u * 16u)   /* 65 536 */
#define C3D_SHARD_PAYLOADS_MIN_OFFSET (C3D_SHARD_HEADER_SIZE + C3D_SHARD_INDEX_SIZE)  /* 65 600 */

typedef struct {
    uint64_t      size;       /* 0 for ABSENT and ZERO                        */
    uint64_t      raw_offset; /* for ABSENT: UINT64_MAX; for ZERO: 0; else arbitrary */
    const uint8_t *data;      /* NULL for ABSENT/ZERO; else points at payload */
    bool          owned;      /* true → free(data) on shard_free              */
} c3d_shard_slot;

struct c3d_shard {
    uint32_t       origin[3];
    uint8_t        shard_lod;
    c3d_shard_slot slots[4096];       /* indexed by Morton-12 of (cx, cy, cz)  */
    /* Embedded context block.  ctx_bytes is the serialized form (owned);
     * parsed_ctx is the parsed form (lazily built, owned). */
    uint8_t       *ctx_bytes;
    uint16_t       ctx_size;
    c3d_ctx       *parsed_ctx;
};

static void c3d_shard_init(c3d_shard *s, const uint32_t origin[3], uint8_t shard_lod) {
    memcpy(s->origin, origin, sizeof s->origin);
    s->shard_lod = shard_lod;
    for (unsigned i = 0; i < 4096; ++i) {
        s->slots[i].size        = 0;
        s->slots[i].raw_offset  = UINT64_MAX;
        s->slots[i].data        = NULL;
        s->slots[i].owned       = false;
    }
    s->ctx_bytes  = NULL;
    s->ctx_size   = 0;
    s->parsed_ctx = NULL;
}

c3d_shard *c3d_shard_new(const uint32_t origin[3], uint8_t shard_lod) {
    c3d_shard *s = malloc(sizeof *s);
    c3d_assert(s);
    c3d_shard_init(s, origin, shard_lod);
    return s;
}

void c3d_shard_free(c3d_shard *s) {
    if (!s) return;
    for (unsigned i = 0; i < 4096; ++i) {
        if (s->slots[i].owned) free((void *)s->slots[i].data);
    }
    free(s->ctx_bytes);
    c3d_ctx_free(s->parsed_ctx);
    free(s);
}

/* Parse helper: validates header, populates slots.  If `deep_copy`, also
 * copies each chunk's payload bytes into heap allocations (slot.owned = true). */
static void c3d_shard_parse_impl(c3d_shard *s, const uint8_t *in, size_t in_len,
                                 bool deep_copy)
{
    c3d_assert(in_len >= C3D_SHARD_HEADER_SIZE + C3D_SHARD_INDEX_SIZE);
    c3d_assert(memcmp(in, "C3DS", 4) == 0);
    uint16_t version = c3d_read_u16_le(in + 4);
    c3d_assert(version == 1);
    uint8_t shard_lod = in[6];
    uint32_t origin[3];
    origin[0] = c3d_read_u32_le(in + 8);
    origin[1] = c3d_read_u32_le(in + 12);
    origin[2] = c3d_read_u32_le(in + 16);
    uint64_t ctx_offset = c3d_read_u64_le(in + 20);
    uint16_t ctx_size   = c3d_read_u16_le(in + 28);

    c3d_shard_init(s, origin, shard_lod);

    /* Embedded ctx support deferred; if present in the input, we'll copy its
     * raw bytes so c3d_shard_serialize can emit them identically.  A future
     * step 12 will parse them into a c3d_ctx. */
    if (ctx_size > 0) {
        c3d_assert(ctx_offset >= C3D_SHARD_PAYLOADS_MIN_OFFSET);
        c3d_assert(ctx_offset + ctx_size <= in_len);
        s->ctx_size  = ctx_size;
        s->ctx_bytes = malloc(ctx_size);
        c3d_assert(s->ctx_bytes);
        memcpy(s->ctx_bytes, in + ctx_offset, ctx_size);
    }

    /* Index entries.  A PRESENT chunk has offset ≥ payloads-start-after-header.
     * We require offset + size ≤ in_len for safety. */
    const uint8_t *idx = in + C3D_SHARD_HEADER_SIZE;
    for (unsigned m = 0; m < 4096; ++m) {
        uint64_t off = c3d_read_u64_le(idx + 16 * m + 0);
        uint64_t sz  = c3d_read_u64_le(idx + 16 * m + 8);
        if (off == UINT64_MAX && sz == 0) {
            /* ABSENT */
            s->slots[m].raw_offset = UINT64_MAX;
            s->slots[m].size       = 0;
            s->slots[m].data       = NULL;
            s->slots[m].owned      = false;
        } else if (off == 0 && sz == 0) {
            /* ZERO */
            s->slots[m].raw_offset = 0;
            s->slots[m].size       = 0;
            s->slots[m].data       = NULL;
            s->slots[m].owned      = false;
        } else {
            /* PRESENT */
            c3d_assert(off >= C3D_SHARD_PAYLOADS_MIN_OFFSET);
            c3d_assert(sz > 0);
            c3d_assert(off + sz <= in_len);
            s->slots[m].raw_offset = off;
            s->slots[m].size       = sz;
            if (deep_copy) {
                uint8_t *buf = malloc(sz);
                c3d_assert(buf);
                memcpy(buf, in + off, sz);
                s->slots[m].data  = buf;
                s->slots[m].owned = true;
            } else {
                s->slots[m].data  = in + off;
                s->slots[m].owned = false;
            }
        }
    }
}

c3d_shard *c3d_shard_parse(const uint8_t *in, size_t in_len) {
    c3d_shard *s = malloc(sizeof *s);
    c3d_assert(s);
    c3d_shard_parse_impl(s, in, in_len, false);
    return s;
}
c3d_shard *c3d_shard_parse_copy(const uint8_t *in, size_t in_len) {
    c3d_shard *s = malloc(sizeof *s);
    c3d_assert(s);
    c3d_shard_parse_impl(s, in, in_len, true);
    return s;
}

/* Chunk slot accessors (Morton-indexed). */
static unsigned c3d_shard_slot_idx(uint32_t cx, uint32_t cy, uint32_t cz) {
    c3d_assert(cx < 16 && cy < 16 && cz < 16);
    return (unsigned)c3d_morton12(cx, cy, cz);
}

c3d_chunk_state c3d_shard_chunk_state(const c3d_shard *s,
                                      uint32_t cx, uint32_t cy, uint32_t cz)
{
    const c3d_shard_slot *sl = &s->slots[c3d_shard_slot_idx(cx, cy, cz)];
    if (sl->size > 0)                                return C3D_CHUNK_PRESENT;
    if (sl->raw_offset == 0 && sl->size == 0)        return C3D_CHUNK_ZERO;
    return C3D_CHUNK_ABSENT;
}

uint32_t c3d_shard_chunk_count(const c3d_shard *s, c3d_chunk_state state) {
    uint32_t n = 0;
    for (unsigned i = 0; i < 4096; ++i) {
        c3d_chunk_state st;
        const c3d_shard_slot *sl = &s->slots[i];
        if (sl->size > 0)                                st = C3D_CHUNK_PRESENT;
        else if (sl->raw_offset == 0 && sl->size == 0)   st = C3D_CHUNK_ZERO;
        else                                              st = C3D_CHUNK_ABSENT;
        if (st == state) n++;
    }
    return n;
}

const uint8_t *c3d_shard_chunk_bytes(const c3d_shard *s,
                                     uint32_t cx, uint32_t cy, uint32_t cz,
                                     size_t *out_size)
{
    const c3d_shard_slot *sl = &s->slots[c3d_shard_slot_idx(cx, cy, cz)];
    c3d_assert(sl->size > 0);   /* panic on ABSENT or ZERO per contract */
    if (out_size) *out_size = (size_t)sl->size;
    return sl->data;
}

void c3d_shard_put_chunk(c3d_shard *s,
                         uint32_t cx, uint32_t cy, uint32_t cz,
                         const uint8_t *in, size_t in_len)
{
    c3d_assert(in && in_len > 0);
    c3d_shard_slot *sl = &s->slots[c3d_shard_slot_idx(cx, cy, cz)];
    if (sl->owned) free((void *)sl->data);
    uint8_t *buf = malloc(in_len);
    c3d_assert(buf);
    memcpy(buf, in, in_len);
    sl->data       = buf;
    sl->size       = in_len;
    sl->raw_offset = 0;  /* will be recomputed on serialize */
    sl->owned      = true;
}

void c3d_shard_mark_zero(c3d_shard *s, uint32_t cx, uint32_t cy, uint32_t cz) {
    c3d_shard_slot *sl = &s->slots[c3d_shard_slot_idx(cx, cy, cz)];
    if (sl->owned) free((void *)sl->data);
    sl->data       = NULL;
    sl->size       = 0;
    sl->raw_offset = 0;  /* (0, 0) = ZERO sentinel */
    sl->owned      = false;
}

/* c3d_shard_set_ctx and c3d_shard_ctx defined in §K after c3d_ctx struct. */

size_t c3d_shard_max_serialized_size(const c3d_shard *s) {
    size_t total = C3D_SHARD_HEADER_SIZE + C3D_SHARD_INDEX_SIZE + s->ctx_size;
    for (unsigned i = 0; i < 4096; ++i) total += (size_t)s->slots[i].size;
    return total;
}

size_t c3d_shard_serialize(const c3d_shard *s, uint8_t *out, size_t out_cap) {
    size_t need = c3d_shard_max_serialized_size(s);
    c3d_assert(out_cap >= need);

    /* Header */
    memcpy(out + 0, "C3DS", 4);
    c3d_write_u16_le(out + 4, 1);
    out[6] = s->shard_lod;
    out[7] = 0;
    c3d_write_u32_le(out + 8,  s->origin[0]);
    c3d_write_u32_le(out + 12, s->origin[1]);
    c3d_write_u32_le(out + 16, s->origin[2]);

    uint64_t ctx_offset = (s->ctx_size > 0) ? C3D_SHARD_PAYLOADS_MIN_OFFSET : 0;
    c3d_write_u64_le(out + 20, ctx_offset);
    c3d_write_u16_le(out + 28, s->ctx_size);
    memset(out + 30, 0, 34);  /* reserved2 */

    /* Index (will overwrite after payloads are placed).  Start by zeroing;
     * we'll fill slot by slot as we emit payloads. */
    uint8_t *idx = out + C3D_SHARD_HEADER_SIZE;
    /* Payload cursor: starts after header + index + ctx. */
    uint64_t cursor = C3D_SHARD_PAYLOADS_MIN_OFFSET + s->ctx_size;

    /* Copy embedded ctx. */
    if (s->ctx_size > 0) {
        memcpy(out + C3D_SHARD_PAYLOADS_MIN_OFFSET, s->ctx_bytes, s->ctx_size);
    }

    for (unsigned m = 0; m < 4096; ++m) {
        const c3d_shard_slot *sl = &s->slots[m];
        if (sl->size > 0) {
            /* PRESENT */
            c3d_write_u64_le(idx + 16 * m + 0, cursor);
            c3d_write_u64_le(idx + 16 * m + 8, sl->size);
            memcpy(out + cursor, sl->data, sl->size);
            cursor += sl->size;
        } else if (sl->raw_offset == 0 && sl->size == 0) {
            /* ZERO sentinel */
            c3d_write_u64_le(idx + 16 * m + 0, 0);
            c3d_write_u64_le(idx + 16 * m + 8, 0);
        } else {
            /* ABSENT sentinel */
            c3d_write_u64_le(idx + 16 * m + 0, UINT64_MAX);
            c3d_write_u64_le(idx + 16 * m + 8, 0);
        }
    }

    return (size_t)cursor;
}

/* --- Shard convenience wrappers ------------------------------------------ */

void c3d_shard_encode_chunk(c3d_shard *s,
                            uint32_t cx, uint32_t cy, uint32_t cz,
                            const uint8_t *in, float target_ratio)
{
    uint8_t *buf = aligned_alloc(C3D_ALIGN, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_assert(buf);
    const c3d_ctx *shard_ctx = c3d_shard_ctx(s);
    size_t n = c3d_chunk_encode(in, target_ratio, shard_ctx, buf, C3D_CHUNK_ENCODE_MAX_SIZE);
    c3d_shard_put_chunk(s, cx, cy, cz, buf, n);
    free(buf);
}

void c3d_shard_decode_chunk_lod(const c3d_shard *s,
                                uint32_t cx, uint32_t cy, uint32_t cz,
                                uint8_t lod, uint8_t *out)
{
    c3d_assert(lod < C3D_N_LODS);
    const c3d_shard_slot *sl = &s->slots[c3d_shard_slot_idx(cx, cy, cz)];
    size_t out_side = (size_t)C3D_CHUNK_SIDE >> lod;
    size_t out_vox  = out_side * out_side * out_side;

    if (sl->size > 0) {
        const c3d_ctx *shard_ctx = c3d_shard_ctx(s);
        c3d_chunk_decode_lod(sl->data, (size_t)sl->size, lod, shard_ctx, out);
        return;
    }
    if (sl->raw_offset == 0 && sl->size == 0) {
        /* ZERO → fill with zero voxels. */
        memset(out, 0, out_vox);
        return;
    }
    /* ABSENT → panic per API contract. */
    c3d_panic(__FILE__, __LINE__, "c3d_shard_decode_chunk on ABSENT slot");
}

void c3d_shard_decode_chunk(const c3d_shard *s,
                            uint32_t cx, uint32_t cy, uint32_t cz,
                            uint8_t *out)
{
    c3d_shard_decode_chunk_lod(s, cx, cy, cz, 0, out);
}

/* ========================================================================= *
 *  §13  c3d_downsample_chunk_2x (box 2^3 average)                           *
 * ========================================================================= */

void c3d_downsample_chunk_2x(const uint8_t *in, uint32_t side, uint8_t *out) {
    c3d_assert(side == 256 || side == 128 || side == 64 || side == 32 || side == 16);
    uint32_t half = side / 2;
    for (uint32_t z = 0; z < half; ++z)
    for (uint32_t y = 0; y < half; ++y)
    for (uint32_t x = 0; x < half; ++x) {
        uint32_t sum = 0;
        for (uint32_t dz = 0; dz < 2; ++dz)
        for (uint32_t dy = 0; dy < 2; ++dy)
        for (uint32_t dx = 0; dx < 2; ++dx) {
            sum += in[(2*z + dz) * side * side + (2*y + dy) * side + (2*x + dx)];
        }
        /* Round to nearest, ties to even (banker's rounding). */
        uint32_t rounded = (sum + 4) >> 3;
        /* Banker's rounding on ties: if sum is odd-half (sum+4 is odd), stays same.
         * For perfectly even ties (sum mod 8 == 4), bias to even. */
        if ((sum & 7u) == 4u && (rounded & 1u)) --rounded;
        out[z * half * half + y * half + x] = (uint8_t)rounded;
    }
}

size_t c3d_ctx_max_size(void) { return 65535; }

/* ========================================================================= *
 *  §K  .c3dx context block parse/serialise + builder                         *
 * ========================================================================= *
 *
 * Wire format (PLAN §3.6):
 *    0   "C3DX"      char[4]
 *    4   version     u16       1
 *    6   block_size  u16       total including this header, ≤ 65535
 *    8   self_hash   u8[16]    c3d_hash128 of bytes [24..block_size)
 *   24   TLV records: u16 tag, u16 length_quads, u8 value[length_quads*4]
 *
 * v1 record tags:
 *   1 LAPLACIAN_ALPHA       36 × f32 in canonical subband order       (144 B = 36 quads)
 *   2 QUANTIZER_BASELINE    (not emitted by v1 builder; parser accepts)
 *   3 SUBBAND_FREQ_TABLES   concatenated 36 freq tables, 4-byte padded
 */

#define C3D_TAG_LAPLACIAN_ALPHA     1u
#define C3D_TAG_QUANTIZER_BASELINE  2u
#define C3D_TAG_SUBBAND_FREQ_TABLES 3u
/* struct c3d_ctx defined in §G above. */

/* --- Serialization helpers ----------------------------------------------- */

/* Write a TLV record header (tag + length_quads) and return the offset past
 * the header; caller then writes `length_quads * 4` bytes of value. */
static size_t c3d_write_tlv_header(uint8_t *out, size_t out_cap, size_t w,
                                    uint16_t tag, uint16_t length_quads)
{
    c3d_assert(w + 4 + (size_t)length_quads * 4 <= out_cap);
    c3d_write_u16_le(out + w + 0, tag);
    c3d_write_u16_le(out + w + 2, length_quads);
    return w + 4;
}

static size_t c3d_ctx_write_body(const c3d_ctx *ctx, uint8_t *out, size_t out_cap) {
    c3d_assert(out_cap >= 24);
    memcpy(out + 0, "C3DX", 4);
    c3d_write_u16_le(out + 4, 1);
    /* block_size and self_hash placeholder: overwritten by caller. */
    memset(out + 6, 0, 2 + 16);

    size_t w = 24;

    if (ctx->has_laplacian_alpha) {
        w = c3d_write_tlv_header(out, out_cap, w, C3D_TAG_LAPLACIAN_ALPHA, 36);
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
            c3d_write_f32_le(out + w, ctx->laplacian_alpha[i]); w += 4;
        }
    }
    if (ctx->has_quantizer_baseline) {
        w = c3d_write_tlv_header(out, out_cap, w, C3D_TAG_QUANTIZER_BASELINE, 36);
        for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
            c3d_write_f32_le(out + w, ctx->quantizer_baseline[i]); w += 4;
        }
    }
    if (ctx->has_freq_tables) {
        /* Serialize each subband's freq table back-to-back into a scratch, then
         * emit as one TLV value with 4-byte padding. */
        uint8_t *tmp = malloc(36 * 800);
        c3d_assert(tmp);
        size_t t = 0;
        for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
            t += c3d_freqs_serialise(ctx->denom_shifts[s], ctx->freqs[s],
                                     tmp + t, 36 * 800 - t);
        }
        /* Pad to 4-byte boundary. */
        size_t pad = (4u - (t & 3u)) & 3u;
        memset(tmp + t, 0, pad);
        size_t value_bytes = t + pad;
        uint16_t length_quads = (uint16_t)(value_bytes / 4u);

        w = c3d_write_tlv_header(out, out_cap, w, C3D_TAG_SUBBAND_FREQ_TABLES, length_quads);
        c3d_assert(w + value_bytes <= out_cap);
        memcpy(out + w, tmp, value_bytes);
        w += value_bytes;
        free(tmp);
    }
    return w;
}

size_t c3d_ctx_serialized_size(const c3d_ctx *ctx) {
    c3d_assert(ctx);
    return ctx->block_size;
}

size_t c3d_ctx_serialize(const c3d_ctx *ctx, uint8_t *out, size_t out_cap) {
    c3d_assert(ctx && out);
    c3d_assert(out_cap >= ctx->block_size);
    size_t sz = c3d_ctx_write_body(ctx, out, out_cap);
    c3d_assert(sz == ctx->block_size);
    c3d_write_u16_le(out + 6, (uint16_t)sz);
    memcpy(out + 8, ctx->self_hash, 16);
    return sz;
}

void c3d_ctx_id(const c3d_ctx *ctx, uint8_t out[16]) {
    c3d_assert(ctx && out);
    memcpy(out, ctx->self_hash, 16);
}

void c3d_ctx_free(c3d_ctx *ctx) { free(ctx); }

/* --- Ctx parse ----------------------------------------------------------- */

c3d_ctx *c3d_ctx_parse(const uint8_t *in, size_t in_len) {
    c3d_assert(in);
    c3d_assert(in_len >= 24);
    c3d_assert(memcmp(in, "C3DX", 4) == 0);
    uint16_t version = c3d_read_u16_le(in + 4);
    c3d_assert(version == 1);
    uint16_t block_size = c3d_read_u16_le(in + 6);
    c3d_assert(block_size >= 24 && block_size <= in_len);

    /* Verify self_hash. */
    uint8_t computed[16];
    c3d_hash128(in + 24, (size_t)block_size - 24, computed);
    c3d_assert(memcmp(computed, in + 8, 16) == 0);

    c3d_ctx *ctx = calloc(1, sizeof *ctx);
    c3d_assert(ctx);
    memcpy(ctx->self_hash, computed, 16);
    ctx->block_size = block_size;

    /* Walk TLV records. */
    size_t r = 24;
    while (r < block_size) {
        c3d_assert(r + 4 <= block_size);
        uint16_t tag = c3d_read_u16_le(in + r + 0);
        uint16_t lq  = c3d_read_u16_le(in + r + 2);
        size_t vbytes = (size_t)lq * 4u;
        c3d_assert(r + 4 + vbytes <= block_size);
        const uint8_t *v = in + r + 4;

        switch (tag) {
        case C3D_TAG_LAPLACIAN_ALPHA:
            c3d_assert(vbytes == C3D_N_SUBBANDS * 4u);
            ctx->has_laplacian_alpha = true;
            for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
                ctx->laplacian_alpha[i] = c3d_read_f32_le(v + 4 * i);
            }
            break;
        case C3D_TAG_QUANTIZER_BASELINE:
            c3d_assert(vbytes == C3D_N_SUBBANDS * 4u);
            ctx->has_quantizer_baseline = true;
            for (unsigned i = 0; i < C3D_N_SUBBANDS; ++i) {
                ctx->quantizer_baseline[i] = c3d_read_f32_le(v + 4 * i);
            }
            break;
        case C3D_TAG_SUBBAND_FREQ_TABLES: {
            ctx->has_freq_tables = true;
            size_t rr = 0;
            for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
                uint32_t ds;
                c3d_assert(rr < vbytes);
                rr += c3d_freqs_parse(v + rr, vbytes - rr,
                                      &ds, ctx->freqs[s]);
                ctx->denom_shifts[s] = ds;
            }
            c3d_assert(rr <= vbytes);
            /* Remaining vbytes - rr is zero padding; must be <4 and all-zero. */
            c3d_assert(vbytes - rr < 4);
            for (size_t k = rr; k < vbytes; ++k) c3d_assert(v[k] == 0);
            break;
        }
        default:
            /* Unknown tag: skip silently (forward-compatible). */
            break;
        }
        r += 4 + vbytes;
    }
    c3d_assert(r == block_size);
    return ctx;
}

/* --- Ctx builder --------------------------------------------------------- */

struct c3d_ctx_builder {
    uint64_t n_observed;
    /* Per-subband symbol histograms, accumulated from all observed chunks at a
     * fixed reference quantizer scalar (q_ref). */
    uint32_t histograms[C3D_N_SUBBANDS][65];
    float    q_ref;
};

/* A moderate compression point used by the builder when accumulating histograms.
 * Picking q_ref too small → many distinct symbols → tables generalise poorly.
 * Picking too large → degenerate histograms that lose detail.  0.1 is a
 * reasonable middle ground matching "moderate" compression in our tests. */
#define C3D_BUILDER_Q_REF 0.1f

c3d_ctx_builder *c3d_ctx_builder_new(void) {
    c3d_ctx_builder *b = calloc(1, sizeof *b);
    c3d_assert(b);
    b->q_ref = C3D_BUILDER_Q_REF;
    return b;
}

void c3d_ctx_builder_free(c3d_ctx_builder *b) { free(b); }

void c3d_ctx_builder_observe_chunk(c3d_ctx_builder *b, const uint8_t *in) {
    c3d_assert(b);
    c3d_check_voxel_alignment(in);

    c3d_encoder *e = c3d_encoder_new();
    uint8_t hdr_scratch[C3D_CHUNK_FIXED_SIZE];
    float dc, cs;
    bool nonempty = c3d_prepare_chunk(in, hdr_scratch, e, &dc, &cs);
    if (nonempty) {
        for (unsigned sidx = 0; sidx < C3D_N_SUBBANDS; ++sidx) {
            c3d_subband_info sb;
            c3d_subband_info_of(sidx, &sb);
            /* Match emit-time step = q * baseline * coeff_scale.  coeff_buf is
             * now in raw (un-normalised) units so we must scale step into the
             * same space. */
            float step = b->q_ref * c3d_subband_baseline(sidx) * e->coeff_scale;
            for (uint32_t z = sb.z0; z < sb.z0 + sb.side; ++z)
            for (uint32_t y = sb.y0; y < sb.y0 + sb.side; ++y)
            for (uint32_t x = sb.x0; x < sb.x0 + sb.side; ++x) {
                float c = e->coeff_buf[z * C3D_STRIDE_Z + y * C3D_STRIDE_Y + x];
                int32_t qv = c3d_quant(c, step);
                uint32_t z_unused;
                uint8_t sym = c3d_quant_to_symbol(qv, &z_unused);
                b->histograms[sidx][sym]++;
            }
        }
    }
    c3d_encoder_free(e);
    b->n_observed++;
}

c3d_ctx *c3d_ctx_builder_finish(c3d_ctx_builder *b, bool include_freq_tables) {
    c3d_assert(b);
    c3d_ctx *ctx = calloc(1, sizeof *ctx);
    c3d_assert(ctx);

    if (include_freq_tables && b->n_observed > 0) {
        ctx->has_freq_tables = true;
        for (unsigned s = 0; s < C3D_N_SUBBANDS; ++s) {
            uint32_t denom_shift = c3d_default_denom_shift(s);
            /* Bump zero bins to 1 so every symbol has nonzero probability in
             * the ctx table, but only when observations come from multiple
             * chunks.  For a single-chunk observation the ctx matches the
             * encoded chunk exactly and smoothing would only dilute freq[0]
             * (adding 64 ones to a hist that sums to 262 K steals ≈0.02 bits
             * per symbol on already-near-zero-cost subbands — a measured
             * 10× regression on highly concentrated test chunks).  With
             * multi-chunk observation the dilution is swamped by real
             * symbol diversity and fallback-avoidance wins net. */
            if (b->n_observed > 1) {
                for (unsigned k = 0; k < 65; ++k)
                    if (b->histograms[s][k] == 0) b->histograms[s][k] = 1;
            }
            c3d_normalise_freqs(b->histograms[s], denom_shift, ctx->freqs[s]);
            ctx->denom_shifts[s] = denom_shift;
        }
    }

    /* Compute block_size + self_hash by serializing to a scratch buffer with
     * a zero placeholder for self_hash, then hashing bytes [24..block_size). */
    uint8_t tmp[65535];
    size_t sz = c3d_ctx_write_body(ctx, tmp, sizeof tmp);
    c3d_assert(sz >= 24 && sz <= 65535);
    c3d_write_u16_le(tmp + 6, (uint16_t)sz);
    /* tmp[8..24) is still zero from the placeholder memset. */
    c3d_hash128(tmp + 24, sz - 24, ctx->self_hash);
    ctx->block_size = (uint16_t)sz;

    c3d_ctx_builder_free(b);
    return ctx;
}

/* --- Shard ↔ ctx wiring -------------------------------------------------- */

void c3d_shard_set_ctx(c3d_shard *s, const c3d_ctx *ctx) {
    c3d_assert(s);
    free(s->ctx_bytes);    s->ctx_bytes = NULL; s->ctx_size = 0;
    c3d_ctx_free(s->parsed_ctx); s->parsed_ctx = NULL;
    if (!ctx) return;
    size_t ctx_sz = c3d_ctx_serialized_size(ctx);
    s->ctx_bytes = malloc(ctx_sz);
    c3d_assert(s->ctx_bytes);
    c3d_ctx_serialize(ctx, s->ctx_bytes, ctx_sz);
    s->ctx_size = (uint16_t)ctx_sz;
    s->parsed_ctx = c3d_ctx_parse(s->ctx_bytes, ctx_sz);
}

const c3d_ctx *c3d_shard_ctx(const c3d_shard *s) {
    c3d_assert(s);
    if (s->parsed_ctx != NULL) return s->parsed_ctx;
    if (s->ctx_bytes != NULL) {
        /* Lazily parse for shards constructed via c3d_shard_parse (non-copy).
         * Cast away const for this cache fill; purely a memoisation. */
        c3d_shard *mut = (c3d_shard *)s;
        mut->parsed_ctx = c3d_ctx_parse(s->ctx_bytes, s->ctx_size);
        return mut->parsed_ctx;
    }
    return NULL;
}
