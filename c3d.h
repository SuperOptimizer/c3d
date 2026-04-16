/* c3d — a 3D volumetric u8 compression codec for larger-than-RAM X-ray data.
 * See LICENSE.  See PLAN.md for the full design spec.  This header is the
 * canonical public API; c3d.c is the canonical implementation.
 *
 * Library-wide rules (short form — full version in PLAN.md §0):
 *   - C23, single TU (c3d.c), libc only.
 *   - In-memory API.  Library never touches disk, network, or fds.
 *   - Fatal on error: every invalid input, OOM, or parser inconsistency calls
 *     c3d_panic() which aborts.  No status codes.  Happy path is the only path.
 *   - Little-endian only (build-time static-assert).
 *   - Same-binary encode is byte-deterministic; cross-binary is not.
 */

#ifndef C3D_H
#define C3D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── fixed hierarchy constants ──────────────────────────────────────────── */

#define C3D_BLOCK_SIDE     16u     /* caller-side RAM cache granularity          */
#define C3D_CHUNK_SIDE     256u    /* codec atom: one encode/decode call         */
#define C3D_SHARD_SIDE     4096u   /* 16^3 = 4096 chunks per shard               */
#define C3D_VOXELS_PER_CHUNK ((size_t)C3D_CHUNK_SIDE * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE)

#define C3D_N_LODS         6u      /* LOD 0 (256^3) .. LOD 5 (8^3)               */
#define C3D_N_DWT_LEVELS   5u
#define C3D_N_SUBBANDS     36u     /* 1 LLL_5 + 5*7 details = 36                 */

#define C3D_ALIGN          32u     /* required alignment for raw voxel buffers   */

/* Upper bound on a c3d_chunk_encode output: raw u8 + fixed header + tables
 * + a small range-coder safety margin.  Small enough to stack-allocate.
 * (Fixed header = 388 B: 40 B base + 144 B qmul + 144 B subband_offset
 *  + 24 B lod_offset + 36 B per-subband Laplacian α.) */
#define C3D_CHUNK_ENCODE_MAX_SIZE \
    ((size_t)16 * 1024 * 1024 + 388 + 4096)

/* ─── panic / assert ─────────────────────────────────────────────────────── */

/* Default: fputs a short message to stderr, then abort().
 * Install your own to capture in tests (but hooks must not return). */
typedef void (*c3d_panic_fn)(const char *file, int line, const char *msg);
void c3d_set_panic_hook(c3d_panic_fn hook);

/* Called by every internal failure.  Does not return.
 * Hooks that attempt to longjmp or otherwise resume control are undefined
 * behaviour; library state is unrecoverable after a panic. */
_Noreturn void c3d_panic(const char *file, int line, const char *msg);

#define c3d_assert(cond)                                                      \
    do {                                                                      \
        if (!(cond)) c3d_panic(__FILE__, __LINE__, #cond);                    \
    } while (0)

/* ─── u64 voxel key ──────────────────────────────────────────────────────── */
/* Layout: [ lod:4 ][ z:20 ][ y:20 ][ x:20 ].  Planar (not Morton). */

static inline uint64_t c3d_key(uint32_t x, uint32_t y, uint32_t z, uint8_t lod) {
    return ((uint64_t)(lod & 0xfu) << 60)
         | ((uint64_t)(z & 0xfffffu) << 40)
         | ((uint64_t)(y & 0xfffffu) << 20)
         | ((uint64_t)(x & 0xfffffu));
}
static inline void c3d_unkey(uint64_t k, uint32_t *x, uint32_t *y, uint32_t *z, uint8_t *lod) {
    *x   = (uint32_t)( k         & 0xfffffu);
    *y   = (uint32_t)((k >> 20)  & 0xfffffu);
    *z   = (uint32_t)((k >> 40)  & 0xfffffu);
    *lod = (uint8_t )((k >> 60)  & 0xfu);
}

/* ─── 128-bit content hash (MurmurHash3_x64_128) ─────────────────────────── */

void c3d_hash128(const void *data, size_t len, uint8_t out[16]);

/* ─── chunk state enum ───────────────────────────────────────────────────── */

typedef enum {
    C3D_CHUNK_ABSENT  = 0,  /* shard index: (UINT64_MAX, 0) — never written    */
    C3D_CHUNK_ZERO    = 1,  /* shard index: (0, 0) — definitionally empty      */
    C3D_CHUNK_PRESENT = 2,  /* real payload exists                              */
} c3d_chunk_state;

/* ─── chunk inspection (metadata-only, fast) ─────────────────────────────── */

typedef struct {
    uint8_t  context_mode;       /* 0 = SELF, 1 = EXTERNAL                      */
    uint8_t  context_id[16];     /* zero if SELF                                */
    uint32_t lod_offsets[C3D_N_LODS];
    float    dc_offset;
    float    coeff_scale;
} c3d_chunk_info;

void c3d_chunk_inspect(const uint8_t *in, size_t in_len, c3d_chunk_info *out);

/* Non-panicking structural check: magic, version, header sizes, table offsets,
 * per-subband frame sizes, TLV bounds.  Does NOT run entropy decode — a
 * structurally-valid chunk with a bad rANS state inside will still panic on
 * actual decode. */
bool c3d_chunk_validate(const uint8_t *in, size_t in_len);

/* ─── stateless chunk codec ──────────────────────────────────────────────── */

typedef struct c3d_ctx c3d_ctx;

/* Reusable encoder/decoder scratch.  Owns ~150 MiB of buffers; create once per
 * thread, reuse across many chunks to avoid alloc/free churn (50-100 ms/chunk
 * saved).  All const-qualified member functions are safe for concurrent use. */
typedef struct c3d_encoder c3d_encoder;
typedef struct c3d_decoder c3d_decoder;

c3d_encoder *c3d_encoder_new(void);
void         c3d_encoder_free(c3d_encoder *);
c3d_decoder *c3d_decoder_new(void);
void         c3d_decoder_free(c3d_decoder *);

/* Reusable-context variants — same semantics as the stateless calls below
 * but allocate-once-reuse-many.  Recommended for any caller doing >1 chunk. */
size_t c3d_encoder_chunk_encode(c3d_encoder *, const uint8_t *in,
                                float target_ratio, const c3d_ctx *ctx,
                                uint8_t *out, size_t out_cap);
size_t c3d_encoder_chunk_encode_at_q(c3d_encoder *, const uint8_t *in,
                                     float q, const c3d_ctx *ctx,
                                     uint8_t *out, size_t out_cap);
void   c3d_decoder_chunk_decode(c3d_decoder *, const uint8_t *in, size_t in_len,
                                const c3d_ctx *ctx, uint8_t *out);
void   c3d_decoder_chunk_decode_lod(c3d_decoder *, const uint8_t *in, size_t in_len,
                                    uint8_t lod, const c3d_ctx *ctx, uint8_t *out);

/* Multi-chunk batched encode — thin wrapper around c3d_encoder_chunk_encode
 * that runs n_chunks encodes back-to-back on the same encoder and writes
 * the per-chunk output sizes to out_sizes[].  Same output as calling the
 * single-chunk API n_chunks times (warm-start kicks in after the first
 * chunk).  Exists so callers can express shard-style work in one call and
 * so future versions can overlap DWT / rANS across chunks without changing
 * the API shape.  All `inputs` must be C3D_ALIGN-aligned, all `outs` must
 * each point to a buffer of at least C3D_CHUNK_ENCODE_MAX_SIZE bytes. */
void c3d_encoder_chunks_encode(c3d_encoder *e,
                               const uint8_t *const *inputs,
                               size_t n_chunks,
                               float target_ratio, const c3d_ctx *ctx,
                               uint8_t *const *outs,
                               size_t *out_sizes);

/* Multi-chunk batched decode — analogous to the encode batch.  Each chunk's
 * encoded size is taken from in_sizes[i]; each out must be 256³ bytes. */
void c3d_decoder_chunks_decode(c3d_decoder *d,
                               const uint8_t *const *ins,
                               const size_t *in_sizes,
                               size_t n_chunks,
                               const c3d_ctx *ctx,
                               uint8_t *const *outs);

size_t c3d_chunk_encode_max_size(void);   /* returns C3D_CHUNK_ENCODE_MAX_SIZE */

/* target_ratio must be > 1.0; ctx may be NULL (→ SELF chunk), else EXTERNAL.
 * Returns bytes written.  Aborts if out_cap < C3D_CHUNK_ENCODE_MAX_SIZE. */
size_t c3d_chunk_encode(const uint8_t *in,
                        float target_ratio,
                        const c3d_ctx *ctx,
                        uint8_t *out, size_t out_cap);

/* Bypass rate control; use the given quantizer scalar q ∈ [2^-6, 2^12].
 * Useful for R-D sweeps and deterministic per-test encodes. */
size_t c3d_chunk_encode_at_q(const uint8_t *in,
                             float q,
                             const c3d_ctx *ctx,
                             uint8_t *out, size_t out_cap);

/* LOD 0 decode — writes 256^3 u8 into out. */
void c3d_chunk_decode(const uint8_t *in, size_t in_len,
                      const c3d_ctx *ctx,
                      uint8_t *out);

/* LOD decode, lod ∈ 0..5.  out must be sized (256>>lod)^3 bytes. */
void c3d_chunk_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod,
                          const c3d_ctx *ctx,
                          uint8_t *out);

/* Post-decode 2× box-average downsample for caller-side pyramids.
 * side ∈ {256, 128, 64, 32, 16}.  Writes (side/2)^3 to out.
 * Intentionally does not match the codec's internal wavelet-synthesis LODs. */
void c3d_downsample_chunk_2x(const uint8_t *in, uint32_t side, uint8_t *out);

/* ─── context block (.c3dx) ──────────────────────────────────────────────── */

size_t   c3d_ctx_max_size(void);              /* upper bound: 65535           */
size_t   c3d_ctx_serialized_size(const c3d_ctx *);  /* exact bytes for this ctx */

/* Always deep-copies.  Caller may free `in` immediately. */
c3d_ctx *c3d_ctx_parse(const uint8_t *in, size_t in_len);

size_t   c3d_ctx_serialize(const c3d_ctx *, uint8_t *out, size_t out_cap);
void     c3d_ctx_id(const c3d_ctx *, uint8_t out[16]);
void     c3d_ctx_free(c3d_ctx *);

/* Builder (single-threaded). */
typedef struct c3d_ctx_builder c3d_ctx_builder;

c3d_ctx_builder *c3d_ctx_builder_new(void);
void             c3d_ctx_builder_observe_chunk(c3d_ctx_builder *,
                                               const uint8_t *in);
c3d_ctx         *c3d_ctx_builder_finish(c3d_ctx_builder *, bool include_freq_tables);
void             c3d_ctx_builder_free(c3d_ctx_builder *);

/* ─── shard (in-memory parsed form) ──────────────────────────────────────── */

typedef struct c3d_shard c3d_shard;

c3d_shard *c3d_shard_new(const uint32_t origin[3], uint8_t shard_lod);

/* Non-copy: shard holds pointers into `in`; `in` must outlive the shard.
 *   - Mutating ops (put_chunk, mark_zero, set_ctx) allocate their own memory.
 *   - Returned pointers from c3d_shard_chunk_bytes for non-mutated chunks
 *     point into the original `in`; for mutated chunks, into shard memory. */
c3d_shard *c3d_shard_parse     (const uint8_t *in, size_t in_len);
c3d_shard *c3d_shard_parse_copy(const uint8_t *in, size_t in_len);

size_t     c3d_shard_max_serialized_size(const c3d_shard *);
size_t     c3d_shard_serialize(const c3d_shard *, uint8_t *out, size_t out_cap);
void       c3d_shard_free(c3d_shard *);

c3d_chunk_state c3d_shard_chunk_state(const c3d_shard *,
                                      uint32_t cx, uint32_t cy, uint32_t cz);
uint32_t        c3d_shard_chunk_count(const c3d_shard *, c3d_chunk_state);

/* Returns pointer into shard memory and size.  Panics on ABSENT or ZERO. */
const uint8_t  *c3d_shard_chunk_bytes(const c3d_shard *,
                                      uint32_t cx, uint32_t cy, uint32_t cz,
                                      size_t *out_size);

void c3d_shard_put_chunk  (c3d_shard *,
                           uint32_t cx, uint32_t cy, uint32_t cz,
                           const uint8_t *in, size_t in_len);
void c3d_shard_mark_zero  (c3d_shard *,
                           uint32_t cx, uint32_t cy, uint32_t cz);

/* Embedded context block (0 or 1 per shard).  set_ctx deep-copies. */
void             c3d_shard_set_ctx(c3d_shard *, const c3d_ctx *);
const c3d_ctx   *c3d_shard_ctx    (const c3d_shard *);

/* Convenience: use the shard's embedded ctx (if any) for encode/decode. */
void c3d_shard_encode_chunk     (c3d_shard *,
                                 uint32_t cx, uint32_t cy, uint32_t cz,
                                 const uint8_t *in, float target_ratio);
void c3d_shard_decode_chunk     (const c3d_shard *,
                                 uint32_t cx, uint32_t cy, uint32_t cz,
                                 uint8_t *out);
void c3d_shard_decode_chunk_lod (const c3d_shard *,
                                 uint32_t cx, uint32_t cy, uint32_t cz,
                                 uint8_t lod, uint8_t *out);

/* ─── shard-level batch helpers ──────────────────────────────────────────── *
 *
 * These wrap c3d_shard_set_ctx + per-chunk encode/decode in a single call so
 * callers who have the full shard up-front get the inter-chunk wins (per-
 * shard freq tables, LL_5 reference) without having to wire ctx training
 * themselves.  The per-chunk APIs above remain fully supported — these
 * helpers are pure additions for the shard-level workflow.
 *
 * c3d_shard_auto_train_ctx: build a ctx from a sample of training chunks
 *   and embed it in the shard.  Subsequent c3d_shard_encode_chunk / decode
 *   calls automatically use the embedded ctx.  When the training chunks
 *   are spatially close (same shard), the corpus-average LL_5 reference
 *   inside the ctx tightens the LL_5 entropy by 5-15 %.
 */
void c3d_shard_auto_train_ctx(c3d_shard *,
                              const uint8_t *const *training_chunks,
                              size_t n_training);

/* c3d_shard_encode_all: auto-train the ctx from the input chunks (or use
 * the shard's existing ctx), then encode every chunk.  `coords[i]` holds
 * (cx, cy, cz) for chunk i.  All chunks must be valid 256³ u8 buffers
 * (32-byte aligned).  After return the shard contains all chunks; serialise
 * with c3d_shard_serialize. */
void c3d_shard_encode_all(c3d_shard *,
                          const uint8_t *const *chunks,
                          const uint32_t (*coords)[3],
                          size_t n_chunks,
                          float target_ratio);

#ifdef __cplusplus
}
#endif

#endif /* C3D_H */
