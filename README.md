# c3d

Compression codec for very-large volumetric u8 X-ray CT data (Vesuvius-style
scroll scans up to ~75 k × 35 k × 35 k voxels). Beats H.264-intra (openh264
baseline) by several dB at comparable bitrates on real scroll data, and keeps
random access cheap via a fixed chunk/shard hierarchy with 6 built-in LODs.

One C23 source file + one header, libc only. No intrinsics.

## Pipeline

```
u8 256³ chunk
    ↓ f32−128, normalise
5-level 3D CDF 9/7 DWT   (separable, lifting, symmetric extension)
    ↓
per-subband dead-zone uniform quantizer (perceptual softness = 0.50)
    ↓
8-way-interleaved rANS (65-symbol alphabet: zigzag 0..63 + escape, static freq)
    ↓ bytes
```

Rate control = log-space bisection on a single chunk-level quantizer scalar;
each iteration uses a cheap Shannon-entropy estimate over the subband
histograms, and the real rANS emit runs exactly once at the chosen q.

## Hierarchy

| level     | side      | role                                                    |
|-----------|-----------|---------------------------------------------------------|
| block     | 16        | caller-side RAM cache granularity                       |
| chunk     | 256       | **codec atom** — one encode/decode call                 |
| shard     | 4 096     | 16³ = 4 096 chunks + 64 KiB Morton-ordered index        |
| subvolume | 65 536    | shard grid                                              |
| volume    | 1 048 576 | 20 bits / axis max                                      |

6 LODs (256³, 128³, 64³, 32³, 16³, 8³) decoded from prefixes of the same
bitstream — no duplicated coefficients. Coarser LODs read a tiny byte
prefix and skip IDWT levels; LOD 5 (8³) decodes in ~10 µs from a single
~760-byte prefix.

Sparse sentinel: shard index slots encode `(UINT64_MAX, 0)` = ABSENT and
`(0, 0)` = ALL-ZERO so empty chunks cost zero bytes and zero decode —
critical for scrolls where 75-85 % of masked chunks are uniformly zero.

## Benchmarks

On a 64-chunk slice of `s3://scrollprize-volumes/esrf/20260311/2.4um_PHerc-Paris4_masked.zarr`
(8-thread `c3d_bench_par`, AArch64 Snapdragon X1E, Release + PGO):

| target ratio | achieved | PSNR    | aggregate throughput |
|--------------|----------|---------|----------------------|
|   5:1        |    5.0   | 51.76 dB| 122 MB/s             |
|  10:1        |    9.9   | 46.44 dB| 140 MB/s             |
|  25:1        |   24.9   | 41.80 dB| 121 MB/s             |
|  50:1        |   49.8   | 38.64 dB| 196 MB/s             |
| 100:1        |   99.8   | 35.16 dB| 212 MB/s             |
| 200:1        |  200.5   | 32.14 dB| 240 MB/s             |

c3d vs H.264 (`openh264`, P-frames + CABAC + adaptive QP, byte-budget-matched):

| QP | ratio   | H.264 PSNR | c3d PSNR | **Δ**      |
|----|---------|-----------|----------|------------|
| 18 |   8.9:1 | 43.10 dB  | 47.32 dB | **+4.22**  |
| 24 |  18.0:1 | 38.93 dB  | 43.13 dB | **+4.20**  |
| 30 |  41.0:1 | 35.08 dB  | 39.60 dB | **+4.52**  |
| 36 | 100.3:1 | 31.03 dB  | 35.11 dB | **+4.08**  |
| 42 | 258.8:1 | 27.30 dB  | 31.14 dB | **+3.84**  |
| 48 | 654.9:1 | 24.04 dB  | 28.06 dB | **+4.03**  |

Consistent **+3.8 to +4.5 dB** advantage at every operating point on real
scroll data.  H.264 uses single-GOP P-frames (exploiting z-axis correlation
like c3d's 3D DWT), CABAC entropy coding, and per-MB adaptive QP — this is
a strong baseline, not the all-I/CAVLC configuration some codec papers use.
See `c3d_bench` for the harness.

Single-chunk perf (`c3d_perf`, same hardware, q=0.10):

```
     q    enc_bytes    enc_ms  enc_MB/s    dec_ms  dec_MB/s
0.1000       126395     113.5     141.0      99.5     160.8
DWT fwd+inv (64 MiB f32): 161.4 ms/iter, 396.5 MB/s f32

LOD sweep:
lod   side   bytes_read    out_vox     dec_ms dec_MB/s_out
  5      8          800        512       0.01       35.0
  4     16         3095       4096       0.06       67.0
  3     32        13759      32768       0.49       64.0
  2     64        61725     262144       3.22       77.7
  1    128       123376    2097152      17.82      112.2
  0    256       126395   16777216      99.46      160.9
```

## Design constraints

- **C23**, single `c3d.c` + `c3d.h`, libc only. `openh264` appears in the
  benchmark harness only, not in the library.
- **In-memory API** — bytes in → bytes out. Callers own all I/O.
- **Fatal on error** — corruption, OOM, or invalid input → `c3d_panic()` →
  `abort()`. No status codes. Panic hook is overridable.
- **Little-endian only.** Static-assert rejects BE targets.
- **Generic vectorisable C, no intrinsics.** Compiled with `-O3 -ffast-math
  -funsafe-math-optimizations -mcpu=native -flto=auto`.
- **Format version = 1 forever, no compat.** Re-encoding from raw u8 is the
  upgrade path during development.

See `PLAN.md` for the full design spec (format bytes, algorithm details,
API signatures).

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

PGO adds ~12 % on real workloads:

```
# 1. instrument
cmake -S . -B build-pgo -DPGO=GENERATE -DCMAKE_BUILD_TYPE=Release
cmake --build build-pgo -j
# 2. train on an 8-chunk mini corpus (≈90 s)
cmake --build build-pgo --target pgo_train
# 3. rebuild with the profile applied
cmake -S . -B build-pgo -DPGO=USE
cmake --build build-pgo --clean-first -j
```

## Usage

Single chunk (stateless):

```c
#include "c3d.h"

uint8_t *in  = aligned_alloc(32, 256*256*256);          /* C3D_ALIGN = 32 */
uint8_t *out = malloc(c3d_chunk_encode_max_size());
uint8_t *dec = aligned_alloc(32, 256*256*256);

size_t n = c3d_chunk_encode(in, /*target_ratio=*/25.0f, /*ctx=*/NULL,
                            out, c3d_chunk_encode_max_size());
c3d_chunk_decode(out, n, NULL, dec);

/* Partial decode to a coarser LOD (no extra work for the skipped levels). */
uint8_t *lod2 = aligned_alloc(32, 64*64*64);
c3d_chunk_decode_lod(out, n, /*lod=*/2, NULL, lod2);
```

Reusable context for many chunks (saves ~50-100 ms/chunk of alloc churn):

```c
c3d_encoder *e = c3d_encoder_new();
c3d_decoder *d = c3d_decoder_new();
for (chunk : chunks) {
    size_t n = c3d_encoder_chunk_encode(e, chunk.in, 25.0f, NULL,
                                        chunk.out, max_sz);
    c3d_decoder_chunk_decode(d, chunk.out, n, NULL, chunk.dec);
}
c3d_encoder_free(e); c3d_decoder_free(d);
```

`c3d_encoder` and `c3d_decoder` are reentrant — one per thread, concurrent
encode/decode is safe. See `c3d_bench_par.c` for a pthreads example.

### External context (`.c3dx`)

Shared frequency tables for a volume of chunks let every chunk skip its
per-subband in-band freq table (~100-200 B/chunk at moderate ratios):

```c
c3d_ctx_builder *b = c3d_ctx_builder_new();
for (sample : training_chunks) c3d_ctx_builder_observe_chunk(b, sample);
c3d_ctx *ctx = c3d_ctx_builder_finish(b, /*include_freq_tables=*/true);

/* Encode / decode with ctx.  Both sides need a matching ctx to decode. */
size_t n = c3d_encoder_chunk_encode(e, chunk, 25.0f, ctx, out, max_sz);
```

`.c3dx` blobs are max 64 KiB and can live inside a shard or as a sidecar
for streaming.

## Tools

| binary           | purpose                                              |
|------------------|------------------------------------------------------|
| `c3d_test`       | unit tests (in-tree `#include "c3d.c"`)              |
| `c3d_perf`       | single-chunk encode/decode throughput + LOD sweep    |
| `c3d_bench_par`  | multi-thread corpus bench                            |
| `c3d_bench`      | c3d vs H.264-intra at matched bitrate (needs openh264) |
| `c3d_inspect`    | dump chunk / shard / .c3dx metadata                  |
| `c3d_compact`    | parse+re-serialise (drops orphaned bytes)            |
| `c3d_train`      | build a `.c3dx` from a corpus directory              |
| `c3d_zarr_to_c3d.py` | zarr v2 → c3d shard converter (via ctypes)       |

## License

See `LICENSE`.
