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
|   5:1        |    5.0   | 51.32 dB| 130 MB/s             |
|  10:1        |   10.0   | 46.06 dB| 140 MB/s             |
|  25:1        |   25.0   | 41.45 dB| 158 MB/s             |
|  50:1        |   50.0   | 38.39 dB| 160 MB/s             |
| 100:1        |  100.3   | 34.95 dB| 165 MB/s             |
| 200:1        |  200.8   | 31.95 dB| 170 MB/s             |

c3d beats H.264-intra (`openh264` at matching byte budgets, QP 18-48) by
**+5 to +8 dB** across the full range on scroll data; see `c3d_bench` for
the side-by-side harness.

Single-chunk perf (`c3d_perf`, same hardware, q=0.10):

```
     q    enc_bytes    enc_ms  enc_MB/s    dec_ms  dec_MB/s
0.1000       142581     142.5     112.3     105.0     152.4
DWT fwd+inv (64 MiB f32): 160.5 ms/iter, 398.6 MB/s f32

LOD sweep:
lod   side   bytes_read    out_vox     dec_ms dec_MB/s_out
  5      8          759        512       0.01       37.3
  4     16         3058       4096       0.06       66.1
  3     32        13875      32768       0.46       67.9
  2     64        64524     262144       3.39       73.8
  1    128       138637    2097152      21.83       91.6
  0    256       142581   16777216     139.60      114.6
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
