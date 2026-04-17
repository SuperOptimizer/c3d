# c3d

Compression codec for very-large volumetric u8 X-ray CT data (Vesuvius-style
scroll scans up to ~75 k × 35 k × 35 k voxels). Beats H.264, H.265, and AV1
baselines by several dB at comparable byte budgets on real scroll data, and
keeps random access cheap via a fixed chunk/shard hierarchy with 6 built-in LODs.

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

**Byte-level progressive decode** at subband granularity: feeding the
decoder fewer bytes than the full chunk yields a valid reconstruction with
gracefully-degraded quality. Subband order is importance-sorted (LL_5
first, then detail subbands coarse-to-fine), so the coarsest information
is always preserved. At 50 % bytes, PSNR typically stays within 3-5 dB of
the full decode on scroll CT; at 10 %, it drops to LL-dominated quality
(~20-25 dB). Useful for streaming, bandwidth-adaptive clients, and
preview-then-refine pipelines.

Sparse sentinel: shard index slots encode `(UINT64_MAX, 0)` = ABSENT and
`(0, 0)` = ALL-ZERO so empty chunks cost zero bytes and zero decode —
critical for scrolls where 75-85 % of masked chunks are uniformly zero.

## Benchmarks

On a 64-chunk slice of `s3://scrollprize-volumes/esrf/20260311/2.4um_PHerc-Paris4_masked.zarr`
(8-thread `c3d_bench_par`, AArch64 Snapdragon X1E, Release + PGO):

| target ratio | achieved | PSNR    | aggregate throughput |
|--------------|----------|---------|----------------------|
|   5:1        |    4.9   | 51.92 dB| 132 MB/s             |
|  10:1        |    9.8   | 46.62 dB| 147 MB/s             |
|  25:1        |   24.6   | 41.89 dB| 133 MB/s             |
|  50:1        |   49.6   | 38.73 dB| 210 MB/s             |
| 100:1        |   99.4   | 35.30 dB| 238 MB/s             |
| 200:1        |  197.8   | 32.34 dB| 290 MB/s             |

c3d vs three video baselines — H.264 (`openh264`), H.265 (`x265` + `libde265`),
AV1 (`libaom`) — byte-budget-matched over the same 64-chunk corpus.  Each row
fixes a rate-control point for the video codec (QP for H.264/H.265,
`cq_level` for AV1) and retargets c3d to the resulting byte size, so both
sides hit essentially the same rate.  Δ is c3d PSNR minus baseline PSNR.

**H.264** — openh264, CABAC, per-MB adaptive QP, single-GOP 1 I + 255 P
(exploiting z-axis correlation like c3d's 3D DWT):

| QP | ratio   | H.264 PSNR | c3d PSNR | **Δ**     |
|----|---------|------------|----------|-----------|
| 18 |   8.9:1 | 43.10 dB   | 47.48 dB | **+4.37** |
| 24 |  18.0:1 | 38.93 dB   | 43.29 dB | **+4.36** |
| 30 |  41.0:1 | 35.08 dB   | 39.69 dB | **+4.61** |
| 36 | 100.3:1 | 31.03 dB   | 35.26 dB | **+4.22** |
| 42 | 258.8:1 | 27.30 dB   | 31.30 dB | **+4.00** |
| 48 | 654.9:1 | 24.04 dB   | 28.14 dB | **+4.11** |

**H.265** — x265 medium + zerolatency, 1 I + 255 P, no B-frames, CQP, libde265
decode:

| QP | ratio   | H.265 PSNR | c3d PSNR | **Δ**     |
|----|---------|------------|----------|-----------|
| 18 |  11.3:1 | 44.35 dB   | 45.77 dB | **+1.42** |
| 24 |  22.6:1 | 40.71 dB   | 42.29 dB | **+1.58** |
| 30 |  46.7:1 | 37.13 dB   | 39.07 dB | **+1.94** |
| 36 | 102.9:1 | 33.47 dB   | 35.15 dB | **+1.68** |
| 42 | 248.5:1 | 29.60 dB   | 31.44 dB | **+1.84** |
| 48 | 690.3:1 | 25.41 dB   | 28.02 dB | **+2.60** |

**AV1** — libaom `AOM_USAGE_ALL_INTRA`, `cpu-used=6`, AOM_Q.  All-intra
because libaom's altref + `lag_in_frames` pipeline drops visible frames at
flush under constant-Q; ALL_INTRA gives a clean R-D curve at the cost of
not exploiting z-axis correlation:

| cq | ratio   | AV1 PSNR  | c3d PSNR | **Δ**     |
|----|---------|-----------|----------|-----------|
| 16 |  11.0:1 | 43.24 dB  | 45.97 dB | **+2.73** |
| 28 |  18.3:1 | 39.65 dB  | 43.21 dB | **+3.56** |
| 40 |  34.8:1 | 35.40 dB  | 40.46 dB | **+5.06** |
| 48 |  55.8:1 | 32.51 dB  | 38.16 dB | **+5.65** |
| 55 |  85.7:1 | 30.07 dB  | 36.03 dB | **+5.96** |
| 60 | 129.2:1 | 28.05 dB  | 34.12 dB | **+6.07** |

c3d wins at every operating point against all three baselines — **+4.0 to +4.6
dB** over H.264, **+1.4 to +2.6 dB** over H.265 (the strongest inter
baseline), **+2.7 to +6.1 dB** over AV1 all-intra.  H.265 closes most of the
gap near-lossless; c3d pulls ahead again as compression gets aggressive.
Harness: `c3d_bench` (runs all three codecs in one sweep).

**Perceptual gap is larger than PSNR gap at high compression.**  The same
bench measures block-SSIM (8×8, mean-averaged across slices) alongside
PSNR, and the SSIM story is substantially more favourable to c3d:

| operating point       | ΔPSNR | ΔSSIM | c3d SSIM | baseline SSIM |
|-----------------------|-------|-------|----------|----------------|
| H.264 Q18 (9:1)       | +4.37 | +0.008 | 0.996   | 0.988          |
| H.264 Q48 (655:1)     | +4.11 | **+0.189** | 0.794 | 0.605      |
| H.265 Q18 (11:1)      | +1.42 | +0.002 | 0.994   | 0.991          |
| H.265 Q48 (690:1)     | +2.60 | **+0.123** | 0.791 | 0.668      |
| AV1 cq16 (11:1)       | +2.73 | +0.005 | 0.994   | 0.988          |
| AV1 cq60 (129:1)      | +6.07 | **+0.160** | 0.928 | 0.768      |

At near-lossless, SSIM saturates near 1.0 for every codec and there's
nothing to distinguish.  At 100:1+ ratios where the video codecs start
producing visible blocking artifacts, c3d's wavelet blur degrades to
0.79-0.93 SSIM (still usable) while the baselines collapse to 0.60-0.77
(visibly broken).  The +0.1-0.2 SSIM gap at high ratios is arguably a
better summary of c3d's practical advantage than the PSNR numbers above.

Single-chunk perf (`c3d_perf`, same hardware, q=0.10):

```
     q    enc_bytes    enc_ms  enc_MB/s    dec_ms  dec_MB/s
0.1000       126395     128.4     124.6     118.2     135.3
DWT fwd+inv (64 MiB f32): 161 ms/iter, 397 MB/s f32

LOD sweep:
lod   side   bytes_read    out_vox     dec_ms dec_MB/s_out
  5      8          800        512       0.01       33.5
  4     16         3095       4096       0.07       55.4
  3     32        13759      32768       0.57       54.6
  2     64        61725     262144       3.88       64.4
  1    128       123376    2097152      23.44       85.3
  0    256       126395   16777216     108.57      147.4
```

## Design constraints

- **C23**, single `c3d.c` + `c3d.h`, libc only. Video-codec baselines
  (`openh264`, `x265` + `libde265`, `libaom`) appear in the benchmark
  harness only, not in the library.
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
| `c3d_bench`      | c3d vs H.264 / H.265 / AV1 at matched byte budget, pthread-parallel (needs openh264, x265, libde265, libaom) |
| `c3d_inspect`    | dump chunk / shard / .c3dx metadata                  |
| `c3d_compact`    | parse+re-serialise (drops orphaned bytes)            |
| `c3d_train`      | build a `.c3dx` from a corpus directory              |
| `c3d_zarr_to_c3d.py` | zarr v2 → c3d shard converter (via ctypes)       |

## License

See `LICENSE`.
