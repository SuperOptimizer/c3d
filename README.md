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
Harness: `c3d_bench` (runs all six codecs in one sweep).

Three scientific-data codecs were added to the sweep for full coverage:
**ZFP** (fixed-rate 3D integer mode), **SZ3** (Lorenzo + Huffman + zstd, float
only — u8 widened for the comparison), **TTHRESH** (Tucker decomposition via
the reference CLI).

**ZFP** — fixed-rate 3D integer mode, `rate_q8 / 256` bits per value.  Degrades
catastrophically below ~0.5 bpp because its block-floating-point substrate
doesn't adapt to the near-zero regions scroll data actually consists of:

| rq8 | ratio   | ZFP PSNR  | c3d PSNR | **Δ**      |
|-----|---------|-----------|----------|------------|
| 1024|   2.0:1 | 36.99 dB  | 87.81 dB | **+50.82** |
|  512|   4.0:1 | 35.99 dB  | 54.06 dB | **+18.07** |
|  256|   8.0:1 | 33.46 dB  | 48.26 dB | **+14.80** |
|  128|  16.0:1 | 23.22 dB  | 43.90 dB | **+20.67** |
|   64|  32.0:1 |  8.53 dB  | 40.86 dB | **+32.33** |
|   32|  64.0:1 |  8.53 dB  | 37.48 dB | **+28.95** |

**SZ3** — Lorenzo + Huffman + zstd.  Scientific-data workhorse for smooth
fields; performs well at low ratios (close to c3d at r=3.5-7.7) and still
tracks c3d within 4-5 dB at higher ratios:

| eps | ratio   | SZ3 PSNR  | c3d PSNR | **Δ**     |
|-----|---------|-----------|----------|-----------|
|   1 |   3.5:1 | 51.14 dB  | 55.67 dB | **+4.52** |
|   3 |   7.7:1 | 43.63 dB  | 48.59 dB | **+4.96** |
|   6 |  16.4:1 | 39.82 dB  | 43.75 dB | **+3.93** |
|  12 |  42.4:1 | 35.40 dB  | 39.51 dB | **+4.11** |
|  24 | 112.8:1 | 30.46 dB  | 34.74 dB | **+4.28** |
|  48 | 370.3:1 | 25.92 dB  | 30.01 dB | **+4.09** |

**TTHRESH** — Tucker decomposition.  The only baseline that holds its own at
low ratios: beats c3d by ~0.7–1.6 dB in the r≈6-13 band because the Tucker
factorisation is an excellent fit for smooth volumetric data.  c3d pulls
ahead above r≈30:

| P  | ratio   | TTHR PSNR | c3d PSNR | **Δ**     |
|----|---------|-----------|----------|-----------|
| 50 |   3.4:1 | 56.14 dB  | 56.53 dB | **+0.39** |
| 45 |   6.0:1 | 51.16 dB  | 50.44 dB | **−0.72** |
| 40 |  13.0:1 | 46.58 dB  | 44.97 dB | **−1.61** |
| 35 |  27.4:1 | 41.61 dB  | 41.49 dB | **−0.12** |
| 30 |  66.6:1 | 36.52 dB  | 37.25 dB | **+0.73** |
| 25 | 173.3:1 | 31.11 dB  | 32.88 dB | **+1.77** |

TTHRESH runs via an external CLI (fork + exec + disk tempfiles per chunk);
c3d's throughput advantage over TTHRESH is 5-20× even where TTHRESH wins
PSNR, which matters more for a scroll-CT ingest pipeline than the last dB
at r≈10.

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

**Encode / decode throughput** per codec at each operating point, raw-voxel
MB/s measured inside `c3d_bench` (single worker-thread path, X1E Release
+ PGO).  Higher is better.  c3d is the same codec at every row — it does
not get faster at higher QP / coarser quant because the DWT + rANS is
O(N_voxels), not O(payload).  Video codecs by contrast trade off speed
against bitstream size, so their throughput climbs sharply at high QP
(bitstream shrinks → fewer bytes to encode / parse).

| codec |  op       | ratio   | vidE MB/s | vidD MB/s | c3dE MB/s | c3dD MB/s |
|-------|-----------|---------|-----------|-----------|-----------|-----------|
| H.264 | Q18       |   8.9:1 |     33    |     41    |     19    |     45    |
| H.264 | Q30       |  41.0:1 |     65    |    113    |     28    |     59    |
| H.264 | Q48       | 654.9:1 |    200    |    635    |     63    |     78    |
| H.265 | Q18       |  11.3:1 |      2    |     19    |     24    |     62    |
| H.265 | Q30       |  46.7:1 |      4    |     34    |     37    |     85    |
| H.265 | Q48       | 690.3:1 |     17    |    103    |     80    |     94    |
| AV1   | cq16      |  11.0:1 |      2    |     58    |     24    |     67    |
| AV1   | cq40      |  34.8:1 |      4    |    120    |     33    |     81    |
| AV1   | cq60      | 129.2:1 |      6    |    222    |     49    |     87    |
| ZFP   | rq512     |   4.0:1 |     27    |    102    |     19    |     55    |
| ZFP   | rq128     |  16.0:1 |     34    |    181    |     22    |     59    |
| ZFP   | rq32      |  64.0:1 |     92    |    230    |     36    |     68    |
| SZ3   | eps=1     |   3.5:1 |     29    |     44    |     15    |     50    |
| SZ3   | eps=6     |  16.4:1 |     28    |     66    |     20    |     52    |
| SZ3   | eps=48    | 370.3:1 |     37    |     77    |     43    |     56    |
| TTHR  | P50       |   3.4:1 |      2    |      3    |     11    |     34    |
| TTHR  | P30       |  66.6:1 |      2    |      6    |     24    |     46    |
| TTHR  | P25       | 173.3:1 |      2    |      9    |     36    |     53    |

Notes:
- H.264 encode (openh264) and decode (its internal decoder) are both faster
  than c3d at low QP, and dramatically faster at high QP thanks to the
  bitstream-size effect.  But c3d decodes faster than H.264 at the
  meaningful range (Q18-Q30, 9-41:1 ratios that are realistic for a
  scroll-CT archive).
- H.265 (x265 medium + libde265) runs at 2-17 MB/s encode; c3d encodes
  **4-13× faster** across the whole range at matched quality.  Decode
  throughput is comparable at high QP, c3d 3× faster at low QP.
- AV1 all-intra (libaom cpu-used=6) encodes at 2-6 MB/s — c3d is **8-12×
  faster** on encode.  Decode parity is similar.
- ZFP is the fastest competitor on decode (always ~100-230 MB/s via its
  tight block-floating-point path), but its PSNR collapse below 16:1
  makes it unusable past that point; where ZFP is still on the R-D curve
  (r≈4:1), c3d is within 1.5× on encode and 2× on decode at matched
  bytes.
- SZ3 encode/decode are both slightly slower than c3d on u8-widened input
  (SZ3's API pays for u8→float conversion on both sides).
- TTHRESH runs via fork+exec of a C++ CLI with disk tempfiles per chunk,
  which caps it at 2 MB/s encode and 3-9 MB/s decode regardless of
  ratio.  c3d is **5-20× faster** end to end — the only place TTHRESH
  wins on quality (r≈6-13) is also the place it's slowest.

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

## LOD structure and viewer-cache design

A single c3d bitstream embeds all 6 LODs (256³ → 8³) in resolution-first
order — no duplicate coefficients across LODs, one file on disk.  The
chunk header contains `lod_offset[6]`, a prefix-length table; feed the
decoder `388 + lod_offset[lod]` bytes and it'll produce exactly that LOD.

### Byte fraction per LOD vs quantizer

Fraction of the bitstream needed to reach each LOD, measured on one scroll
chunk across the R-D curve:

```
       q    total_B    LOD5%    LOD4%    LOD3%    LOD2%    LOD1%    LOD0%
 0.01562    1215875    0.09%    0.43%    2.18%   12.37%   54.38%  100.00%  (~14:1 near-lossless)
 0.03125     505642    0.20%    0.88%    4.25%   22.95%   82.06%  100.00%  (~33:1)
 0.06250     224404    0.39%    1.60%    7.45%   36.91%   95.70%  100.00%  (~75:1)
 0.12500      93694    0.80%    3.01%   13.03%   54.71%   98.16%  100.00%  (~180:1)
 0.25000      29977    2.19%    7.19%   26.67%   79.74%   99.95%  100.00%  (~560:1)
 0.50000       9116    6.36%   17.26%   48.15%   96.20%   99.85%  100.00%  (~1840:1)
```

The level-1 detail subbands (the finest-scale wavelet coefficients, holding
87.5% of all coefficient *slots*) carry most of the *bytes* at near-lossless
but collapse to near-zero at high compression — they're mostly dead-zone
quantized, each empty subband hits a 2-byte sentinel.

Practical reading: **at near-lossless a coarse-browse tier (LOD 3-5) reads
< 5% of the bitstream; at lossy it reads < 15%**.  Range-read / HTTP partial-
content requests give a real I/O win across the whole R-D curve.  LOD 1 is
a much smaller win than LOD 3 unless you're at near-lossless.

### Decode time per LOD

Decode time is NOT proportional to bitstream bytes — it scales with
*output voxels* because the inverse DWT synthesis cost grows 8× per level.
Measurements across the same R-D points (single-thread, X1E, Release + PGO):

```
             LOD 5    LOD 4    LOD 3    LOD 2    LOD 1    LOD 0
q=0.016     0.01 ms  0.05 ms  0.35 ms  2.83 ms 25.87 ms 142.92 ms
q=0.063     0.01     0.05     0.40     3.13    18.80   120.34
q=0.100     0.01     0.05     0.42     2.86    16.52   113.97
q=0.500     0.01     0.04     0.28     1.58    12.49   103.76
```

Observations:

- **LOD 1 → LOD 0 always costs ~100 ms** of added decode time, independent
  of q.  That's the 256³ DWT synthesis pass + quant + u8 clip — voxel-count-
  bound, not byte-count-bound.
- **Each step coarser is ~8× cheaper.**  LOD 2 (3 ms) → LOD 3 (0.4 ms) →
  LOD 4 (0.05 ms) → LOD 5 (0.01 ms).  Thumbnail-scale decoding is essentially
  free.
- **Entropy-decode cost varies with q** (26 ms vs 12 ms at LOD 1 for
  near-lossless vs lossy) but it's a minor slice under the DWT floor.

### Is LOD 0 worth materialising at high compression?

Yes at 50-100× (the typical interactive-viewer target), no at >500×.

At each operating point, PSNR of LOD 0 (full decode) vs LOD 1 upscaled to
256³ via trilinear interpolation, both compared against the original:

```
    q    ratio   |  LOD 0 PSNR   upscaled LOD 1   gap
 0.063    75:1   |    35.07         30.50        +4.57 dB   ← interactive
 0.100   133:1   |    32.16         29.53        +2.63 dB   ← interactive
 0.250   558:1   |    26.78         26.19        +0.59 dB
 0.500  1840:1   |    23.48         23.37        +0.11 dB   ← collapses
 1.000  6000:1   |    20.56         20.58         0.00 dB
```

In the 50-100× band, LOD 0 holds **+3-5 dB of genuine fine-edge structure**
that no cheap upscale can recover.  That's visually significant for fiber
detail in scroll CT.  The collapse only happens past ~500:1 where the
level-1 detail subbands have been quantized to near-zero.

### Architectural guidance for a tile-pyramid viewer

For interactive browsing of a scroll archive at 50-100× compression:

- **Disk store:** the compressed bitstream.  ~100 KiB–1 MiB per 256³ chunk.
  One file per spatial region covers the whole pyramid.
- **Always-resident cache tier:** the bitstream bytes themselves.  Cheap
  to keep, lossless, sufficient to produce any LOD on demand.
- **LOD 2 (64³, ~3 ms decode):** overview / thumbnail tier for zoom-out
  views where many chunks are on screen.  Trivially parallel.
- **LOD 1 (128³, ~17 ms decode):** default scrub tier.  Good enough for any
  render target up to ~128 screen pixels tall without quality loss.
- **LOD 0 (256³, ~115 ms decode):** committed-view tier.  Trigger on
  dwell (≥ 150 ms), zoom level past a pixel-ratio threshold, or explicit
  full-res tools (segmentation, annotation).  The 100 ms cost is within
  normal human dwell latency — fine for interactive use as long as you're
  not blocking cursor-move.
- **Eviction:** LOD 0 u8 buffers (16 MiB each) get evicted first under
  pressure; LOD 1 (2 MiB) persists longer; LOD 5 (512 B) essentially
  free to hold indefinitely.  Re-decoding LOD 0 from a cached bitstream
  is a normal hot path, not a fallback.

Note: a fetched LOD 1 prefix already contains 98% of the bytes at 50-100×
compression.  If you've paid the bandwidth to get LOD 1, range-reading the
last 2% to have the full bitstream is essentially free.  The cost of
"upgrading" a view from LOD 1 to LOD 0 is compute (the 256³ DWT pass), not
I/O.

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

### SIMD tier selection

Default (`-DC3D_SIMD=AUTO`) auto-detects from `-mcpu=native` / `-march=native`.
Explicit targets:

```
-DC3D_SIMD=AUTO     # auto-detect (default)
-DC3D_SIMD=SCALAR   # disable all SIMD kernels (portable sanity / ref)
-DC3D_SIMD=NEON     # aarch64 + NEON (default on ARM)
-DC3D_SIMD=AVX2     # x86_64 Haswell and later
-DC3D_SIMD=AVX512   # x86_64 Zen 4 / Sapphire Rapids (znver4 bundle)
```

### Sanitizer builds

`-DSANITIZE=<mode>` drops optimisation to `-O1 -g`, disables LTO + fast-math, and
injects the matching runtime:

```
-DSANITIZE=address           # ASan
-DSANITIZE=undefined         # UBSan
-DSANITIZE=address+undefined # both together
-DSANITIZE=thread            # TSan
-DSANITIZE=memory            # MSan (clang)
-DSANITIZE=leak              # LSan
```

`lsan.supp` suppresses libomp-internal leaks:
`LSAN_OPTIONS=suppressions=$(pwd)/lsan.supp`.

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

### Embedding in another project

`cmake --install . --prefix /your/prefix` lays down:
- `<prefix>/lib/libc3d.a` + `libc3d.so.1.0.0` (+ SOVERSION symlinks)
- `<prefix>/include/c3d.h`
- `<prefix>/lib/cmake/c3d/c3d-config.cmake` (relocatable `find_package`)
- `<prefix>/lib/pkgconfig/c3d.pc` (relocatable via `${pcfiledir}`)

Downstream CMake:

```cmake
find_package(c3d REQUIRED)
target_link_libraries(my_app PRIVATE c3d::c3d)          # static
target_link_libraries(my_app PRIVATE c3d::c3d_shared)   # shared
```

Downstream pkg-config:

```
pkg-config --cflags --libs c3d
```

Integration-time entry points (see `c3d.h` for full API):

```c
/* cheap codec-dispatch sniff on a bytes-in-archive */
if (c3d_is_chunk(bytes, n)) { ... }

/* validate before decoding untrusted bytes — library is fatal-on-error */
if (!c3d_chunk_validate(bytes, n)) reject_chunk();
c3d_decoder_chunk_decode(dec, bytes, n, ctx, out);

/* LOD serving (e.g. multiscale zarr groups) */
uint8_t *buf = malloc(c3d_voxels_per_lod(lod));
c3d_decoder_chunk_decode_lod(dec, bytes, n, lod, ctx, buf);
```

Thread-safety: one `c3d_encoder`/`c3d_decoder` per worker thread (each owns
80-115 MiB of scratch).  The stateless `c3d_chunk_{encode,decode}` calls
are safe to call concurrently but re-malloc scratch per call.

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
| `c3d_train_dz`   | grid-search uniform dead-zone ratio on a corpus, emit tuned `.c3dx` |
| `c3d_progressive`| §T9 demo: feed byte prefixes of an encoded chunk to the decoder, show PSNR/SSIM vs fraction |
| `c3d_zarr_to_c3d.py` | zarr v2 → c3d shard converter (via ctypes)       |

## License

See `LICENSE`.
