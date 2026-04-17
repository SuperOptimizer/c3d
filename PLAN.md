# c3d — SOTA Design Plan

Target: one coherent state-of-the-art 3D volumetric codec, planned end-to-end, implemented end-to-end. No incremental milestones. Best-in-class compression ratio, quality, and CPU-side throughput for larger-than-RAM u8 X-ray volumes.

## 0. Non-negotiable locks

- **C23**, single `c3d.h` + `c3d.c`, libc only. Third-party video codecs (`openh264`, `x265` + `libde265`, `libaom`) are permitted in the benchmark harness only — they never link into the shipped library.
- **CDF 9/7 float32 DWT**, 5 levels, symmetric extension. No integer wavelet, no dedicated lossless mode — near-lossless is the highest-quality point on a single lossy R-D curve.
- **Per-subband rANS entropy coding** with static (per-chunk or per-shard) frequency tables. 8-way interleaved decode path. No EBCOT, no PCRD, no tier-1/tier-2 distinction.
- **Little-endian only.**
- **256³ chunks**, **4096³ shards**, **6 native LODs per chunk** shared across one embedded bitstream.
- **In-memory API.** The library never touches disk, network, or fds. Every operation is (bytes in) → (bytes out).
- **Fatal on error.** Invalid input, corruption, OOM → `c3d_panic()` → `abort()`. No status codes.
- **Rate control by `target_ratio`.** Only rate knob. `target_ratio ∈ (1.0, ∞)`; typical 2, 10, 100. Internal rate-control bisection over `chunk_scalar ∈ [2^-12, 2^12]` (widened from the originally-locked 2^-6 lower bound to accommodate perceptual subband weights, which compress HF subbands aggressively).
- **Generic vectorisable C, no intrinsics in v1.** Built with `-O3 -ffast-math -funsafe-math-optimizations`. Determinism policy: same binary + same inputs → byte-identical encoded output (same-build reproducibility). Across different builds/compilers/architectures, no guarantee — tests compare decoded voxels with tolerance, not encoded bytes.
- **Format version = 1, forever, no compat.** During development the only upgrade path is re-encoding from raw u8.
- **Permitted dev-time deps:** the benchmark harness (`c3d_bench.c`) may invoke `openh264` (H.264 baseline), `x265` + `libde265` (H.265 baseline), and `libaom` (AV1 baseline) for byte-budget-matched PSNR comparison. None are linked into any shipped binary. The rANS reference (`ryg_rans`) is public-domain and inlined into `c3d.c` rather than carried as a dep.
- **Library-wide parser rule.** Every parser (shard, chunk, `.c3dx`, TLV) bounds-checks before every read. On any inconsistency — bad magic, wrong version, offset past end, size overflow, TLV truncation, hash mismatch, sentinel byte in a data field, frequency table that does not sum to `M`, rANS state inconsistency — `c3d_panic()` immediately. No partial-recovery attempts. Multi-byte reads use `memcpy` into typed locals, not direct casts, to handle unaligned offsets (the per-subband bitstream layout has u32 fields after variable-length sections).
- **Panic reentrancy.** `c3d_panic()` must not return. Hooks that `longjmp` or otherwise resume control are undefined behaviour; library state after a panic is not recoverable.

---

## 1. Hierarchy & addressing

### 1.1 Fixed cubic pow2 hierarchy

| Level       | Side       | Role                                                   |
|-------------|------------|--------------------------------------------------------|
| block       | 16         | caller-side RAM cache granularity                      |
| chunk       | 256        | **codec atom** — one encode/decode call                |
| shard       | 4 096      | container — 16³ = 4 096 chunks + 64 KiB index          |
| subvolume   | 65 536     | shard grid                                             |
| volume      | 1 048 576  | logical max (20 bits / axis)                           |

### 1.2 6 native LODs per chunk, shared bitstream

5 levels of 3D CDF 9/7 DWT on a 256³ chunk produce these resolutions:

```
LOD 0 : 256³   full decode
LOD 1 : 128³   IDWT truncated before last synthesis level
LOD 2 :  64³
LOD 3 :  32³
LOD 4 :  16³
LOD 5 :   8³   coarsest — single LLL subband
```

All 6 LODs are decoded from **prefixes of one bitstream**. The payload is laid out resolution-first (LLL_5 first, then detail subbands at level 5, then level 4, … then level 1). Decoding to LOD k reads `lod_offset[k]` bytes and runs `5 − k` IDWT synthesis steps. No duplicated coefficients across LODs.

**Byte-level truncatable decode (§T9):** the decoder also accepts *shorter* `in_len` than the emitted chunk within a given LOD. Any subband whose entropy range extends past the supplied bytes is zero-filled; the LL and remaining HF subbands that fit are decoded normally. The effect is monotonic progressive decode at subband granularity — valid reconstruction at every truncation point, quality non-decreasing as bytes append. Useful for streaming, bandwidth-adaptive clients, and progressive preview-then-refine pipelines. Subbands are emitted biggest-magnitude-first by construction (LL_5 → level-5 details → level-4 details → … → level-1 details), so early truncation keeps the most important coefficients.

For zoom-out beyond LOD 5 (chunk thumbnails smaller than 8³), callers build a pyramid of shards at coarser resolutions. The library exposes `c3d_downsample_chunk_2x` to make each step a one-liner; these pyramid shards are independent `.c3ds` files with their `shard_lod` incremented.

### 1.3 u64 voxel key

```
bit  63                                 0
     [ lod:4 ][ z:20 ][ y:20 ][ x:20 ]
```

4 top bits = LOD (0..15; 0..5 codec-native, 6..15 reserved for caller pyramid layers). 60 bits = linear coords, planar layout (not Morton). Morton helpers exposed separately.

---

## 2. Container format

All multi-byte on-disk fields are **little-endian**. The library static-asserts `__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__` at build time; big-endian targets do not build.

### 2.1 Shard file layout

```
offset 0                    : shard header            (64 B)
offset 64                   : chunk index             (4096 × 16 B = 64 KiB)
offset 64 + 65536           : embedded context block  (optional, 0..65535 B)
offset 64 + 65536 + ctx_size: chunk payloads          (packed, append order)
```

**Shard header (64 B):**

| off | field         | type     | notes                                                   |
|-----|---------------|----------|---------------------------------------------------------|
| 0   | magic         | char[4]  | `"C3DS"`                                                |
| 4   | version       | u16      | `1`                                                     |
| 6   | shard_lod     | u8       | caller-pyramid LOD this shard represents                |
| 7   | reserved      | u8       | 0                                                       |
| 8   | shard_origin  | u32[3]   | voxel coords of shard's (0,0,0) at `shard_lod`          |
| 20  | ctx_offset    | u64      | byte offset to embedded `.c3dx`, or 0 if none           |
| 28  | ctx_size      | u16      | embedded `.c3dx` size, or 0 (matches `.c3dx block_size`) |
| 30  | reserved2     | u8[34]   | 0                                                       |
| 64  | —             | —        | end                                                     |

All fixed dimensions (`chunk_side = 256`, `shard_side = 4096`, `dtype = u8`) are v1 constants baked into the library — no header fields needed. No flags, no timestamps: if you want telemetry, carry it in your filename or a sidecar.

**Chunk index entry (16 B):**

```
 0   offset u64   absolute byte offset into shard, or sentinel
 8   size   u64   compressed payload size, or sentinel
```

**Index order: Morton / Z-order.** Chunk `(cx, cy, cz)` (each 0..15) is stored at index slot `morton12(cx, cy, cz)` — the 12-bit Morton code interleaving the 4 bits of each axis as `z3 y3 x3 z2 y2 x2 z1 y1 x1 z0 y0 x0`. Spatially adjacent chunks land at adjacent index slots, which helps prefetch on spatial queries (e.g. walking a 2D slice that touches 16 sibling chunks). Encoding is 3 `pdep` ops or a 256-entry LUT; trivially cheap. The mapping is bijective over 4 096 slots; no gaps.

**Sentinels:**

| State    | offset          | size | meaning                                         |
|----------|-----------------|------|-------------------------------------------------|
| ABSENT   | `UINT64_MAX`    | 0    | never written                                   |
| ALL_ZERO | 0               | 0    | definitionally empty                            |
| PRESENT  | `≥ 64 + 65536`  | > 0  | real payload                                    |

### 2.2 Chunk payload layout

```
 0                              : chunk header (fixed 40 B)
40                              : qmul table         (36 × f32 = 144 B)
184                             : subband_offset     (36 × u32 = 144 B)
328                             : lod_offset         ( 6 × u32 =  24 B)
352                             : entropy bitstream (resolution-first)
total_size                      :
```

**Chunk header (fixed 40 B):**

| off | field          | type     | notes                                            |
|-----|----------------|----------|--------------------------------------------------|
| 0   | magic          | char[4]  | `"C3DC"`                                         |
| 4   | version        | u16      | `1`                                              |
| 6   | context_mode   | u8       | 0 = SELF, 1 = EXTERNAL                           |
| 7   | reserved       | u8       | 0                                                |
| 8   | dc_offset      | f32      | chunk-mean removed before DWT                    |
| 12  | coeff_scale    | f32      | post-DWT coefficient normaliser                  |
| 16  | reserved2      | u8[8]    | 0                                                |
| 24  | context_id     | u8[16]   | all-zero if SELF; otherwise c3d_hash128 of ctx   |

Header + tables = **40 + 144 + 144 + 24 = 352 B** per chunk, fixed position. At very high target ratios (>1000:1) this becomes a notable fraction of the payload; acceptable and inherent to random-access design. Empty chunks (uniform-after-centering or budget-exhausted, §3.1 and §3.5) still emit the 352 B and all-zero `lod_offset`.

**Subband count.** 5 DWT levels. Levels 1..4 each contribute 7 detail subbands (their LLL recurses). Level 5 contributes 8 (7 details + 1 LLL terminal). Total = 4·7 + 8 = **36**.

### 2.3 Subband naming and canonical order

Three-letter tag orders axes as **Z, Y, X** (outermost to innermost, standard C memory layout). Letters: **L** = lowpass, **H** = highpass. `LLL` = all-lowpass (the pyramid's recursion target); `HHH` = three-way highpass (combined with the level index gives the absolute scale of the detail — `HHH_1` is the finest, `HHH_5` coarser).

Canonical storage order within the bitstream — deepest level first, then descending:

```
index  0: LLL_5
index  1: HHH_5   2: HHL_5   3: HLH_5   4: LHH_5   5: HLL_5   6: LHL_5   7: LLH_5
index  8: HHH_4   9: HHL_4  10: HLH_4  11: LHH_4  12: HLL_4  13: LHL_4  14: LLH_4
index 15: HHH_3  16: HHL_3  17: HLH_3  18: LHH_3  19: HLL_3  20: LHL_3  21: LLH_3
index 22: HHH_2  23: HHL_2  24: HLH_2  25: LHH_2  26: HLL_2  27: LHL_2  28: LLH_2
index 29: HHH_1  30: HHL_1  31: HLH_1  32: LHH_1  33: HLL_1  34: LHL_1  35: LLH_1
```

This indexing is the **canonical subband ordering**, used everywhere a per-subband array appears on disk: chunk header's `qmul[36]` and `subband_offset[36]`, `.c3dx` records `LAPLACIAN_ALPHA`, `QUANTIZER_BASELINE`, `SUBBAND_FREQ_TABLES`. Order is a storage convention — entropy coding is independent per subband (§3.4).

### 2.4 Bitstream order and LOD prefixes

Resolution-first. `lod_offset[k]` = first byte past the LOD-k prefix:

- Decode to LOD 5: read `[0, lod_offset[5])` — just `LLL_5`.
- Decode to LOD 4: read `[0, lod_offset[4])` — `LLL_5` + level-5 details.
- …
- Decode to LOD 0: read `[0, lod_offset[0])` = entire payload.

`subband_offset[i]` marks each subband's start byte within the payload, useful for inspection and future per-subband recovery.

### 2.5 Random access model

- **Chunk-level:** O(1) via shard index. Primary API.
- **LOD-level within a chunk:** first-class. `c3d_chunk_decode_lod(bytes, len, k, ctx, out)` reads only the needed prefix and runs `5 − k` synthesis levels.
- **Block-level (16³) within a chunk:** not supported. Entropy coding is chunk-global for ratio. Callers cache decoded chunks and index blocks inside the decoded buffer.

---

## 3. Compression pipeline

Per chunk, encode (decode reverses):

```
u8[256³]
  → f32 − 128 ingest                         §3.1
  → 5-level 3D CDF 9/7 DWT, symmetric        §3.2
  → uniform dead-zone quantizer              §3.3
  → per-subband rANS encode                  §3.4
  → rate control (quantizer-scalar bisection) §3.5
  → bytes
```

### 3.1 Ingest

Subtract 128 to centre the u8 distribution; then subtract the per-chunk mean (stored as `dc_offset: f32` in the chunk header). After DWT, divide coefficients by `coeff_scale = max(|c|)` so quantizer steps are dimensionless; stored in header.

**Uniform-chunk edge case.** If after centering the chunk is exactly zero (all voxels had the same value), all DWT coefficients are zero and `max(|c|) = 0`. Encoder special-cases this: write `coeff_scale = 1.0`, set `lod_offset[k] = 0` for all k, emit no entropy payload. Decoder sees `lod_offset[0] = 0` and reconstructs a uniform chunk at value `128 + dc_offset`, clamped to u8. Same code path as the high-ratio empty-chunk case in §3.5.

### 3.2 3D DWT

- CDF 9/7 lifting, float32, symmetric extension (whole-sample symmetry).
- Separable: 1D lift along X, Y, Z per level. Recurse on LLL.
- 5 levels on 256³.
- Lifting runs **in place** on the 64 MiB f32 coefficient buffer (`256³ × 4 B`). Scratch is a single 1D line (≤ 256 floats ≈ 1 KiB) per axis pass; no per-thread aux buffer beyond that.
- Implementation: plain loops over 32-byte-aligned arrays. The compiler vectorises the inner lift stages under `-O3 -ffast-math`.

### 3.3 Quantizer

- Uniform scalar dead-zone, dead zone = step. Per-subband **final** step `q_i` stored as f32 in `qmul[36]` and is what the decoder uses directly — no implicit baseline multiplication at decode time.
- Encoder computes `q_i = chunk_scalar · baseline[i]`, where `baseline[i]` is the library's fixed perceptual / synthesis-gain table (or the shard's `.c3dx QUANTIZER_BASELINE` override, if any), and `chunk_scalar` is the single knob the rate-control loop (§3.5) adjusts to hit `target_ratio`.
- `QUANTIZER_BASELINE` (when present in `.c3dx`) is an **encoder-only hint** — it replaces the library's baseline table before multiplication. Decoder never reads it; decoder only ever sees `qmul[36]`.
- Dequantizer: `c_hat = sign(q) · (|q| + α) · q_i` with α from `.c3dx LAPLACIAN_ALPHA` if present, else `α = 0.375` default. Gives the standard ~+0.5 dB over mid-tread. `LAPLACIAN_ALPHA` **is** decoder-visible.
- No vector quantization.

### 3.4 Per-subband rANS entropy coding

Each of the 36 subbands is entropy-coded **independently** with range-encoded Asymmetric Numeral Systems (rANS). Per-subband static models give tight compression (within 0.01 % of Shannon entropy for each subband's own distribution); inter-subband context modelling is left on the table — a known ~0.5-1.5 dB cost vs 3D-EBCOT on medical volumetric data, paid back by ~10× decode speed and ~5× less code.

**Symbol alphabet (65 symbols).** Quantized coefficients are mapped to symbols via **zigzag + capped magnitude**:

- Let `v` be the signed quantized coefficient and `z = zigzag(v) = (v << 1) ^ (v >> 31)` (standard unsigned zigzag).
- If `z < 64`, emit symbol `z` directly (symbols `0..63`).
- Else emit escape symbol `64`, and also write `z` to the per-subband **escape stream** as one LEB128 varint.

Covers > 99.9 % of coefficients with a direct symbol at typical quantizer steps; escape is rare.

**Frequency table** (one per subband per chunk in SELF mode):

```
 0    denom_shift  u8     log2(M); the cumulative-frequency denominator is M
 1    n_nonzero    u8     count of nonzero-frequency symbols (1..65)
 ...  n_nonzero × { u8 symbol_index, LEB128 freq }
```

v1 uses `denom_shift = 12` (M = 4096) for all detail subbands and `denom_shift = 14` (M = 16384) for `LLL_5`. A typical detail-subband table encodes in 50-200 B; `LLL_5` runs a bit more. **Invariant:** `Σ freq[i]` across the listed symbols equals `M` exactly. Parser verifies and panics on mismatch.

**Varints are LEB128 unsigned** (7-bit groups, high bit = continuation, little-endian byte order, no sign bit — the zigzag transform carries sign into the value).

**Per-subband bitstream layout:**

```
 0     freq_table_size  u16   bytes of frequency table that follow (0 if EXTERNAL)
 2     freq_table       variable (omitted entirely when size == 0)
       n_symbols        u32   quantized-coefficient count for this subband
       rans_block_size  u32   total bytes of rans_header + rans_renorm
       rans_header      32 B  8 × u32 final rANS states (the 8 interleaved streams)
       rans_renorm      variable   the renormalisation byte stream, read forward
       escape_stream    variable   concatenated LEB128 varints, one per symbol-64 emission
```

All u16/u32 fields are little-endian; the post-`freq_table` fields land at arbitrary alignment, so the library reads them via `memcpy`.

- `rans_block_size` lets the decoder find where `escape_stream` begins; `escape_stream` runs to the end of the subband payload (inferred from `subband_offset[i+1]` or from the enclosing chunk boundary for the last subband).
- **Escape count is implicit**: decoder reads varints in order as it encounters symbol-64 emissions during rANS decode. If the escape stream is exhausted prematurely or leaves unread bytes, parser panics.
- **Degenerate case** (allowed, no special-case needed): if one symbol has `freq = M`, rANS's state never renormalises and `rans_renorm` is exactly 0 bytes. The 32 B `rans_header` still appears. A subband encoded this way costs ~40 B + frequency table.

**rANS engine.** Follows Fabian Giesen's `ryg_rans_byte` architecture (public-domain reference, ~200 scalar LOC, inlined into `c3d.c` — no separate dependency). Encoder: 8 rANS states interleaved, symbols dealt round-robin across the 8 streams, renormalisation bytes appended to a shared growing buffer in the order renormalisations occur. Final 8 u32 states written as the 32 B `rans_header`; the growing renorm buffer becomes `rans_renorm`. Decoder reads the 8 states, then reads `rans_renorm` forward, advancing 8 symbols per inner iteration.

**Expected throughput.** Pure scalar (autovectorised by the compiler on the 8-way state machine): **~200-300 MB/s** decode. The often-cited **540-750 MB/s** numbers come from hand-written SIMD intrinsics on the 8 states; that's a v2 lever, not v1. Scalar throughput still comfortably beats any EBCOT-class coder's ~50 MB/s.

See §3.6 for SELF vs EXTERNAL mode and how frequency tables can alternatively live in `.c3dx SUBBAND_FREQ_TABLES` instead of in-band.

### 3.5 Rate control

No PCRD-opt. `target_ratio` is hit by choosing the quantizer scalar (§3.3). The chunk bitstream is exactly whatever rANS produces at that step — no post-hoc truncation.

**Algorithm (bisection with Laplacian warm-start):**

1. Compute `target_bytes = 256³ / target_ratio − 352` (subtracting fixed header + tables).
2. Warm-start: from per-subband L1 norms (cheap, measured during DWT), estimate the initial quantizer scalar `q₀` under a Laplacian-tail model. Closed-form: given subband variances and a target rate in bits/voxel, `q₀ ∝ 2^(−R̄/6)`. Warm-starts typically land within 20 % of target on the first pass.
3. Quantize all 36 subbands and run rANS. Measure `actual_bytes`.
4. If `|actual_bytes − target_bytes| > 2 %`, bisect on `q` and re-run quantize+rANS. Warm-started bisection converges in 3-5 iterations on typical data. DWT is **not** re-run — it runs once per chunk and the coefficient buffer is kept.

**Search range for `chunk_scalar`.** Bisection is confined to `q ∈ [2^-6, 2^12]` (≈ 1/64 to 4096). Warm-start clamps to this range. If the bisection walks into an endpoint and still can't meet `target_bytes`, the encoder accepts whatever size falls out at the endpoint — at `q = 2^-6` (min) that's near-lossless; at `q = 2^12` (max) that's the empty-chunk path. Cap of 8 bisection iterations regardless; whatever `q` is current at that point is used.

Per iteration: 36 × (quantize + rANS encode) over ~64 MiB of coefficients ≈ ~50 ms at scalar rANS encode speed. Typical rate-controlled encode: 150-300 ms per chunk. Rate-uncontrolled ("use this quantizer, whatever size falls out"): one pass, ~40-80 ms per chunk.

**Bitstream emission.** Subbands are written in canonical order (§2.3) into the entropy payload. `subband_offset[i]` and `lod_offset[k]` are filled in as bytes are produced. No truncation, no packet headers, no tag-trees.

**High-ratio / empty-chunk path.** If `q` grows large enough that every quantized coefficient of every subband is zero: emit a chunk whose `lod_offset[k] = 0` for all k. Decoder reconstructs from `dc_offset` alone. Same path as §3.1's uniform-chunk case. Threshold is data-dependent but typically above `target_ratio ≈ 10 000-40 000` depending on the chunk's dynamic range.

### 3.6 External context block (`.c3dx`)

Every chunk is SELF or EXTERNAL per its `context_mode` byte.

- **SELF (0):** standalone; decode uses the library default for α (0.375) and reads per-subband frequency tables in-band. `context_id` is zero.
- **EXTERNAL (1):** decoder must have a context block whose `c3d_hash128` equals the chunk's `context_id`. Missing context → `c3d_panic`.

Context blocks live in one of two places:

1. **Inside the shard** — shard header's `ctx_offset` / `ctx_size` locate the block between the index and the chunk payloads. Only one per shard.
2. **As a sidecar byte buffer** — caller ships it alongside chunks (streaming, loose chunk files).

**Context block format (max 65 535 B total, `block_size` is u16):**

| off | field       | type      | notes                                             |
|-----|-------------|-----------|---------------------------------------------------|
| 0   | magic       | char[4]   | `"C3DX"`                                          |
| 4   | version     | u16       | `1`                                               |
| 6   | block_size  | u16       | total bytes of this block (incl. header), ≤ 65535 |
| 8   | self_hash   | u8[16]    | `c3d_hash128` of bytes [24..block_size)           |
| 24  | param_records | variable | TLV records packed back-to-back                   |

**TLV record format:** `u16 tag` + `u16 length_quads` (length in 4-byte units) + `u8 value[length_quads*4]`. All records are 4-byte aligned. Unknown tags are skipped by the decoder. `c3d_ctx_parse` panics on `self_hash` mismatch.

**v1 record tags:**

| tag | name              | value                                                             |
|-----|-------------------|-------------------------------------------------------------------|
| 1   | LAPLACIAN_ALPHA   | 36 × f32 per-subband α for dequantizer offset                     |
| 2   | QUANTIZER_BASELINE| 36 × f32 per-subband step multipliers                             |
| 3   | SUBBAND_FREQ_TABLES | 36 concatenated frequency tables in canonical subband order (§2.3), each encoded exactly as the per-subband in-band format (§3.4): `u8 denom_shift, u8 n_nonzero, n_nonzero × {u8 symbol, varint freq}`. Values stream directly; no per-table length prefix inside the record (reader consumes by walking `n_nonzero`). |

`LAPLACIAN_ALPHA` and `QUANTIZER_BASELINE` values are in canonical subband order: `value[0] = LLL_5`, `value[1] = HHH_5`, … `value[35] = LLH_1`.

No `CONTEXT_PRIORS` or `CODEBLOCK_SHAPES` records in v1 — rANS has no contexts, no code-blocks. Tags 4 and above are reserved.

**Hash.** `c3d_hash128` is **MurmurHash3_x64_128** (~60 lines of trivial public-domain C). Non-cryptographic; sufficient for content addressing of small blobs in a non-adversarial local setting. (Earlier drafts specified XXH3-128, but a faithful XXH3-128 impl is ~500 lines and the distribution advantage over MurmurHash3 is irrelevant at our blob sizes.)

**No corpus-baked priors.** The only default hard-coded in `c3d.c` is `α = 0.375` and the fixed per-subband perceptual/synthesis-gain baseline. Corpus-learned parameters travel as `.c3dx` data, never as compile-time constants.

**Producing `.c3dx`.** Three modes:

- **Fast single-pass (default).** Emit SELF chunks, no `.c3dx`.
- **Shard-assisted single-pass.** Accumulate per-subband statistics during encoding (running L1 norm for α, subband energies for quantizer baseline); write a `.c3dx` with `LAPLACIAN_ALPHA` + `QUANTIZER_BASELINE`. Chunks mark EXTERNAL. No encode-speed cost of note.
- **Shard-assisted two-pass.** As above, plus gather per-subband coefficient histograms across all chunks in pass 1 and emit a shard-global `SUBBAND_FREQ_TABLES` record; pass 2 re-encodes using those tables (EXTERNAL-mode chunks, `freq_table_size = 0`). ~1.4× encode time, saves 1-5 KB per chunk and ~1-3 % ratio on homogeneous shards by using a better-fitted global prior.
- **Offline `c3d_train` tool.** Takes a directory of raw 256³ chunks, outputs a `.c3dx`. Useful for reusing a single prior across many shards of similar data.

---

## 4. LOD semantics (precise)

- 6 LODs per chunk: LOD 0 = 256³ u8, LOD k = (256 >> k)³ u8 for k ∈ 1..5.
- Decoding to LOD k reads `lod_offset[k]` payload bytes, runs `5 − k` IDWT synthesis levels, outputs `(256 >> k)³` voxels.
- A LOD k reconstruction is the inverse DWT of the (possibly truncated) coarsest k+1 subband sets. It is **not** bit-equal to filtering LOD 0 and stride-2 subsampling k times; it is the natural wavelet-pyramid low-pass, which is close-but-not-identical.
- Callers wanting further zoom-out build pyramid shards via `c3d_downsample_chunk_2x`. Pyramid shards increment `shard_lod`.

---

## 5. Public API (C23, all in-memory, fatal-on-error)

```c
// ─── types & conventions ───────────────────────────────────────────────────
// Alignment:
//   - Raw voxel buffers (256³ u8 inputs, LOD out buffers, 256³ u8 outputs):
//     must be 32-byte aligned.
//   - Encoded byte buffers (chunk payloads, shard bytes, .c3dx blobs):
//     no alignment requirement; may come straight from a network socket.
//
// All failures call c3d_panic() → aborts. No status returns.
// const-qualified calls are safe for concurrent callers; mutating calls require
// external synchronisation per object.
//
// c3d_assert(cond):
//   #define c3d_assert(x) do { if (!(x)) c3d_panic(__FILE__, __LINE__, #x); } while (0)
// c3d_panic default: fputs to stderr, then abort(). Override via c3d_set_panic_hook.

typedef struct c3d_shard   c3d_shard;    // opaque, parsed shard in memory
typedef struct c3d_ctx     c3d_ctx;      // opaque, parsed .c3dx
typedef struct c3d_ctx_builder c3d_ctx_builder;

typedef enum { C3D_CHUNK_ABSENT, C3D_CHUNK_ZERO, C3D_CHUNK_PRESENT } c3d_chunk_state;

// c3d_panic: caller-installable diagnostic hook. Not thread-safe — install
// before issuing any c3d_* call from another thread.
typedef void (*c3d_panic_fn)(const char *file, int line, const char *msg);
void c3d_set_panic_hook(c3d_panic_fn);

// ─── voxel key ─────────────────────────────────────────────────────────────
static inline uint64_t c3d_key  (uint32_t x, uint32_t y, uint32_t z, uint8_t lod);
static inline void     c3d_unkey(uint64_t k, uint32_t *x, uint32_t *y, uint32_t *z, uint8_t *lod);

// ─── reusable encoder/decoder contexts ────────────────────────────────────
// Each context owns ~115 MiB (encoder) / ~80 MiB (decoder) of scratch.
// Create once per thread and reuse across many chunks to avoid 50-100 ms of
// alloc/free churn per call.  Const-qualified calls are reentrant.
typedef struct c3d_encoder c3d_encoder;
typedef struct c3d_decoder c3d_decoder;
c3d_encoder *c3d_encoder_new(void);
void         c3d_encoder_free(c3d_encoder *);
c3d_decoder *c3d_decoder_new(void);
void         c3d_decoder_free(c3d_decoder *);
size_t       c3d_encoder_chunk_encode      (c3d_encoder *, const uint8_t *in,
                                            float target_ratio, const c3d_ctx *,
                                            uint8_t *out, size_t out_cap);
size_t       c3d_encoder_chunk_encode_at_q (c3d_encoder *, const uint8_t *in,
                                            float q, const c3d_ctx *,
                                            uint8_t *out, size_t out_cap);
void         c3d_decoder_chunk_decode      (c3d_decoder *, const uint8_t *in,
                                            size_t in_len, const c3d_ctx *,
                                            uint8_t *out);
void         c3d_decoder_chunk_decode_lod  (c3d_decoder *, const uint8_t *in,
                                            size_t in_len, uint8_t lod,
                                            const c3d_ctx *, uint8_t *out);

// ─── stateless chunk codec (creates a temp encoder/decoder each call) ─────
// Upper bound = 256³ + 352 (header/tables) + 4096 (range-coder safety margin).
// #define C3D_CHUNK_ENCODE_MAX_SIZE (16*1024*1024 + 352 + 4096)
// Small enough that a caller can stack-allocate or use a static buffer.
size_t c3d_chunk_encode_max_size(void);

// target_ratio must be > 1.0; ≤ 1.0 → c3d_panic. ctx may be NULL → SELF chunk.
// Returns bytes written. Aborts if out_cap < c3d_chunk_encode_max_size().
size_t c3d_chunk_encode(const uint8_t in[256*256*256],
                        float target_ratio,
                        const c3d_ctx *ctx,
                        uint8_t *out, size_t out_cap);

// Debug/bench sibling: bypass rate control, use the given quantizer scalar
// directly. q must lie in [2^-6, 2^12]; outside → panic. Useful for R-D
// benchmarking ("sweep q, plot the curve") and deterministic per-test encodes.
size_t c3d_chunk_encode_at_q(const uint8_t in[256*256*256],
                             float q,
                             const c3d_ctx *ctx,
                             uint8_t *out, size_t out_cap);

// Non-panicking validation for inspection tools and fuzzers. Returns true iff
// the buffer parses as a valid chunk (magic, version, header sizes, table
// structure, rANS frame sizes). Does NOT run entropy decode, so it catches
// structural corruption but not ALL corruption — a fully-structured chunk with
// a bad rANS state inside will still panic when decoded.
bool c3d_chunk_validate(const uint8_t *in, size_t in_len);

// LOD 0 decode.
void c3d_chunk_decode(const uint8_t *in, size_t in_len,
                      const c3d_ctx *ctx,
                      uint8_t out[256*256*256]);

// LOD decode. lod ∈ 0..5. out must be sized (256>>lod)³ bytes.
void c3d_chunk_decode_lod(const uint8_t *in, size_t in_len, uint8_t lod,
                          const c3d_ctx *ctx,
                          uint8_t *out);

// Post-decode 2× downsample helper for caller-side pyramids.
// `side` must be one of {256, 128, 64, 32, 16}; any other value panics.
// Writes (side/2)³ out from side³ in.
// Filter: box average over 2³ voxels (rounded to nearest, ties to even).
// Intentionally simple; does not match the codec's internal wavelet-synthesis
// LODs and is not meant to. Purely for caller-side shard-pyramid construction.
void c3d_downsample_chunk_2x(const uint8_t *in, uint32_t side, uint8_t *out);

// Metadata-only inspect.
typedef struct {
    uint8_t  context_mode;
    uint8_t  context_id[16];
    uint32_t lod_offsets[6];
    float    dc_offset;
    float    coeff_scale;
} c3d_chunk_info;
void c3d_chunk_inspect(const uint8_t *in, size_t in_len, c3d_chunk_info *out);

// ─── context block (.c3dx) ─────────────────────────────────────────────────
// Upper bound on any .c3dx blob: 65535 B (the u16 block_size cap). Actual
// ctx blobs are typically ~500-1500 B.
size_t   c3d_ctx_max_size(void);                             // returns 65535
// Exact serialised size for this ctx (= its block_size field).
size_t   c3d_ctx_serialized_size(const c3d_ctx *);

// Always deep-copies the incoming bytes. Caller may free `in` immediately.
// Panics on magic/version/hash mismatch or TLV truncation.
c3d_ctx *c3d_ctx_parse(const uint8_t *in, size_t in_len);
// out_cap must be ≥ c3d_ctx_serialized_size(ctx) or panics. (Callers who don't
// want to pre-query can pass a buffer of c3d_ctx_max_size() bytes; wasteful
// but safe.)
size_t   c3d_ctx_serialize(const c3d_ctx *, uint8_t *out, size_t out_cap);
// Returns the BLAKE- err, XXH3-128-truncated self_hash stored in the ctx.
void     c3d_ctx_id(const c3d_ctx *, uint8_t out[16]);
void     c3d_ctx_free(c3d_ctx *);

// Builder: single-threaded, accumulates stats from chunks, finalises into a
// c3d_ctx. For parallel accumulation, use multiple builders and merge via a
// future c3d_ctx_builder_merge (not in v1) — or just feed chunks sequentially.
//
// observe_chunk accumulates per-subband stats (L1 norms, energies) and — when
// include_freq_tables is later requested — per-subband coefficient histograms
// (after running DWT + quantize at a baseline step). It does NOT run rANS.
// finish() computes the ctx's self_hash from the final serialised bytes and
// stores it; c3d_ctx_id returns this hash afterwards.
c3d_ctx_builder *c3d_ctx_builder_new(void);
void             c3d_ctx_builder_observe_chunk(c3d_ctx_builder *, const uint8_t in[256*256*256]);
// include_freq_tables=true emits a SUBBAND_FREQ_TABLES record built from the
// accumulated histograms. Observation cost is ~1.4× vs stats-only.
c3d_ctx         *c3d_ctx_builder_finish(c3d_ctx_builder *, bool include_freq_tables);
void             c3d_ctx_builder_free(c3d_ctx_builder *);

// ─── shard (in-memory parsed form) ─────────────────────────────────────────
c3d_shard *c3d_shard_new(const uint32_t origin[3], uint8_t shard_lod);

// Non-copy: shard holds pointers into `in`; `in` must outlive the shard.
c3d_shard *c3d_shard_parse     (const uint8_t *in, size_t in_len);
// Copy: shard owns its bytes.
c3d_shard *c3d_shard_parse_copy(const uint8_t *in, size_t in_len);

// Max serialised size = 64 + 65536 + ctx_size + Σ chunk payload sizes.
size_t     c3d_shard_max_serialized_size(const c3d_shard *);
size_t     c3d_shard_serialize(const c3d_shard *, uint8_t *out, size_t out_cap);
void       c3d_shard_free(c3d_shard *);

// Chunk ops on a shard.
c3d_chunk_state c3d_shard_chunk_state(const c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz);

// Count chunks in a given state across the 4096-slot index. O(4096), cheap.
uint32_t c3d_shard_chunk_count(const c3d_shard *, c3d_chunk_state);

// Returns a pointer into shard memory and size. Aborts if ABSENT or ZERO.
const uint8_t *c3d_shard_chunk_bytes(const c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz,
                                     size_t *out_size);

// Installs a pre-encoded payload. Shard copies.
void c3d_shard_put_chunk(c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz,
                         const uint8_t *in, size_t in_len);

void c3d_shard_mark_zero(c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz);

// Embedded context block (0 or 1 per shard).
// set_ctx: shard deep-copies from the input. Caller may free immediately.
// c3d_shard_ctx: pointer valid until c3d_shard_free or the next set_ctx.
// When a shard is produced via c3d_shard_parse (non-copy) with an embedded
// .c3dx inside the input bytes, the shard still deep-parses the ctx into its
// own heap allocation. Ctx lifetime is never tied to the input buffer.
void             c3d_shard_set_ctx(c3d_shard *, const c3d_ctx *);
const c3d_ctx   *c3d_shard_ctx    (const c3d_shard *);

// Convenience wrappers. These use the shard's embedded ctx (if any) as the
// decode ctx. If a chunk is EXTERNAL and either (a) the shard has no ctx or
// (b) the shard's ctx_id ≠ chunk's context_id, c3d_panic. Callers who need
// to decode with a sidecar .c3dx should drop to the stateless path:
//   const uint8_t *b = c3d_shard_chunk_bytes(s, cx, cy, cz, &n);
//   c3d_chunk_decode(b, n, caller_provided_ctx, out);
void c3d_shard_encode_chunk     (c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz,
                                 const uint8_t in[256*256*256], float target_ratio);
// Decode — ABSENT → c3d_panic; ZERO → fills out with zeros (respects LOD size).
void c3d_shard_decode_chunk    (const c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz,
                                 uint8_t out[256*256*256]);
void c3d_shard_decode_chunk_lod(const c3d_shard *, uint32_t cx, uint32_t cy, uint32_t cz,
                                uint8_t lod, uint8_t *out);
```

---

## 6. Implementation topology

One TU. No runtime dispatch, no intrinsics, no feature flags beyond `C3D_BUILD_REF` (defined by `c3d_test.c` to compile in the reference decoder path).

```
c3d.c
├── §A  types, c3d_panic, c3d_assert, bit-io primitives, alignment, Morton-12 LUT
├── §B  c3d_hash128 (MurmurHash3_x64_128, public domain)
├── §C  rANS engine (encode, decode, 8-way interleaved, ryg_rans-style)
├── §D  frequency-table build + serialise/parse (per-subband)
├── §E  CDF 9/7 lifting, separable 3D, symmetric extension
├── §F  quantizer + Laplacian dequant + zigzag+escape symbol mapping
├── §G  rate-control loop (Laplacian warm-start, bisection on quantizer scalar)
├── §H  chunk encoder pipeline (DWT → quantize → rANS → pack)
├── §I  chunk decoder pipeline + LOD partial decoder
├── §J  shard parse/serialise (in-memory, Morton-indexed)
├── §K  .c3dx parse/serialise + builder
├── §L  public API wrappers
```

Rough size budget: A 200, B 150, C 500, D 200, E 500, F 200, G 300, H 250, I 250, J 400, K 400, L 300 → **~3650 lines**. Ceiling 5000 including comments. The entropy stage shrank from ~3400 to ~700; the plan easily fits in a single TU.

**Reference decoder.** Compiled-in only when `C3D_BUILD_REF` is defined. `c3d_test.c` defines the macro and links against a test build; shipped library builds don't. Keeps the release TU ~15 % smaller.

When enabled, the main TU exposes internal `c3d_chunk_decode_ref()` and `c3d_chunk_decode_lod_ref()` functions alongside the fast paths. The reference versions use clarity-over-speed implementations of §E/§F/§G (DWT, quantizer, rate control don't apply to decode — the ref path is DWT-inverse + dequant + rANS-decode with canonical FP evaluation order, no loop-carried reuse tricks, extra `c3d_assert`s throughout). They live in the same TU as the fast path so there is exactly one copy of the format, parsing, and tables.

Under `-O3 -ffast-math`, the compiler is free to reassociate FP ops. To keep the reference path meaningful, each ref function is annotated:

```c
#if defined(__GNUC__) || defined(__clang__)
  #define C3D_REF_ATTR __attribute__((optimize("no-fast-math","no-associative-math")))
#else
  #define C3D_REF_ATTR
#endif
static C3D_REF_ATTR void c3d_ref_idwt_1d(float *line) { … }
```

**Compiler caveat.** `__attribute__((optimize("no-fast-math","no-associative-math")))` is honoured by **GCC**; Clang accepts the syntax but silently ignores the option strings. Consequence: under Clang builds, the reference path is reassociated exactly like the fast path, so fast-vs-ref divergence is ~always zero — the cross-check is trivially satisfied but tells you nothing. **The fast-vs-ref comparison in §8 is therefore gated on GCC builds.** Clang builds run every other test (round-trip, LOD coherence, determinism, sentinel semantics) but skip fast-vs-ref. MSVC is not a build target in v1.

**`c3d_chunk_validate` implementation.** Shares its structural-check logic with the decoder. Internally the TU defines one `bool c3d_chunk_validate_structure(const uint8_t *, size_t)` that checks magic, version, header sizes, table offsets, per-subband frame sizes, and TLV bounds without running rANS decode. The decoder calls it and panics on `false`; `c3d_chunk_validate` returns the bool directly. One logic copy, two behaviours.

---

## 7. Development order

Internal build sequence. No public shipping milestones — the library is "done" when every stage below is implemented, tested, and meets the R-D and perf gates in §8.

1. Scaffolding: `c3d_panic`, `c3d_assert`, alignment macros, bit-io, TLV reader, Morton-12 helpers.
2. `c3d_hash128` (MurmurHash3_x64_128) + round-trip test.
3. Scalar rANS encode/decode + random-distribution round-trip test.
4. 8-way interleaved rANS + throughput benchmark vs scalar.
5. 1D CDF 9/7 lifting + impulse-response and double-precision cross-check.
6. Separable 3D DWT round-trip (forward ∘ inverse ≈ identity within 1e-6).
7. Quantizer + Laplacian dequant + zigzag+escape symbol mapping round-trip.
8. Per-subband frequency-table build + compact serialisation round-trip.
9. Rate-control loop: Laplacian warm-start, bisection on quantizer scalar to hit `target_ratio` within 2 %.
10. Full chunk encode/decode for SELF mode; LOD partial decode. `c3d_chunk_encode_at_q` (bypass rate control) and `c3d_chunk_validate` (non-panicking structural check) ship alongside the core encode/decode.
11. Shard container: parse, serialise, Morton-indexed sentinels, embedded `.c3dx` slot. Includes `c3d_shard_chunk_count`.
12. `.c3dx` parse/serialise (incl. `c3d_ctx_serialized_size`), builder (single-pass stats + optional `SUBBAND_FREQ_TABLES`), EXTERNAL-mode chunks.
13. `c3d_downsample_chunk_2x` helper (box 2³).
14. `c3d_train`, `c3d_inspect`, `c3d_compact` CLI tools.
15. `c3d_bench` + openh264 baseline harness.

Each step lands with tests in `c3d_test.c`. The reference code path grows alongside the fast path as each stage is added; once full chunk encode/decode exists (step 10+), every subsequent test cross-checks fast-vs-ref on the corpus.

---

## 8. Testing, benchmarking, gates

**Project layout:**

```
c3d/
├── CMakeLists.txt
├── c3d.h
├── c3d.c
├── c3d_test.c
├── c3d_bench.c
├── c3d_train.c
├── c3d_inspect.c
├── c3d_compact.c
├── third_party/openh264/   (bench only)
├── corpus/                  (gitignored; user supplies)
├── PLAN.md
├── CLAUDE.md
├── README.md
└── LICENSE
```

Every `c3d*.{c,h}` file begins with a short header comment pointing at `LICENSE`. No per-file copyright boilerplate beyond that.

**Tests (`c3d_test.c`):**

- **DWT round-trip:** `max |x − idwt(dwt(x))| < 1e-5` on random-float inputs.
- **Quantizer round-trip:** for each tested step, `|dequant(quant(x)) − x| ≤ step/2 + α·step` per coefficient.
- **Full round-trip** at `target_ratio ∈ {2, 10, 100, 1000}`: decode matches encode-source within PSNR targets stated for each ratio on the scroll corpus (TBD once we measure v1; these become regression gates).
- **LOD coherence:**
  - For every corpus chunk, compute the ideal wavelet pyramid from the original by running the same CDF 9/7 5-level forward DWT on `f32(original)` and keeping the LLL coefficients at each level. Clamp back to u8 to get `ground_truth_lod_k` for `k ∈ 0..5`.
  - Compare to `c3d_chunk_decode_lod(encoded, in_len, k, ctx, out)`:
    `PSNR(decoded_lod_k, ground_truth_lod_k) ≥ PSNR(decoded_lod_0, original) − 1 dB` at the same `target_ratio`. LOD k should degrade no worse than LOD 0 because coarser subbands get a larger relative bit budget.
- **Sentinel semantics:** ABSENT decode → panic (caught via hook); ZERO decode → zero-filled output.
- **`.c3dx` round-trip:** parse(serialize(ctx)) produces identical records; `c3d_ctx_id` is stable across round-trips; hash-mismatch panics.
- **Shard parse/serialise idempotence:** `parse(serialize(parse(bytes)))` produces byte-identical bytes, including preservation of ABSENT and ALL_ZERO sentinels.
- **Same-build encode determinism:** encoding the same input twice with the same binary and same options produces byte-identical encoded chunks. Cross-binary determinism is not tested or guaranteed.
- **Fast vs reference decoder:** for every corpus chunk and every `lod ∈ {0..5}`, `max |fast[i] − ref[i]| ≤ 1` across all voxels, `mean |fast[i] − ref[i]| < 0.1`. PSNR(fast, ref) ≥ 60 dB. Hard bit-equality is not required and not guaranteed under `-ffast-math`.

**Corpus.** Directory of raw u8 files, each exactly `256*256*256 = 16 777 216` bytes, any filename. Path supplied via env var `C3D_CORPUS`. User will provide the scroll corpus. `c3d_inspect` / `c3d_compact` CLI tools emit human-readable text to stdout; their exact format is not stable across versions and not considered part of the public API.

**Benchmarks (`c3d_bench.c`):** encode/decode throughput, compressed size, PSNR / SSIM / 3D-SSIM vs original, all at target_ratio ∈ {2, 10, 100, 1000}. Baseline: `openh264` intra-only at matched bitrates, same corpus.

**Quality gate to "ship."** c3d PSNR ≥ openh264 PSNR at every tested target_ratio on every corpus chunk. **Met as of the perceptual-weights + lowered q_min revisions** — c3d wins at every QP from 18 to 48 by 5-8 dB on the 64-chunk middle-of-scroll corpus.

**Performance gate.** Measured on the ARM X Elite dev box, scalar autovectorised (no intrinsics):
   - DWT fwd+inv (256³ f32 alone): ~295 MB/s
   - End-to-end decode: 70-110 MB/s (DWT-bound)
   - End-to-end encode (rate-controlled): 20-35 MB/s (8-iter bisection)
   - End-to-end encode (`encode_at_q`, no rate control): 85-95 MB/s

Major optimization steps (each individually verified against the corpus):
   - 4-column-tiled Y/Z DWT passes → 2.9× DWT, ~2× decode (no intrinsics; compiler autovectorises the 4-way inner loops as 128-bit NEON FMAs)
   - Reusable `c3d_encoder` / `c3d_decoder` contexts → +25-30 % throughput by killing 50-100 ms/call malloc churn
   - 8-lane manual unroll of rANS decode → +5 % on top
   - Per-subband perceptual quantizer baseline (`baseline ∝ 1/w^0.25`) → +5-8 dB PSNR at every byte budget
   - Widened rate-control range to `q ∈ [2^-12, 2^12]` so bisection can match large byte targets under perceptual weighting

**Primary dev hardware.** Caller's main workstation (the CLI env reports Linux on ARM X Elite). Measurements reported on that box. Cross-platform validation (x86_64, both GCC and Clang) happens in CI once the implementation is stable.

**Fuzzing.** Out of scope for v1.

**Error recovery.** None. Every malformed input, every memory exhaustion, every assertion failure → `c3d_panic`. Happy path is the only path.

---

## 9. Risks & expected working set

1. **rANS gives up 0.5-1.5 dB vs 3D-EBCOT.** Accepted. Payback: ~10× decode speed, ~5× less code, trivially SIMD-friendly. If the gap to H.264 on the scroll corpus is larger than this, the lever to pull is *not* switching back to EBCOT — it's training a shard-level `SUBBAND_FREQ_TABLES` `.c3dx` record, which converts static per-chunk tables into a shard-global better-fitted prior at zero decode-time cost.
2. **Rate-control convergence.** Pathological chunks may not converge within the 8-iteration cap (§3.5). When that happens the encoder accepts whatever byte count the current `q` produces — deterministic, reproducible, and at worst off-target by ~10-20 %.
3. **Float32 DWT drift at level 5.** CDF 9/7 is numerically stable; ~0.02 dB drift vs double-precision reference across 5 levels on 256³ per literature. Tested against a double-precision reference in the DWT round-trip.
4. **`-ffast-math` reorders floating-point ops.** Encode output may differ by a few ULPs across compilers/platforms. Tests compare decoded voxels with tolerance (§8), not encoded streams byte-for-byte across binaries. Same-binary reproducibility is guaranteed. The reference decoder uses canonical FP evaluation order so that fast-vs-ref comparison on the same platform is meaningful.
5. **Single-TU size.** ~3500-4000 lines projected — well under single-file pain thresholds. Unchanged as a principle.
6. **High-ratio / uniform-chunk edge cases.** `target_ratio → ∞` and chunks uniform-after-centering both collapse to "reconstruct from `dc_offset` alone" (§3.1, §3.5).

**Expected per-encoder working set (one chunk in flight, single thread):**

| Buffer                                               | Size       |
|------------------------------------------------------|------------|
| f32 coefficient buffer (256³ × 4)                    | 64 MiB     |
| DWT line scratch                                     | ~1 KiB     |
| i16 quantized-coefficient buffer (256³ × 2)          | 32 MiB     |
| u8 symbol-index buffer (256³)                        | 16 MiB     |
| Per-subband rANS output (accumulated during bisection) | 0.5-4 MiB |
| Per-subband frequency tables (36 × ~100 B)           | ~4 KiB     |
| Escape streams (very small at typical ratios)        | ≪ 1 MiB    |
| **Peak encoder**                                     | **~115-125 MiB** |

Decoder working set: 64 MiB coefficient buffer + 16 MiB symbol buffer + referenced input bytes + trivial scratch ≈ **~85 MiB**. No PCRD slope tables; encoder is a bit heavier than decoder because of the rate-control bisection state.

Callers parallelising N encodes should budget ~125 MiB per thread; N decodes ≈ ~85 MiB per thread. A fully-populated 4096-chunk shard at 10:1 held in memory (copy mode) is ≈ **6.8 GiB** of shard bytes; prefer `c3d_shard_parse` (non-copy) over `c3d_shard_parse_copy` for large shards.

**Minimum chunk sizes.** Empty-chunk path (all `lod_offset[k] = 0`): 352 B header + tables only. Non-empty floor (all 36 subbands present, each with a single-symbol freq table and zero rANS renorm bytes): 352 + 36 × (2 + ~3 + 4 + 4 + 32) = ~1.9 KiB. Between 352 B and ~1.9 KiB there is a quantisation gap: target sizes in that window snap to one end or the other via the rate-control loop. Inherent; not a bug.

---

## 10. What's locked

Bytes on disk (format version 1, permanent during dev):

- u64 key: `[lod:4][z:20][y:20][x:20]`.
- Shard header: 64 B fixed; includes `ctx_offset`, `ctx_size` for embedded `.c3dx`.
- Shard layout: header, then 64 KiB **Morton-ordered** chunk index, then optional `.c3dx`, then chunk payloads.
- Sentinels: `ABSENT = (UINT64_MAX, 0)`, `ALL_ZERO = (0, 0)`.
- Chunk header: 40 B fixed + 144 B qmul + 144 B subband_offset + 24 B lod_offset + entropy payload. Total fixed = 352 B.
- Subband naming: `ZYX` letter order, `L`/`H` filter. Canonical bitstream order: LLL_5 then per-level `HHH HHL HLH LHH HLL LHL LLH` from level 5 down to 1.
- 5 DWT levels, 36 subbands, 6 LODs — all hardcoded constants, not header fields.
- Entropy: **per-subband static-model rANS** (`ryg_rans_byte`-style, 8-way interleaved). 65-symbol alphabet (zigzag 0..63 + escape, escape payload = LEB128 varint in a separate per-subband escape stream). Denominator `M = 4096` for detail subbands, `M = 16384` for `LLL_5`. Frequencies must sum to `M`; parser verifies.
- Per-subband bitstream layout: `u16 freq_table_size; freq_table; u32 n_symbols; u32 rans_block_size; u8[32] rans_header (8×u32 states); u8[] rans_renorm; u8[] escape_stream`.
- LEB128 unsigned varints for all varint fields (frequency values and escape magnitudes).
- All multi-byte on-disk fields are little-endian; build-time static-assert rejects big-endian targets.
- Rate-control search range: `chunk_scalar ∈ [2^-12, 2^12]`; bisection capped at 8 iterations. (Originally locked at 2^-6 lower bound; widened during implementation when perceptual weights compressed HF subbands so aggressively that the natural entropy floor was reached at q=2^-6, preventing rate control from hitting low-compression byte targets.)
- Per-subband baseline quantizer weights derived from CDF 9/7 synthesis-gain² products: `baseline[i] ∝ 1/w_i^0.25` where `w_i = G_L²^(L_count) · G_H²^(H_count)` over the subband's L/H pattern across all DWT levels. Normalised so the geomean across 36 subbands is 1.0. Step computed at encode is `q × baseline[i]` (or `q × ctx_baseline[i]` if a `.c3dx QUANTIZER_BASELINE` record is provided).
- Caller-visible encode upper bound: `C3D_CHUNK_ENCODE_MAX_SIZE = 16·1024·1024 + 352 + 4096` bytes. The macro is guaranteed stable; `c3d_chunk_encode_max_size()` returns the same value.
- `.c3dx`: 24 B header + TLV records (`u16 tag, u16 length_quads, u8[] value`), **65 535 B max** (u16 `block_size`). Shard header carries `ctx_size: u16` matching this.
- v1 `.c3dx` record tags: 1 = LAPLACIAN_ALPHA, 2 = QUANTIZER_BASELINE, 3 = SUBBAND_FREQ_TABLES. Tags ≥ 4 reserved.
- Per-subband arrays (on-disk) indexed by **canonical subband order** (§2.3): `LLL_5, HHH_5..LLH_5, HHH_4..LLH_4, …, HHH_1..LLH_1`.
- `c3d_hash128` = MurmurHash3_x64_128 (16 bytes).
- Little-endian only.

---

## 11. Not in v1

- Bit-exact u8 round-trip (float compute throughout).
- Strict cross-platform bit-identical encode output (close-enough determinism only).
- Multi-threaded encode/decode of a single chunk. Parallelism = one thread per chunk.
- Spatial intra prediction.
- SIMD intrinsics (scalar autovectorisable only).
- Chunk-level CRC. rANS decode fails loudly on corrupt bitstreams (invalid state or symbol lookup) and hits a panic; anything subtler than that is out of scope for v1.
- GPU decode.
- File I/O, network, mmap inside the library.
- Fuzzing harness.
- Forward / backward compatibility.
- Compile-time corpus-baked priors (corpus-learned parameters only travel as `.c3dx`).

### Deferred — gated scaffolding on main

- **Per-chunk R-D allocator (Q3).** Two implementations gated off on main:
  - `c3d_rd_allocate` (fine-histogram, v1, `C3D_RD_ALLOCATOR=1`): rate
    estimate undercounts ~20–25 % vs the accurate quant-scan estimator,
    so byte targets drift.
  - `c3d_rd_allocate_hybrid` (v2, `C3D_RD_HYBRID=1`): rate accuracy is
    fine (uses the same accurate estimator as global-q bisection, targets
    the budget that estimator predicts at the converged q).  The
    distortion metric regresses PSNR by 0.1–1.0 dB regardless of grid
    width because weighted-coef-MSE + escape-bin mean-|q| approximation
    doesn't match true pixel MSE under bi-orthogonal CDF 9/7.  The
    perceptual softness=0.5 baseline already balances R-D slopes in pixel
    space, so moves away from mult=1 need a more faithful distortion
    model (closed-form Laplacian integral, or exact per-escape dequant
    accounting).  Leave scaffolding, revisit when willing to derive the
    full closed-form dist formula.

- **Context-adaptive rANS (Q4).** Would use per-symbol freq tables keyed on
  the causal neighbour's class for another +0.3–1.0 dB.  Not attempted: the
  8-lane interleaved decode fundamentally relies on a single shared freq
  table so all lanes advance independently.  Per-symbol table selection
  introduces a read-after-write dependency (the current symbol's table
  needs the previous symbol, which the adjacent lane just wrote) and
  serialises the 8 lanes, costing ~3× decode rANS time.  Net effect would
  likely regress end-to-end decode throughput before the dB win paid off.
  Revisit only if willing to move to a single-state rANS path for the
  context-adaptive subbands, or to rework the interleave so the causal
  dependency is lane-local.
