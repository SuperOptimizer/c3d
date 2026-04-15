# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

c3d ("compress3d") is a custom compression codec for very-large-than-RAM/disk 3D volumetric u8 grayscale X-ray data (e.g. 75k × 35k × 35k voxel Vesuvius scroll scans). Target: better compression-ratio + quality + CPU-side encode/decode throughput than any general-purpose compressor (Blosc+Zstd, LZ4) or scientific codec (ZFP, JP3D, TTHRESH) on scroll CT data, in a minimal single-file C23 library.

See `PLAN.md` for the full design spec; CLAUDE.md only summarises what shapes day-to-day editing.

## Hard constraints (non-negotiable, see `PLAN.md` §0)

- **C23**, single `c3d.h` + `c3d.c`, libc only. `openh264` permitted in the benchmark harness only; the rANS reference (`ryg_rans`) is inlined into `c3d.c`.
- **In-memory API.** Library never touches disk, network, or fds. Everything is (bytes in) → (bytes out). Callers own all I/O.
- **Fatal on error.** No status codes, no recoverable errors. Invalid input / corruption / OOM → `c3d_panic()` → `abort()`. Happy path is the only path. `c3d_panic` is overridable via `c3d_set_panic_hook`.
- **Little-endian only.** Build-time static-assert rejects BE targets.
- **Format version = 1 forever, no compat.** Re-encoding from raw u8 is the upgrade path during development.
- **Generic vectorisable C, no intrinsics in v1.** Built with `-O3 -ffast-math -funsafe-math-optimizations`. Same-binary encode is byte-deterministic; cross-binary is not guaranteed.

## Pipeline

u8 256³ chunk → f32−128 ingest → 5-level 3D CDF 9/7 DWT (symmetric, float32, lifting) → per-subband uniform dead-zone quantizer → per-subband static-model rANS (8-way interleaved, `ryg_rans_byte`-style, 65-symbol alphabet: zigzag 0..63 + escape) → bytes. Rate control is a bisection loop on a single chunk-level quantizer scalar; no PCRD-opt, no tier-1/tier-2 distinction, no code-blocks.

## Fixed hierarchy (all power-of-2, cubic)

| level     | side      | role                                                    |
|-----------|-----------|---------------------------------------------------------|
| block     | 16        | caller-side RAM cache granularity                       |
| chunk     | 256       | **codec atom** — one encode/decode call                 |
| shard     | 4 096     | 16³ = 4 096 chunks + 64 KiB Morton-ordered index        |
| subvolume | 65 536    | shard grid                                              |
| volume    | 1 048 576 | max (20 bits / axis)                                    |

6 native LODs per chunk (256³, 128³, 64³, 32³, 16³, 8³) decoded from prefixes of one bitstream — no duplicated coefficients across LODs. Further zoom-out = caller-layered pyramid of separate shards (`c3d_downsample_chunk_2x` helper does 2³-box downsamples).

## u64 voxel key

`[lod:4][z:20][y:20][x:20]` — planar (not Morton). The shard *chunk index* is Morton-ordered for prefetch locality; individual voxel keys are not.

## Sparse / sentinel handling (first-class)

Shard chunk-index entry `(offset:u64, size:u64)` uses two sentinels:
- `ABSENT = (UINT64_MAX, 0)` — never written
- `ALL_ZERO = (0, 0)` — definitionally empty, no payload bytes
- `PRESENT` → any real `(offset ≥ 64 + 65536, size > 0)`

Empty chunks cost zero storage and zero decode. Critical for scroll data (75-85 % of chunks are uniformly zero after masking).

## External context block (`.c3dx`)

Every chunk is `SELF` or `EXTERNAL` per its `context_mode` byte:
- `SELF`: standalone, uses library default α = 0.375 and in-band frequency tables.
- `EXTERNAL`: requires a `.c3dx` context block whose `c3d_hash128` matches the chunk's `context_id`. Context blocks live either inside a shard (shard header points at them) or as a sidecar byte buffer (for streaming).

`.c3dx` is a TLV bag (tags: `LAPLACIAN_ALPHA`, `QUANTIZER_BASELINE`, `SUBBAND_FREQ_TABLES`). Max 65 535 B. `c3d_hash128` = XXH3-128 truncated to 16 bytes.

## Public API shape

- **Reusable contexts** (recommended for >1 chunk): `c3d_encoder_new/free`, `c3d_decoder_new/free`; `c3d_encoder_chunk_encode(enc, in, target_ratio, ctx, out, cap)` and `c3d_decoder_chunk_decode_lod(dec, in, len, lod, ctx, out)`. Each owns ~115 MiB / 80 MiB of scratch and saves 50-100 ms/chunk of malloc churn.
- Stateless chunk codec (allocates scratch each call): `c3d_chunk_encode`, `c3d_chunk_decode`, `c3d_chunk_decode_lod`.
- Debug / bench: `c3d_chunk_encode_at_q(in, q, ctx, out, cap)` bypasses rate control; `c3d_chunk_validate(in, len) → bool` non-panicking structural check.
- Inspect: `c3d_chunk_inspect(in, len, &info)`.
- Context: `c3d_ctx_parse`, `c3d_ctx_serialize`, `c3d_ctx_builder_*`.
- Shard: `c3d_shard_new`, `c3d_shard_parse` (non-copy), `c3d_shard_parse_copy`, `c3d_shard_serialize`, per-chunk ops (`_chunk_bytes`, `_put_chunk`, `_mark_zero`, `_encode_chunk`, `_decode_chunk`, `_decode_chunk_lod`, `_chunk_count`).

Full signatures in `PLAN.md` §5.

## Alignment

- Raw voxel buffers (256³ u8 inputs, LOD outputs): **32-byte aligned**, panic otherwise.
- Encoded byte buffers (chunk payloads, shard bytes, `.c3dx` blobs): no alignment requirement; multi-byte on-disk reads go through `memcpy` into typed locals.

## Code layout (`c3d.c`, single TU, ~3650 lines target)

```
§A  types, c3d_panic, c3d_assert, bit-io, alignment, Morton-12 helpers
§B  c3d_hash128 (XXH3-128, inlined public-domain)
§C  rANS engine (encode, decode, 8-way interleaved, ryg_rans-style)
§D  frequency-table build + serialise/parse
§E  CDF 9/7 lifting, separable 3D, symmetric extension
§F  quantizer + Laplacian dequant + zigzag/escape symbol mapping
§G  rate-control loop (Laplacian warm-start + bisection, q ∈ [2^-12, 2^12])
§H  chunk encoder pipeline
§I  chunk decoder pipeline + LOD partial decoder
§J  shard parse/serialise (in-memory, Morton-indexed)
§K  .c3dx parse/serialise + builder
§L  public API wrappers
```

Reference decoder lives alongside the fast path under `#ifdef C3D_BUILD_REF` (defined by `c3d_test.c`, not by shipped library builds). Reference functions are annotated `__attribute__((optimize("no-fast-math","no-associative-math")))`.

## Project layout

```
c3d/
├── CMakeLists.txt
├── c3d.h
├── c3d.c
├── c3d_test.c       (defines C3D_BUILD_REF)
├── c3d_bench.c      (links openh264 baseline)
├── c3d_train.c      (offline .c3dx builder CLI)
├── c3d_inspect.c    (CLI: dump chunk/shard/.c3dx metadata)
├── c3d_compact.c    (CLI: parse+serialize to drop any orphaned bytes)
├── third_party/openh264/   (bench only; submodule or fetched)
├── corpus/                  (gitignored; user supplies raw 256³ u8 files, env var C3D_CORPUS points at it)
├── PLAN.md         (the full design spec — canonical)
├── CLAUDE.md       (this file)
├── README.md
└── LICENSE
```

Every `c3d*.{c,h}` file begins with a short header comment pointing at `LICENSE`.

## Editing conventions

- **PLAN.md is canonical** for format bytes and algorithm spec. If code disagrees with PLAN.md, the bug is in the code.
- **Don't add features not in PLAN.md §5 or §11** without the user's explicit ask. PLAN.md §11 enumerates what's deliberately out of v1.
- **Don't bake corpus-learned priors into `c3d.c`** as compile-time constants; they belong in `.c3dx` data.
- **No intrinsics, no target-specific SIMD in v1.** Plain loops over aligned arrays; let the compiler autovectorise.
- **`c3d_assert(cond)`** = `do { if (!(cond)) c3d_panic(__FILE__, __LINE__, #cond); } while (0)`. Use it liberally on invariants. Never use plain `assert`.
- **No `errno`, no status-code returns.** Library functions either succeed or panic.
- **Multi-byte on-disk reads go through `memcpy`** into typed locals — the per-subband bitstream mixes u16 and u32 fields at arbitrary offsets.

## Reference: prior v1 (pre-rework) results

A previous DWT + Zstd prototype reached parity with H.264 at ~25:1 but lagged by 7-12 dB at higher ratios. See `memory/project_rd_status.md` for the table. The current design (rANS replacing Zstd, no prediction, no EBCOT) targets closing most of that gap via proper wavelet-tailored entropy coding at ~10× decode speed; ~0.5-1.5 dB vs EBCOT is the accepted cost for the simplicity and speed win.
