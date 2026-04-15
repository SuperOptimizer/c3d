#!/usr/bin/env python3
"""Fetch zarr chunks from multiple regions of the scroll and stitch 2×2×2
groups into 256³ c3d chunks.  Writes raw u8 files to ./corpus/.

Pulls from three (y,x) offsets across the scroll's middle z range to get
varied structure: dense interior, offset-from-center, and closer to edge.

Usage:  AWS_PROFILE=AdministratorAccess-585768151128 python3 fetch_corpus.py
"""
import os, subprocess, sys, pathlib, concurrent.futures
import numpy as np

ZARR  = "s3://scrollprize-volumes/esrf/20260311/2.4um_PHerc-Paris4_masked.zarr/0"
SHAPE = (75784, 32693, 32693)
CHUNK = 128

# Three tiles, each a 4×4×4 zarr region → 2×2×2 = 8 stitched c3d chunks.
# Covers three different spatial locations for variety.
center_cz = (SHAPE[0] // CHUNK) // 2   # ~296
center_cy = (SHAPE[1] // CHUNK) // 2   # ~127
center_cx = (SHAPE[2] // CHUNK) // 2   # ~127

TILES = [
    ("mid",    (center_cz - 2,  center_cy - 2,  center_cx - 2)),   # deep centre
    ("shallow",(center_cz - 60, center_cy - 2,  center_cx - 2)),   # ~60 chunks earlier in z
    ("deep",   (center_cz + 60, center_cy - 2,  center_cx - 2)),   # ~60 chunks later in z
    ("off1",   (center_cz - 2,  center_cy - 20, center_cx - 2)),   # y-offset
    ("off2",   (center_cz - 2,  center_cy + 20, center_cx - 2)),   # y-offset other way
    ("off3",   (center_cz - 2,  center_cy - 2,  center_cx - 20)),  # x-offset
    ("off4",   (center_cz - 2,  center_cy - 2,  center_cx + 20)),  # x-offset other way
    ("corner", (center_cz - 2,  center_cy - 40, center_cx - 40)),  # further off-centre
]
NZ = NY = NX = 4   # 4x4x4 zarr chunks per tile → 2x2x2 = 8 c3d chunks per tile

out_dir = pathlib.Path("corpus")
out_dir.mkdir(exist_ok=True)
tmp = pathlib.Path("/tmp/zarr_cache")
tmp.mkdir(parents=True, exist_ok=True)

def fetch_one(cz, cy, cx):
    local = tmp / f"{cz}_{cy}_{cx}.u8"
    if local.exists() and local.stat().st_size == CHUNK**3:
        return local
    s3 = f"{ZARR}/{cz}/{cy}/{cx}"
    r = subprocess.run(["aws", "s3", "cp", s3, str(local)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # fill-value chunks (all zeros) are absent in zarr v2 when no compressor
        np.zeros(CHUNK**3, dtype=np.uint8).tofile(local)
    return local

# Gather full fetch list, dedupe.
fetch_list = set()
for name, (cz0, cy0, cx0) in TILES:
    for dz in range(NZ):
        for dy in range(NY):
            for dx in range(NX):
                fetch_list.add((cz0 + dz, cy0 + dy, cx0 + dx))
print(f"Fetching {len(fetch_list)} zarr chunks across {len(TILES)} tiles...")

# Parallel fetch (16 concurrent).
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    futures = [ex.submit(fetch_one, *c) for c in fetch_list]
    for i, f in enumerate(concurrent.futures.as_completed(futures)):
        f.result()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(fetch_list)}")

# Stitch each tile → 8 × 256^3 c3d chunks.
n_written = 0
for name, (cz0, cy0, cx0) in TILES:
    for sz in range(NZ // 2):
        for sy in range(NY // 2):
            for sx in range(NX // 2):
                out = np.zeros((256, 256, 256), dtype=np.uint8)
                for dz in range(2):
                    for dy in range(2):
                        for dx in range(2):
                            cz = cz0 + 2*sz + dz
                            cy = cy0 + 2*sy + dy
                            cx = cx0 + 2*sx + dx
                            local = tmp / f"{cz}_{cy}_{cx}.u8"
                            arr = np.fromfile(local, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
                            out[dz*CHUNK:(dz+1)*CHUNK,
                                dy*CHUNK:(dy+1)*CHUNK,
                                dx*CHUNK:(dx+1)*CHUNK] = arr
                outname = out_dir / f"{name}_z{cz0+2*sz:04d}_y{cy0+2*sy:04d}_x{cx0+2*sx:04d}.u8"
                out.tofile(outname)
                nz = int((out != 0).sum())
                n_written += 1
                print(f"{n_written:3d}. {outname.name}  nonzero={nz/out.size:5.1%}")

print(f"\n{n_written} chunks in {out_dir.absolute()}")
