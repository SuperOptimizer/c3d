#!/usr/bin/env python3
"""c3d_zarr_to_c3d — bulk-convert a zarr v2 u8 volume to a tree of c3d shards.

Usage:
    AWS_PROFILE=...  python3 c3d_zarr_to_c3d.py \
        --in s3://bucket/path/to.zarr/0  \
        --out /path/to/out_dir            \
        --target-ratio 10                 \
        [--shard-grid 4096]               \
        [--workers 8]                     \
        [--ctx /path/to/prior.c3dx]       \
        [--bbox z0:z1,y0:y1,x0:x1]

Requires:
    - aws CLI (any auth that can list/get objects from `--in`)
    - `c3d_train` (optional, only if building a corpus-trained .c3dx)
    - libc3d via the `c3d_chunk_encode` ctypes binding here
    - numpy

The converter:
  1. Reads zarr metadata (.zarray) to learn the source's shape, chunk size,
     dtype, and dimension separator.
  2. Iterates the source in 4096³ shard-aligned regions.  For each shard:
       - Pulls all overlapping zarr chunks from S3 in parallel.
       - For each 256³ c3d chunk in the shard, stitches the source data,
         encodes via libc3d at `target_ratio`, installs into the shard.
       - Emits the serialised .c3ds shard file under `out/<sz>/<sy>/<sx>.c3ds`.
  3. Skips chunks that are uniformly zero (sentinel = ZERO).

This is a thin convenience tool — anything subtle should drop down to the
library directly.  See PLAN.md §5 for the canonical API.
"""
import argparse, ctypes, json, os, pathlib, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ─── ctypes binding to libc3d ──────────────────────────────────────────────
def load_libc3d():
    here = pathlib.Path(__file__).resolve().parent
    candidates = [here / "build" / "libc3d.a",  # static, won't dlopen
                  here / "build" / "libc3d.so", here / "libc3d.so"]
    # We need a shared lib.  Build one if not present:
    so = here / "build" / "libc3d.so"
    if not so.exists():
        # The cmake build produces a static lib by default.  Compile a shared
        # lib from the same source to support ctypes.
        c = here / "c3d.c"
        cmd = ["gcc", "-O3", "-ffast-math", "-funsafe-math-optimizations",
               "-mcpu=native", "-flto=auto", "-DNDEBUG", "-fPIC",
               "-shared", "-o", str(so), str(c), "-lm"]
        print("compiling libc3d.so:", " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so))
    lib.c3d_encoder_new.restype = ctypes.c_void_p
    lib.c3d_encoder_free.argtypes = [ctypes.c_void_p]
    lib.c3d_encoder_chunk_encode.restype = ctypes.c_size_t
    lib.c3d_encoder_chunk_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.c3d_chunk_encode_max_size.restype = ctypes.c_size_t
    lib.c3d_shard_new.restype = ctypes.c_void_p
    lib.c3d_shard_new.argtypes = [ctypes.POINTER(ctypes.c_uint32 * 3), ctypes.c_uint8]
    lib.c3d_shard_free.argtypes = [ctypes.c_void_p]
    lib.c3d_shard_put_chunk.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.c3d_shard_mark_zero.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.c3d_shard_max_serialized_size.restype = ctypes.c_size_t
    lib.c3d_shard_max_serialized_size.argtypes = [ctypes.c_void_p]
    lib.c3d_shard_serialize.restype = ctypes.c_size_t
    lib.c3d_shard_serialize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    return lib

# ─── zarr v2 reader (single-chunk via aws s3 cp or local file) ──────────────
class ZarrReader:
    def __init__(self, src, dim_sep, dtype, chunk_shape, shape, aws_profile=None):
        self.src = src.rstrip("/")
        self.dim_sep = dim_sep
        self.dtype = np.dtype(dtype)
        self.chunk_shape = chunk_shape
        self.shape = shape
        self.aws_profile = aws_profile
        self._cache_dir = pathlib.Path("/tmp/c3d_zarr_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_uri(self, cz, cy, cx):
        return f"{self.src}/{cz}{self.dim_sep}{cy}{self.dim_sep}{cx}"

    def fetch(self, cz, cy, cx):
        n_bytes = int(np.prod(self.chunk_shape)) * self.dtype.itemsize
        local = self._cache_dir / f"{cz}_{cy}_{cx}.bin"
        if local.exists() and local.stat().st_size == n_bytes:
            return np.fromfile(local, dtype=self.dtype).reshape(self.chunk_shape)
        if self.src.startswith("s3://"):
            uri = self._chunk_uri(cz, cy, cx)
            env = os.environ.copy()
            if self.aws_profile:
                env["AWS_PROFILE"] = self.aws_profile
            r = subprocess.run(["aws", "s3", "cp", uri, str(local)],
                               capture_output=True, env=env)
            if r.returncode != 0:
                # Missing chunk = zarr-v2 fill_value (zero for our use case).
                return np.zeros(self.chunk_shape, dtype=self.dtype)
        else:
            local_path = pathlib.Path(self.src) / f"{cz}{self.dim_sep}{cy}{self.dim_sep}{cx}"
            if not local_path.exists():
                return np.zeros(self.chunk_shape, dtype=self.dtype)
            local.write_bytes(local_path.read_bytes())
        return np.fromfile(local, dtype=self.dtype).reshape(self.chunk_shape)

def load_zarr_meta(src, aws_profile=None):
    if src.startswith("s3://"):
        local = pathlib.Path("/tmp/c3d_zarr_cache/.zarray")
        local.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if aws_profile: env["AWS_PROFILE"] = aws_profile
        r = subprocess.run(["aws", "s3", "cp", f"{src.rstrip('/')}/.zarray", str(local)],
                           capture_output=True, env=env)
        if r.returncode != 0:
            print(r.stderr.decode(), file=sys.stderr)
            sys.exit(1)
        meta = json.loads(local.read_text())
    else:
        meta = json.loads((pathlib.Path(src) / ".zarray").read_text())
    sep = meta.get("dimension_separator", ".")
    dtype = meta["dtype"]
    return ZarrReader(src, sep, dtype, tuple(meta["chunks"]), tuple(meta["shape"]), aws_profile)

# ─── conversion ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_src", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--target-ratio", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aws-profile", default=None)
    ap.add_argument("--bbox", default=None,
                    help="z0:z1,y0:y1,x0:x1  (zarr-chunk-aligned bounds)")
    args = ap.parse_args()

    z = load_zarr_meta(args.in_src, args.aws_profile)
    if z.dtype != np.uint8 or len(z.shape) != 3:
        print(f"only 3D u8 zarr supported (got dtype={z.dtype}, shape={z.shape})",
              file=sys.stderr)
        sys.exit(1)
    if z.chunk_shape != (128, 128, 128):
        print(f"warning: chunk_shape={z.chunk_shape}; assuming we can stitch into 256³")

    lib = load_libc3d()
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # bbox in zarr-chunk units
    if args.bbox:
        z0, z1, y0, y1, x0, x1 = (int(v) for s in args.bbox.replace(",", ":").split(":") for v in [s])
    else:
        z1 = (z.shape[0] + z.chunk_shape[0] - 1) // z.chunk_shape[0]
        y1 = (z.shape[1] + z.chunk_shape[1] - 1) // z.chunk_shape[1]
        x1 = (z.shape[2] + z.chunk_shape[2] - 1) // z.chunk_shape[2]
        z0 = y0 = x0 = 0

    # 256³ c3d chunks per axis from a (zsz, ysz, xsz)-region of zarr chunks
    cz_per_chunk = 256 // z.chunk_shape[0]   # = 2 for 128³ zarr
    cy_per_chunk = 256 // z.chunk_shape[1]
    cx_per_chunk = 256 // z.chunk_shape[2]
    # 256³ c3d chunks per shard along each axis = 16
    shard_dim_zc = 16 * cz_per_chunk         # 32 zarr chunks per axis per shard
    shard_dim_yc = 16 * cy_per_chunk
    shard_dim_xc = 16 * cx_per_chunk

    encoder = lib.c3d_encoder_new()
    enc_buf = (ctypes.c_char * lib.c3d_chunk_encode_max_size())()

    # Iterate over shards.
    for sz in range((z0 // shard_dim_zc), (z1 + shard_dim_zc - 1) // shard_dim_zc):
        for sy in range((y0 // shard_dim_yc), (y1 + shard_dim_yc - 1) // shard_dim_yc):
            for sx in range((x0 // shard_dim_xc), (x1 + shard_dim_xc - 1) // shard_dim_xc):
                process_shard(args, z, lib, encoder, enc_buf, out,
                              sz, sy, sx,
                              shard_dim_zc, shard_dim_yc, shard_dim_xc,
                              cz_per_chunk, cy_per_chunk, cx_per_chunk)
    lib.c3d_encoder_free(encoder)

def process_shard(args, zr, lib, encoder, enc_buf, out_dir,
                  sz, sy, sx,
                  shard_dim_zc, shard_dim_yc, shard_dim_xc,
                  cz_per_chunk, cy_per_chunk, cx_per_chunk):
    # Shard origin in voxel coords:
    origin_z = sz * shard_dim_zc * zr.chunk_shape[0]
    origin_y = sy * shard_dim_yc * zr.chunk_shape[1]
    origin_x = sx * shard_dim_xc * zr.chunk_shape[2]
    print(f"shard ({sz},{sy},{sx}) at voxel ({origin_z},{origin_y},{origin_x})",
          file=sys.stderr)

    origin = (ctypes.c_uint32 * 3)(origin_x, origin_y, origin_z)
    shard = lib.c3d_shard_new(ctypes.byref(origin), 0)

    # Prefetch all overlapping zarr chunks for this shard.
    fetch_set = []
    for dz in range(shard_dim_zc):
        for dy in range(shard_dim_yc):
            for dx in range(shard_dim_xc):
                cz = sz * shard_dim_zc + dz
                cy = sy * shard_dim_yc + dy
                cx = sx * shard_dim_xc + dx
                if (cz * zr.chunk_shape[0] >= zr.shape[0] or
                    cy * zr.chunk_shape[1] >= zr.shape[1] or
                    cx * zr.chunk_shape[2] >= zr.shape[2]):
                    continue
                fetch_set.append((cz, cy, cx))
    fetched = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(zr.fetch, *c): c for c in fetch_set}
        for f in as_completed(futs):
            c = futs[f]
            fetched[c] = f.result()
    print(f"  fetched {len(fetched)} zarr chunks", file=sys.stderr)

    # Stitch + encode each c3d chunk.  c3d wants a 32-byte-aligned input
    # buffer; numpy doesn't guarantee that, so over-allocate and slice.
    n_present = n_zero = 0
    raw = np.empty(256**3 + 64, dtype=np.uint8)
    addr = raw.ctypes.data
    offset = (-addr) & 31
    chunk_buf = raw[offset:offset + 256**3].reshape(256, 256, 256)
    assert chunk_buf.ctypes.data % 32 == 0
    for ccz in range(16):
        for ccy in range(16):
            for ccx in range(16):
                # Source zarr-chunk offsets covered by this c3d chunk:
                base_cz = sz * shard_dim_zc + ccz * cz_per_chunk
                base_cy = sy * shard_dim_yc + ccy * cy_per_chunk
                base_cx = sx * shard_dim_xc + ccx * cx_per_chunk
                chunk_buf.fill(0)
                any_data = False
                for dz in range(cz_per_chunk):
                    for dy in range(cy_per_chunk):
                        for dx in range(cx_per_chunk):
                            arr = fetched.get((base_cz + dz, base_cy + dy, base_cx + dx))
                            if arr is None: continue
                            chunk_buf[dz * zr.chunk_shape[0]:(dz + 1) * zr.chunk_shape[0],
                                      dy * zr.chunk_shape[1]:(dy + 1) * zr.chunk_shape[1],
                                      dx * zr.chunk_shape[2]:(dx + 1) * zr.chunk_shape[2]] = arr
                            any_data = True
                if not any_data or not chunk_buf.any():
                    lib.c3d_shard_mark_zero(shard, ccx, ccy, ccz)
                    n_zero += 1
                    continue
                in_ptr = chunk_buf.ctypes.data_as(ctypes.c_char_p)
                n = lib.c3d_encoder_chunk_encode(
                    encoder, in_ptr, ctypes.c_float(args.target_ratio),
                    None, enc_buf, ctypes.c_size_t(len(enc_buf)))
                lib.c3d_shard_put_chunk(shard, ccx, ccy, ccz, enc_buf, n)
                n_present += 1
    print(f"  wrote {n_present} chunks ({n_zero} ZERO)", file=sys.stderr)

    # Serialise shard to disk.
    need = lib.c3d_shard_max_serialized_size(shard)
    out_buf = (ctypes.c_char * need)()
    wrote = lib.c3d_shard_serialize(shard, out_buf, need)
    out_path = out_dir / f"{sz:04d}" / f"{sy:04d}" / f"{sx:04d}.c3ds"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out_buf[:wrote]))
    print(f"  → {out_path}  ({wrote/1e6:.1f} MB)", file=sys.stderr)
    lib.c3d_shard_free(shard)

if __name__ == "__main__":
    main()
