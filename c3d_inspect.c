/* c3d_inspect — dump c3d chunk / shard / .c3dx metadata to stdout.
 * See LICENSE.  Output format is not stable across versions.
 *
 * Usage: c3d_inspect <file>
 *    Auto-detects based on the first 4 bytes of the file's magic:
 *      "C3DC" → chunk
 *      "C3DS" → shard
 *      "C3DX" → context block
 */

#include "c3d.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *slurp_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    if (fseek(fp, 0, SEEK_END) != 0) { perror("fseek"); exit(1); }
    long sz = ftell(fp);
    if (sz < 0) { perror("ftell"); exit(1); }
    rewind(fp);
    /* 32-byte aligned for voxel-friendliness; most call sites don't need it
     * for encoded buffers but it doesn't hurt. */
    uint8_t *buf = aligned_alloc(32, ((size_t)sz + 31u) & ~(size_t)31u);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { perror("fread"); exit(1); }
    fclose(fp);
    *out_len = (size_t)sz;
    return buf;
}

static void inspect_chunk(const uint8_t *in, size_t in_len) {
    printf("type: chunk\n");
    printf("size: %zu bytes\n", in_len);
    if (!c3d_chunk_validate(in, in_len)) {
        printf("valid: FALSE (structural check)\n");
        return;
    }
    printf("valid: true\n");
    c3d_chunk_info info;
    c3d_chunk_inspect(in, in_len, &info);
    printf("context_mode: %s\n", info.context_mode == 0 ? "SELF" : "EXTERNAL");
    if (info.context_mode == 1) {
        printf("context_id:  ");
        for (int i = 0; i < 16; ++i) printf("%02x", info.context_id[i]);
        printf("\n");
    }
    printf("dc_offset:    %.6f\n", (double)info.dc_offset);
    printf("coeff_scale:  %.6f\n", (double)info.coeff_scale);
    printf("lod_offsets:  ");
    for (unsigned k = 0; k < 6; ++k) printf("%u%s", info.lod_offsets[k], k==5 ? "\n" : " ");
    if (info.lod_offsets[0] == 0) {
        printf("               (empty chunk — reconstructs from dc_offset only)\n");
    }
}

static void inspect_shard(const uint8_t *in, size_t in_len) {
    printf("type: shard\n");
    printf("size: %zu bytes\n", in_len);
    c3d_shard *s = c3d_shard_parse(in, in_len);
    uint32_t n_abs = c3d_shard_chunk_count(s, C3D_CHUNK_ABSENT);
    uint32_t n_zer = c3d_shard_chunk_count(s, C3D_CHUNK_ZERO);
    uint32_t n_pre = c3d_shard_chunk_count(s, C3D_CHUNK_PRESENT);
    printf("chunks: ABSENT=%u  ZERO=%u  PRESENT=%u  (total=%u)\n",
           n_abs, n_zer, n_pre, n_abs + n_zer + n_pre);
    const c3d_ctx *ctx = c3d_shard_ctx(s);
    printf("embedded_ctx: %s\n", ctx ? "yes" : "no");
    if (ctx) {
        uint8_t id[16]; c3d_ctx_id(ctx, id);
        printf("ctx_size: %zu bytes\n", c3d_ctx_serialized_size(ctx));
        printf("ctx_id:   ");
        for (int i = 0; i < 16; ++i) printf("%02x", id[i]);
        printf("\n");
    }
    c3d_shard_free(s);
}

static void inspect_ctx(const uint8_t *in, size_t in_len) {
    printf("type: context block (.c3dx)\n");
    printf("size: %zu bytes\n", in_len);
    c3d_ctx *ctx = c3d_ctx_parse(in, in_len);
    uint8_t id[16]; c3d_ctx_id(ctx, id);
    printf("ctx_id: ");
    for (int i = 0; i < 16; ++i) printf("%02x", id[i]);
    printf("\n");
    printf("serialized_size: %zu\n", c3d_ctx_serialized_size(ctx));
    c3d_ctx_free(ctx);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <file>\n", argv[0]);
        return 2;
    }
    size_t n; uint8_t *buf = slurp_file(argv[1], &n);
    if (n < 4) { fprintf(stderr, "file too small (%zu bytes)\n", n); return 1; }

    if      (memcmp(buf, "C3DC", 4) == 0) inspect_chunk(buf, n);
    else if (memcmp(buf, "C3DS", 4) == 0) inspect_shard(buf, n);
    else if (memcmp(buf, "C3DX", 4) == 0) inspect_ctx  (buf, n);
    else {
        fprintf(stderr, "unknown magic: %02x%02x%02x%02x\n",
                buf[0], buf[1], buf[2], buf[3]);
        return 1;
    }
    free(buf);
    return 0;
}
