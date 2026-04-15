/* c3d_train — build a .c3dx context block from a directory of raw u8 chunks.
 * See LICENSE.
 *
 * Usage:  c3d_train <corpus_dir> <out.c3dx>
 *   Iterates every regular file in corpus_dir; each file must be exactly
 *   256³ = 16 777 216 bytes (raw u8 voxels, z-major).  Observations accumulate
 *   into a c3d_ctx_builder; on completion, emits a SUBBAND_FREQ_TABLES ctx
 *   suitable for attaching to shards encoded from similar data.
 */

#include "c3d.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define CHUNK_BYTES (256u * 256u * 256u)

static int read_chunk(const char *path, uint8_t *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return -1; }
    size_t n = fread(out, 1, CHUNK_BYTES, fp);
    fclose(fp);
    if (n != CHUNK_BYTES) {
        fprintf(stderr, "%s: expected %u bytes, got %zu\n", path, CHUNK_BYTES, n);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <corpus_dir> <out.c3dx>\n", argv[0]);
        return 2;
    }
    const char *dir_path = argv[1];
    const char *out_path = argv[2];

    DIR *dir = opendir(dir_path);
    if (!dir) { perror(dir_path); return 1; }

    c3d_ctx_builder *b = c3d_ctx_builder_new();
    uint8_t *chunk = aligned_alloc(32, CHUNK_BYTES);
    if (!chunk) { fprintf(stderr, "oom\n"); return 1; }

    size_t n_observed = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof full, "%s/%s", dir_path, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (!S_ISREG(st.st_mode)) continue;
        if ((size_t)st.st_size != CHUNK_BYTES) {
            fprintf(stderr, "skip %s: wrong size %lld\n",
                    full, (long long)st.st_size);
            continue;
        }
        if (read_chunk(full, chunk) != 0) continue;
        c3d_ctx_builder_observe_chunk(b, chunk);
        ++n_observed;
        fprintf(stderr, "[%zu] %s\n", n_observed, ent->d_name);
    }
    closedir(dir);

    if (n_observed == 0) {
        fprintf(stderr, "no chunks observed — nothing to train\n");
        return 1;
    }

    c3d_ctx *ctx = c3d_ctx_builder_finish(b, true);
    size_t sz = c3d_ctx_serialized_size(ctx);
    uint8_t *out = malloc(sz);
    c3d_ctx_serialize(ctx, out, sz);

    FILE *fp = fopen(out_path, "wb");
    if (!fp) { perror(out_path); return 1; }
    if (fwrite(out, 1, sz, fp) != sz) { perror("fwrite"); return 1; }
    fclose(fp);

    uint8_t id[16]; c3d_ctx_id(ctx, id);
    fprintf(stderr, "wrote %zu bytes; ctx_id=", sz);
    for (int i = 0; i < 16; ++i) fprintf(stderr, "%02x", id[i]);
    fprintf(stderr, "\n");

    free(out); free(chunk);
    c3d_ctx_free(ctx);
    return 0;
}
