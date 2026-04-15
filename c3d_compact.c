/* c3d_compact — read a shard, re-serialise it (drops orphan payload bytes).
 * See LICENSE.  Usage: c3d_compact <in.c3ds> <out.c3ds>
 *
 * "Compact" is a thin wrapper around parse→serialise.  It is useful when the
 * input shard was produced by an external appender that never rewrote the
 * index after a chunk update, leaving orphan payloads.  c3d's own serialise
 * path never produces orphans, so the output is always compact.
 */

#include "c3d.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint8_t *slurp_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    rewind(fp);
    uint8_t *buf = aligned_alloc(32, ((size_t)sz + 31u) & ~(size_t)31u);
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { perror("fread"); exit(1); }
    fclose(fp);
    *out_len = (size_t)sz;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in.c3ds> <out.c3ds>\n", argv[0]);
        return 2;
    }
    size_t n; uint8_t *in = slurp_file(argv[1], &n);
    c3d_shard *s = c3d_shard_parse_copy(in, n);
    free(in);

    size_t need = c3d_shard_max_serialized_size(s);
    uint8_t *out = malloc(need);
    if (!out) { fprintf(stderr, "oom\n"); return 1; }
    size_t wrote = c3d_shard_serialize(s, out, need);

    FILE *fp = fopen(argv[2], "wb");
    if (!fp) { perror(argv[2]); return 1; }
    if (fwrite(out, 1, wrote, fp) != wrote) { perror("fwrite"); return 1; }
    fclose(fp);

    fprintf(stderr, "in: %zu bytes, out: %zu bytes (saved %lld)\n",
            n, wrote, (long long)(n - wrote));
    c3d_shard_free(s);
    free(out);
    return 0;
}
