/* c3d_labels_inspect — dump label chunk + schema metadata.  See LICENSE.
 *
 * Usage:  c3d_labels_inspect <schema.c3dls> [chunk.c3dl]
 *
 * With one argument, prints schema channel list + hash.  With two, also
 * prints the per-channel state + byte count for the chunk. */

#include "c3d.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: failed\n", path); exit(1); }
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "seek failed\n"); exit(1); }
    long sz = ftell(f);
    if (sz < 0) { fprintf(stderr, "tell failed\n"); exit(1); }
    rewind(f);
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "read short\n"); exit(1); }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

static void print_hash(const uint8_t h[16]) {
    for (int i = 0; i < 16; ++i) printf("%02x", h[i]);
}

static const char *state_name(uint8_t s) {
    switch (s) {
        case C3D_LABEL_STATE_ABSENT:  return "ABSENT";
        case C3D_LABEL_STATE_UNIFORM: return "UNIFORM";
        case C3D_LABEL_STATE_ENCODED: return "ENCODED";
        default:                      return "???";
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <schema.c3dls> [chunk.c3dl]\n", argv[0]);
        return 1;
    }

    size_t schema_len;
    uint8_t *schema_bytes = read_file(argv[1], &schema_len);
    c3d_label_schema *s = c3d_label_schema_parse(schema_bytes, schema_len);

    uint32_t n = c3d_label_schema_channel_count(s);
    printf("schema: %s\n", argv[1]);
    printf("  bytes    : %zu\n", schema_len);
    printf("  channels : %u\n", n);
    uint8_t h[16];
    c3d_label_schema_hash(s, h);
    printf("  hash     : "); print_hash(h); printf("\n");
    for (uint32_t i = 0; i < n; ++i) {
        printf("  [%u] name=%-24s num_values=%u\n",
               i, c3d_label_schema_channel_name(s, i),
               c3d_label_schema_channel_num_values(s, i));
    }

    if (argc == 3) {
        size_t chunk_len;
        uint8_t *chunk_bytes = read_file(argv[2], &chunk_len);

        if (!c3d_label_chunk_validate(s, chunk_bytes, chunk_len)) {
            fprintf(stderr, "chunk: structural validation FAILED\n");
            return 2;
        }

        c3d_label_chunk_info info;
        c3d_label_chunk_inspect(s, chunk_bytes, chunk_len, &info);

        printf("\nchunk: %s\n", argv[2]);
        printf("  bytes       : %zu\n", chunk_len);
        printf("  schema_hash : "); print_hash(info.schema_hash); printf("\n");
        printf("  chan_count  : %u\n", info.chan_count);
        for (uint32_t i = 0; i < info.chan_count; ++i) {
            printf("  [%u] %-10s", i, state_name(info.channel_state[i]));
            if (info.channel_state[i] == C3D_LABEL_STATE_UNIFORM) {
                printf(" value=%u\n", info.channel_uniform_value[i]);
            } else if (info.channel_state[i] == C3D_LABEL_STATE_ENCODED) {
                printf(" stream_bytes=%u\n", info.channel_stream_bytes[i]);
            } else {
                printf("\n");
            }
        }
        free(chunk_bytes);
    }

    c3d_label_schema_free(s);
    free(schema_bytes);
    return 0;
}
