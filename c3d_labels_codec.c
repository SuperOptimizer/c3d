/* c3d_labels_codec — round-trip CLI for the labels codec.  See LICENSE.
 *
 * Usage:
 *   c3d_labels_codec mkschema SCHEMA.c3dls NAME:NUM_VALUES [NAME:NUM_VALUES ...]
 *   c3d_labels_codec encode   SCHEMA.c3dls CHUNK.c3dl CHAN0.raw [CHAN1.raw ...]
 *   c3d_labels_codec decode   SCHEMA.c3dls CHUNK.c3dl CHAN0.raw [CHAN1.raw ...]
 *
 * Each CHAN*.raw is either a 16 MiB 256^3 u8 file or "-" to mean NULL (the
 * encoder encodes NULL as ABSENT; the decoder skips NULL slots on the way
 * out).  Argument order follows the schema's channel order. */

#include "c3d.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: failed\n", path); exit(1); }
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "seek %s\n", path); exit(1); }
    long sz = ftell(f);
    if (sz < 0) { fprintf(stderr, "tell %s\n", path); exit(1); }
    rewind(f);
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "read %s short\n", path); exit(1); }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

/* Read exactly 16 MiB into an aligned buffer.  Returns NULL if path is "-" */
static uint8_t *read_chan_raw(const char *path) {
    if (strcmp(path, "-") == 0) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: failed\n", path); exit(1); }
    uint8_t *buf = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    size_t nr = fread(buf, 1, C3D_VOXELS_PER_CHUNK, f);
    if (nr != C3D_VOXELS_PER_CHUNK) {
        fprintf(stderr, "%s: expected %zu bytes, got %zu\n",
                path, (size_t)C3D_VOXELS_PER_CHUNK, nr);
        exit(1);
    }
    fclose(f);
    return buf;
}

static void write_file(const char *path, const uint8_t *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s for write: failed\n", path); exit(1); }
    if (fwrite(buf, 1, n, f) != n) { fprintf(stderr, "write %s short\n", path); exit(1); }
    fclose(f);
}

static int cmd_mkschema(int argc, char **argv) {
    /* argv: mkschema SCHEMA NAME:NUM [NAME:NUM ...] */
    if (argc < 4) {
        fprintf(stderr, "mkschema: need schema path + at least one NAME:NUM\n");
        return 1;
    }
    const char *schema_path = argv[2];
    c3d_label_schema *s = c3d_label_schema_new();
    for (int i = 3; i < argc; ++i) {
        char *colon = strchr(argv[i], ':');
        if (!colon) {
            fprintf(stderr, "mkschema: arg '%s' missing ':'\n", argv[i]);
            return 1;
        }
        *colon = 0;
        const char *name = argv[i];
        int nv = atoi(colon + 1);
        if (nv < 2 || nv > 255) {
            fprintf(stderr, "mkschema: num_values %d out of [2,255]\n", nv);
            return 1;
        }
        c3d_label_schema_add_channel(s, name, (uint8_t)nv);
    }
    size_t sz = c3d_label_schema_serialized_size(s);
    uint8_t *buf = malloc(sz);
    if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
    size_t w = c3d_label_schema_serialize(s, buf, sz);
    write_file(schema_path, buf, w);
    fprintf(stderr, "mkschema: wrote %zu bytes to %s (%u channels)\n",
            w, schema_path, c3d_label_schema_channel_count(s));
    free(buf);
    c3d_label_schema_free(s);
    return 0;
}

static int cmd_encode(int argc, char **argv) {
    /* argv: encode SCHEMA CHUNK CHAN0 [CHAN1 ...] */
    if (argc < 5) {
        fprintf(stderr, "encode: need schema + chunk + at least one channel\n");
        return 1;
    }
    const char *schema_path = argv[2];
    const char *chunk_path  = argv[3];
    int n_chans = argc - 4;

    size_t schema_len;
    uint8_t *schema_bytes = read_file(schema_path, &schema_len);
    c3d_label_schema *s = c3d_label_schema_parse(schema_bytes, schema_len);

    uint32_t n = c3d_label_schema_channel_count(s);
    if ((int)n != n_chans) {
        fprintf(stderr, "encode: schema has %u channels, got %d arg(s)\n", n, n_chans);
        return 1;
    }

    const uint8_t **channels = calloc(n, sizeof *channels);
    if (!channels) { fprintf(stderr, "oom\n"); exit(1); }
    for (uint32_t i = 0; i < n; ++i) {
        channels[i] = read_chan_raw(argv[4 + i]);
    }

    c3d_label_encoder *e = c3d_label_encoder_new(s);
    size_t cap = c3d_label_encoder_max_chunk_size(e);
    uint8_t *out = malloc(cap);
    if (!out) { fprintf(stderr, "oom\n"); exit(1); }
    size_t nb = c3d_label_encoder_chunk_encode(e, channels, out, cap);
    write_file(chunk_path, out, nb);

    fprintf(stderr, "encode: wrote %zu bytes to %s\n", nb, chunk_path);

    for (uint32_t i = 0; i < n; ++i) {
        void *p = (void *)(uintptr_t)channels[i];
        free(p);
    }
    free(channels); free(out);
    c3d_label_encoder_free(e);
    c3d_label_schema_free(s);
    free(schema_bytes);
    return 0;
}

static int cmd_decode(int argc, char **argv) {
    /* argv: decode SCHEMA CHUNK CHAN0 [CHAN1 ...] */
    if (argc < 5) {
        fprintf(stderr, "decode: need schema + chunk + at least one channel\n");
        return 1;
    }
    const char *schema_path = argv[2];
    const char *chunk_path  = argv[3];
    int n_chans = argc - 4;

    size_t schema_len, chunk_len;
    uint8_t *schema_bytes = read_file(schema_path, &schema_len);
    uint8_t *chunk_bytes  = read_file(chunk_path, &chunk_len);
    c3d_label_schema *s = c3d_label_schema_parse(schema_bytes, schema_len);

    uint32_t n = c3d_label_schema_channel_count(s);
    if ((int)n != n_chans) {
        fprintf(stderr, "decode: schema has %u channels, got %d output arg(s)\n", n, n_chans);
        return 1;
    }

    uint8_t **outs = calloc(n, sizeof *outs);
    if (!outs) { fprintf(stderr, "oom\n"); exit(1); }
    for (uint32_t i = 0; i < n; ++i) {
        if (strcmp(argv[4 + i], "-") == 0) {
            outs[i] = NULL;
        } else {
            outs[i] = aligned_alloc(C3D_ALIGN, C3D_VOXELS_PER_CHUNK);
            if (!outs[i]) { fprintf(stderr, "oom\n"); exit(1); }
        }
    }

    c3d_label_decoder *d = c3d_label_decoder_new(s);
    c3d_label_decoder_chunk_decode(d, chunk_bytes, chunk_len, outs);

    for (uint32_t i = 0; i < n; ++i) {
        if (outs[i]) {
            write_file(argv[4 + i], outs[i], C3D_VOXELS_PER_CHUNK);
            free(outs[i]);
        }
    }
    free(outs);
    fprintf(stderr, "decode: %u channel(s) written\n", n);

    c3d_label_decoder_free(d);
    c3d_label_schema_free(s);
    free(schema_bytes);
    free(chunk_bytes);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage:\n"
            "  %s mkschema SCHEMA.c3dls NAME:NUM [NAME:NUM ...]\n"
            "  %s encode   SCHEMA.c3dls CHUNK.c3dl CHAN0.raw [CHAN1.raw ...]\n"
            "  %s decode   SCHEMA.c3dls CHUNK.c3dl CHAN0.raw [CHAN1.raw ...]\n"
            "  Use \"-\" as a channel path to mean NULL (absent/skip).\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "mkschema") == 0) return cmd_mkschema(argc, argv);
    if (strcmp(argv[1], "encode")   == 0) return cmd_encode(argc, argv);
    if (strcmp(argv[1], "decode")   == 0) return cmd_decode(argc, argv);
    fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 1;
}
