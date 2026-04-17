/* c3d_train_dz — grid-search over uniform dz_ratio values, emit best-
 * performing .c3dx.  Iterates candidate dz values in [0.40, 0.70], encodes
 * the corpus at a reference target_ratio, measures aggregate PSNR, picks
 * the value with the highest PSNR.  Output is a fully-populated .c3dx
 * (LL_REFERENCE + SUBBAND_FREQ_TABLES + DZ_RATIO) that c3d_bench / the
 * shard encoder can load via the existing ctx parameter.
 *
 * This is intentionally a *uniform* search (one dz per corpus) rather
 * than a per-subband coordinate descent — the goal is first to see if
 * the knob has any win at all on this data.  If yes, a per-subband
 * training variant is a follow-up.
 *
 * See LICENSE. */

#include "c3d.h"

#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define CHUNK_BYTES (256u * 256u * 256u)
#define N_SUBBANDS 36

static int read_chunk(const char *path, uint8_t *out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return -1; }
    size_t n = fread(out, 1, CHUNK_BYTES, fp);
    fclose(fp);
    return (n == CHUNK_BYTES) ? 0 : -1;
}

static double psnr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double e = (double)a[i] - (double)b[i]; sse += e * e;
    }
    if (sse <= 0.0) return 200.0;
    return 10.0 * log10(255.0 * 255.0 / (sse / (double)n));
}

/* Build a ctx with corpus freq tables + LL_REFERENCE + uniform dz_ratio.
 * Caller frees via c3d_ctx_free. */
static c3d_ctx *build_ctx_with_dz(uint8_t **chunks, size_t n_chunks,
                                  float dz_value) {
    c3d_ctx_builder *b = c3d_ctx_builder_new();
    for (size_t i = 0; i < n_chunks; ++i)
        c3d_ctx_builder_observe_chunk(b, chunks[i]);
    float dz_ratio[N_SUBBANDS];
    for (unsigned s = 0; s < N_SUBBANDS; ++s) dz_ratio[s] = dz_value;
    c3d_ctx_builder_set_dz_ratio(b, dz_ratio);
    return c3d_ctx_builder_finish(b, /*include_freq_tables=*/true);
}

static double measure_ctx_psnr(uint8_t **chunks, size_t n_chunks,
                               const c3d_ctx *ctx, float target_ratio) {
    c3d_encoder *enc = c3d_encoder_new();
    c3d_decoder *dec = c3d_decoder_new();
    uint8_t *enc_buf = aligned_alloc(32, c3d_chunk_encode_max_size());
    uint8_t *dec_buf = aligned_alloc(32, CHUNK_BYTES);
    if (!enc_buf || !dec_buf) { fprintf(stderr, "oom\n"); exit(1); }

    double psnr_sum = 0.0;
    double bytes_sum = 0.0;
    for (size_t i = 0; i < n_chunks; ++i) {
        size_t sz = c3d_encoder_chunk_encode(enc, chunks[i], target_ratio, ctx,
                                             enc_buf, c3d_chunk_encode_max_size());
        c3d_decoder_chunk_decode(dec, enc_buf, sz, ctx, dec_buf);
        psnr_sum += psnr_u8(chunks[i], dec_buf, CHUNK_BYTES);
        bytes_sum += (double)sz;
    }

    free(enc_buf); free(dec_buf);
    c3d_encoder_free(enc); c3d_decoder_free(dec);
    return psnr_sum / (double)n_chunks;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s <corpus_dir> <out.c3dx> [target_ratio]\n"
                "  Sweeps uniform dz_ratio ∈ {0.40 .. 0.70} at 0.05 step on\n"
                "  the corpus, picks the value maximizing avg PSNR at the\n"
                "  given target_ratio (default 50), emits .c3dx at out path.\n",
                argv[0]);
        return 2;
    }
    const char *corpus_dir = argv[1];
    const char *out_path   = argv[2];
    float target_ratio = (argc == 4) ? (float)atof(argv[3]) : 50.0f;

    /* Load corpus. */
    DIR *dir = opendir(corpus_dir);
    if (!dir) { perror(corpus_dir); return 1; }
    size_t cap = 64, n_chunks = 0;
    uint8_t **chunks = malloc(cap * sizeof *chunks);
    char (*names)[256] = malloc(cap * 256);
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof full, "%s/%s", corpus_dir, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode) ||
            (size_t)st.st_size != CHUNK_BYTES) continue;
        if (n_chunks == cap) {
            cap *= 2;
            chunks = realloc(chunks, cap * sizeof *chunks);
            names  = realloc(names,  cap * 256);
        }
        chunks[n_chunks] = aligned_alloc(32, CHUNK_BYTES);
        if (!chunks[n_chunks] || read_chunk(full, chunks[n_chunks]) != 0) {
            free(chunks[n_chunks]); continue;
        }
        snprintf(names[n_chunks], 256, "%s", ent->d_name);
        n_chunks++;
    }
    closedir(dir);
    if (n_chunks == 0) { fprintf(stderr, "no chunks in %s\n", corpus_dir); return 1; }
    fprintf(stderr, "loaded %zu chunks\n", n_chunks);

    /* Sweep uniform dz_ratio candidates. */
    static const float candidates[] = {
        0.40f, 0.45f, 0.50f, 0.55f, 0.60f, 0.625f, 0.65f, 0.675f,
        0.70f, 0.725f, 0.75f, 0.80f, 0.85f, 0.90f
    };
    const size_t n_cand = sizeof candidates / sizeof candidates[0];

    fprintf(stderr, "sweeping dz_ratio at target_ratio=%.1f on %zu chunks:\n",
            target_ratio, n_chunks);
    double best_psnr = -1.0;
    float best_dz = 0.55f;
    for (size_t k = 0; k < n_cand; ++k) {
        float dz = candidates[k];
        c3d_ctx *ctx = build_ctx_with_dz(chunks, n_chunks, dz);
        double psnr = measure_ctx_psnr(chunks, n_chunks, ctx, target_ratio);
        fprintf(stderr, "  dz=%.3f  PSNR=%.3f dB\n", dz, psnr);
        if (psnr > best_psnr) { best_psnr = psnr; best_dz = dz; }
        c3d_ctx_free(ctx);
    }
    fprintf(stderr, "best uniform dz_ratio = %.3f (PSNR = %.3f dB)\n",
            best_dz, best_psnr);

    /* Emit final ctx with the best dz_ratio. */
    c3d_ctx *final_ctx = build_ctx_with_dz(chunks, n_chunks, best_dz);
    size_t sz = c3d_ctx_serialized_size(final_ctx);
    uint8_t *out = malloc(sz);
    c3d_ctx_serialize(final_ctx, out, sz);
    FILE *fp = fopen(out_path, "wb");
    if (!fp) { perror(out_path); return 1; }
    if (fwrite(out, 1, sz, fp) != sz) { perror("fwrite"); return 1; }
    fclose(fp);
    free(out);
    c3d_ctx_free(final_ctx);

    for (size_t i = 0; i < n_chunks; ++i) free(chunks[i]);
    free(chunks); free(names);
    fprintf(stderr, "wrote %s (%zu bytes, dz_ratio=%.3f uniform)\n",
            out_path, sz, best_dz);
    return 0;
}
