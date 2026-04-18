/* c3d_train_dz — corpus training for per-subband quant parameters.
 *
 * Two-pass coordinate descent:
 *   Pass A — per-kind dead-zone ratio (4 h-count groups: 0,1,2,3).
 *   Pass B — per-kind Laplacian α (same grouping).
 *
 * Each candidate measures PSNR + MAE + max_error aggregated over the
 * corpus at a given target_ratio.  Best-PSNR value is picked per kind,
 * but MAE and max_error are reported so the operator can eyeball the
 * trade-off on tail metrics.  Output is a .c3dx carrying
 * LL_REFERENCE + SUBBAND_FREQ_TABLES + DZ_RATIO + LAPLACIAN_ALPHA.
 *
 * Usage: c3d_train_dz <corpus_dir> <out.c3dx> [target_ratio]
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

/* Map subband index → h_count group (0..3) using the same layout as
 * c3d_kind_h_count().  Duplicated here because the helper is static.
 * Index layout from c3d_subband_info_of: kind 0 = LLL_5 (s=0); kinds
 * 1..7 cycle per level for 5 levels → s = 1 + (level-1)*7 + (kind-1). */
static unsigned subband_h_count(unsigned s) {
    if (s == 0) return 0;                        /* LLL_5 */
    unsigned k = ((s - 1u) % 7u) + 1u;           /* kind 1..7 */
    switch (k) {
    case 1: return 3;                            /* HHH        */
    case 2: case 3: case 4: return 2;            /* HHL/HLH/LHH */
    case 5: case 6: case 7: return 1;            /* HLL/LHL/LLH */
    default: return 0;
    }
}

typedef struct {
    double psnr;        /* avg over corpus */
    double mae;         /* avg */
    double max_err;     /* max over all voxels across corpus */
    double avg_bytes;
} metric_row;

static metric_row measure(uint8_t **chunks, size_t n_chunks,
                          const float dz[N_SUBBANDS],
                          const float alpha[N_SUBBANDS],
                          float target_ratio) {
    c3d_ctx_builder *b = c3d_ctx_builder_new();
    for (size_t i = 0; i < n_chunks; ++i)
        c3d_ctx_builder_observe_chunk(b, chunks[i]);
    c3d_ctx_builder_set_dz_ratio(b, dz);
    c3d_ctx_builder_set_laplacian_alpha(b, alpha);
    c3d_ctx *ctx = c3d_ctx_builder_finish(b, /*include_freq_tables=*/true);

    c3d_encoder *enc = c3d_encoder_new();
    c3d_decoder *dec = c3d_decoder_new();
    uint8_t *enc_buf = aligned_alloc(32, c3d_chunk_encode_max_size());
    uint8_t *dec_buf = aligned_alloc(32, CHUNK_BYTES);
    if (!enc_buf || !dec_buf) { fprintf(stderr, "oom\n"); exit(1); }

    double psnr_sum = 0.0;
    double mae_sum = 0.0;
    double max_err = 0.0;
    double bytes_sum = 0.0;
    for (size_t i = 0; i < n_chunks; ++i) {
        size_t sz = c3d_encoder_chunk_encode(enc, chunks[i], target_ratio, ctx,
                                             enc_buf, c3d_chunk_encode_max_size());
        c3d_decoder_chunk_decode(dec, enc_buf, sz, ctx, dec_buf);
        bytes_sum += (double)sz;

        double sse = 0.0;
        double mae = 0.0;
        double chunk_max = 0.0;
        for (size_t k = 0; k < CHUNK_BYTES; ++k) {
            double e = (double)chunks[i][k] - (double)dec_buf[k];
            double a = e < 0 ? -e : e;
            sse += e * e;
            mae += a;
            if (a > chunk_max) chunk_max = a;
        }
        double psnr = sse > 0.0
            ? 10.0 * log10(255.0 * 255.0 / (sse / (double)CHUNK_BYTES))
            : 200.0;
        psnr_sum += psnr;
        mae_sum += mae / (double)CHUNK_BYTES;
        if (chunk_max > max_err) max_err = chunk_max;
    }

    free(enc_buf); free(dec_buf);
    c3d_encoder_free(enc); c3d_decoder_free(dec);
    c3d_ctx_free(ctx);

    metric_row r = {
        .psnr = psnr_sum / (double)n_chunks,
        .mae = mae_sum / (double)n_chunks,
        .max_err = max_err,
        .avg_bytes = bytes_sum / (double)n_chunks,
    };
    return r;
}

/* Sweep one kind's parameter (dz or alpha) across `candidates`, holding
 * everything else fixed.  Updates the best value into *out_best, prints
 * all candidate metrics.  `is_alpha` controls whether we modify dz or
 * alpha; the other array stays as passed in. */
static void sweep_kind(uint8_t **chunks, size_t n_chunks,
                       float dz[N_SUBBANDS], float alpha[N_SUBBANDS],
                       unsigned target_h,
                       const float *candidates, size_t n_cand,
                       float target_ratio, bool is_alpha,
                       float *out_best) {
    double best_psnr = -1.0;
    float  best_val = is_alpha ? alpha[0] : dz[0];
    /* Pick a subband in the target group to read/write — we mutate every
     * subband with matching h_count. */
    for (size_t k = 0; k < n_cand; ++k) {
        float v = candidates[k];
        float saved[N_SUBBANDS];
        memcpy(saved, is_alpha ? alpha : dz, sizeof saved);
        for (unsigned s = 0; s < N_SUBBANDS; ++s) {
            if (subband_h_count(s) != target_h) continue;
            if (is_alpha) alpha[s] = v;
            else          dz[s]    = v;
        }
        metric_row m = measure(chunks, n_chunks, dz, alpha, target_ratio);
        fprintf(stderr, "    %s=%.3f  PSNR=%.3f  MAE=%.3f  maxE=%.0f  bytes=%.0f\n",
                is_alpha ? "α" : "dz", (double)v, m.psnr, m.mae, m.max_err,
                m.avg_bytes);
        if (m.psnr > best_psnr) { best_psnr = m.psnr; best_val = v; }
        /* Restore the other groups before the next candidate iterates. */
        memcpy(is_alpha ? alpha : dz, saved, sizeof saved);
    }
    /* Apply the best value permanently for all target-h subbands. */
    for (unsigned s = 0; s < N_SUBBANDS; ++s) {
        if (subband_h_count(s) != target_h) continue;
        if (is_alpha) alpha[s] = best_val;
        else          dz[s]    = best_val;
    }
    *out_best = best_val;
    fprintf(stderr, "  → best %s[h=%u] = %.3f (PSNR=%.3f dB)\n",
            is_alpha ? "α" : "dz", target_h, (double)best_val, best_psnr);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr,
                "usage: %s <corpus_dir> <out.c3dx> [target_ratio]\n"
                "  Per-kind coordinate descent on dz then α, emits .c3dx.\n",
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
        }
        chunks[n_chunks] = aligned_alloc(32, CHUNK_BYTES);
        if (!chunks[n_chunks] || read_chunk(full, chunks[n_chunks]) != 0) {
            free(chunks[n_chunks]); continue;
        }
        n_chunks++;
    }
    closedir(dir);
    if (n_chunks == 0) { fprintf(stderr, "no chunks in %s\n", corpus_dir); return 1; }
    fprintf(stderr, "loaded %zu chunks, target_ratio=%.1f\n",
            n_chunks, (double)target_ratio);

    /* Starting point: uniform dz = 0.55 (library default), kind-based α. */
    float dz[N_SUBBANDS], alpha[N_SUBBANDS];
    for (unsigned s = 0; s < N_SUBBANDS; ++s) {
        dz[s] = 0.55f;
        /* Mirror c3d_default_alpha() — library static helper not exported. */
        if (s == 0) { alpha[s] = 0.45f; continue; }
        switch (subband_h_count(s)) {
        case 1: alpha[s] = 0.40f;  break;
        case 2: alpha[s] = 0.375f; break;
        case 3: alpha[s] = 0.33f;  break;
        default: alpha[s] = 0.375f;
        }
    }

    /* Baseline row. */
    metric_row base = measure(chunks, n_chunks, dz, alpha, target_ratio);
    fprintf(stderr, "baseline  PSNR=%.3f  MAE=%.3f  maxE=%.0f  bytes=%.0f\n",
            base.psnr, base.mae, base.max_err, base.avg_bytes);

    /* Pass A — dz sweep per kind. */
    static const float dz_cand[] = {
        0.45f, 0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f
    };
    const size_t n_dz = sizeof dz_cand / sizeof dz_cand[0];
    fprintf(stderr, "--- Pass A: dz per kind ---\n");
    for (unsigned h = 0; h <= 3; ++h) {
        fprintf(stderr, "  kind h=%u:\n", h);
        float best;
        sweep_kind(chunks, n_chunks, dz, alpha, h,
                   dz_cand, n_dz, target_ratio, /*is_alpha=*/false, &best);
    }

    /* Pass B — α sweep per kind at the best dz. */
    static const float a_cand[] = {
        0.28f, 0.33f, 0.375f, 0.40f, 0.42f, 0.45f, 0.48f, 0.50f
    };
    const size_t n_a = sizeof a_cand / sizeof a_cand[0];
    fprintf(stderr, "--- Pass B: α per kind ---\n");
    for (unsigned h = 0; h <= 3; ++h) {
        fprintf(stderr, "  kind h=%u:\n", h);
        float best;
        sweep_kind(chunks, n_chunks, dz, alpha, h,
                   a_cand, n_a, target_ratio, /*is_alpha=*/true, &best);
    }

    /* Final row. */
    metric_row final_m = measure(chunks, n_chunks, dz, alpha, target_ratio);
    fprintf(stderr,
            "final     PSNR=%.3f (Δ=%+0.3f)  MAE=%.3f (Δ=%+0.3f)  "
            "maxE=%.0f (Δ=%+0.0f)  bytes=%.0f\n",
            final_m.psnr, final_m.psnr - base.psnr,
            final_m.mae, final_m.mae - base.mae,
            final_m.max_err, final_m.max_err - base.max_err,
            final_m.avg_bytes);

    /* Emit final ctx. */
    c3d_ctx_builder *b = c3d_ctx_builder_new();
    for (size_t i = 0; i < n_chunks; ++i)
        c3d_ctx_builder_observe_chunk(b, chunks[i]);
    c3d_ctx_builder_set_dz_ratio(b, dz);
    c3d_ctx_builder_set_laplacian_alpha(b, alpha);
    c3d_ctx *ctx = c3d_ctx_builder_finish(b, /*include_freq_tables=*/true);
    size_t sz = c3d_ctx_serialized_size(ctx);
    uint8_t *out = malloc(sz);
    c3d_ctx_serialize(ctx, out, sz);
    FILE *fp = fopen(out_path, "wb");
    if (!fp) { perror(out_path); return 1; }
    if (fwrite(out, 1, sz, fp) != sz) { perror("fwrite"); return 1; }
    fclose(fp);
    free(out);
    c3d_ctx_free(ctx);

    for (size_t i = 0; i < n_chunks; ++i) free(chunks[i]);
    free(chunks);
    fprintf(stderr, "wrote %s (%zu bytes)\n", out_path, sz);
    return 0;
}
