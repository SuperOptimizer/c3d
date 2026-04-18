/* c3d_perf — tight encode+decode throughput microbench.  No openh264.
 *
 * Usage:  c3d_perf <chunk.u8>   (one 256^3 u8 file)
 * Output: encode/decode MB/s at several q values.
 *
 * Intended for compiler-flag tuning: same binary, same input, repeatable. */

/* Include c3d.c directly to reach internal statics (DWT, rANS, etc). */
#include "c3d.c"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHUNK_BYTES ((size_t)256*256*256)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <chunk.u8>\n", argv[0]); return 2; }

    uint8_t *in  = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *dec = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *enc = aligned_alloc(32, c3d_chunk_encode_max_size());
    if (!in || !dec || !enc) { fprintf(stderr, "oom\n"); return 1; }

    FILE *fp = fopen(argv[1], "rb"); if (!fp) { perror(argv[1]); return 1; }
    if (fread(in, 1, CHUNK_BYTES, fp) != CHUNK_BYTES) { perror("fread"); return 1; }
    fclose(fp);

    /* Reusable contexts to avoid 50-100 ms/call alloc churn in the timing. */
    c3d_encoder *encoder = c3d_encoder_new();
    c3d_decoder *decoder = c3d_decoder_new();

    /* Warm up caches / JIT / etc. */
    (void)c3d_encoder_chunk_encode_at_q(encoder, in, 0.1f, NULL, enc, c3d_chunk_encode_max_size());

    const float qs[] = { 1.0f/32.0f, 0.05f, 0.1f, 0.5f };
    const int N_REPEAT = 3;

    printf("%6s %12s %9s %9s %9s %9s\n",
           "q", "enc_bytes", "enc_ms", "enc_MB/s", "dec_ms", "dec_MB/s");
    for (size_t qi = 0; qi < sizeof qs / sizeof qs[0]; ++qi) {
        float q = qs[qi];
        size_t esz = 0;
        double t_enc = 0.0, t_dec = 0.0;
        for (int r = 0; r < N_REPEAT; ++r) {
            double t0 = now_s();
            esz = c3d_encoder_chunk_encode_at_q(encoder, in, q, NULL, enc, c3d_chunk_encode_max_size());
            t_enc += now_s() - t0;

            t0 = now_s();
            c3d_decoder_chunk_decode(decoder, enc, esz, NULL, dec);
            t_dec += now_s() - t0;
        }
        double enc_ms = 1000.0 * t_enc / N_REPEAT;
        double dec_ms = 1000.0 * t_dec / N_REPEAT;
        double enc_mbps = ((double)CHUNK_BYTES / (1024.0 * 1024.0)) / (t_enc / N_REPEAT);
        double dec_mbps = ((double)CHUNK_BYTES / (1024.0 * 1024.0)) / (t_dec / N_REPEAT);
        printf("%6.4f %12zu %9.1f %9.1f %9.1f %9.1f\n",
               (double)q, esz, enc_ms, enc_mbps, dec_ms, dec_mbps);
    }

    /* LOD decode sweep: decode_lod(k) for k = 0..5 at a fixed q.  Reports the
     * bytes touched (from lod_offsets) and decode throughput per LOD.  Higher
     * k = coarser resolution = fewer bytes read and fewer IDWT levels run. */
    {
        float q_fixed = 0.1f;
        size_t esz = c3d_encoder_chunk_encode_at_q(encoder, in, q_fixed, NULL,
                                                   enc, c3d_chunk_encode_max_size());
        c3d_chunk_info info;
        c3d_chunk_inspect(enc, esz, &info);

        /* Per-LOD output buffer size: (256 >> lod)^3 voxels. */
        uint8_t *out_lod = aligned_alloc(32, CHUNK_BYTES);

        printf("\nLOD sweep (q=%.3f, chunk_bytes=%zu):\n", (double)q_fixed, esz);
        printf("%3s %6s %12s %10s %10s %10s\n",
               "lod", "side", "bytes_read", "out_vox", "dec_ms", "dec_MB/s_out");
        for (int lod = (int)C3D_N_LODS - 1; lod >= 0; --lod) {
            uint32_t side = 256u >> lod;
            size_t out_vox = (size_t)side * side * side;
            /* lod_offsets[k] = cumulative bytes needed up to LOD k (entropy).
             * Actual payload read bytes ≈ FIXED_SIZE + lod_offsets[lod]. */
            size_t bytes_read = C3D_CHUNK_FIXED_SIZE + info.lod_offsets[lod];

            const int REPS = 5;
            double t0 = now_s();
            for (int r = 0; r < REPS; ++r) {
                c3d_decoder_chunk_decode_lod(decoder, enc, esz, (uint8_t)lod,
                                             NULL, out_lod);
            }
            double t = (now_s() - t0) / REPS;
            double mbps_out = ((double)out_vox / (1024.0 * 1024.0)) / t;
            printf("%3d %6u %12zu %10zu %10.2f %10.1f\n",
                   lod, side, bytes_read, out_vox, t * 1000.0, mbps_out);
        }
        free(out_lod);
    }

    /* Raw DWT micro-measurement (no encode/decode, just DWT forward+inverse). */
    {
        float *coef = aligned_alloc(32, (size_t)256*256*256 * sizeof(float));
        float scratch[2 * C3D_TILE_X * 256];
        for (size_t i = 0; i < CHUNK_BYTES; ++i) coef[i] = (float)in[i] - 128.0f;
        /* Warm */
        c3d_dwt3_fwd(coef, scratch);
        c3d_dwt3_inv_levels(coef, 5, scratch);
        const int REPS = 3;
        double t0 = now_s();
        for (int r = 0; r < REPS; ++r) {
            c3d_dwt3_fwd(coef, scratch);
            c3d_dwt3_inv_levels(coef, 5, scratch);
        }
        double t = (now_s() - t0) / REPS;
        double mbps = ((double)CHUNK_BYTES * sizeof(float) / (1024.0*1024.0)) / t;
        printf("\nDWT fwd+inv (64 MiB f32): %.1f ms/iter, %.1f MB/s f32\n",
               t * 1000.0, mbps);
        free(coef);
    }

    c3d_encoder_free(encoder); c3d_decoder_free(decoder);
    free(in); free(dec); free(enc);
    return 0;
}
