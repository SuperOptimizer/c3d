/* c3d_bench — byte-budget-matched benchmark of c3d vs H.264 intra (openh264).
 *
 * For each corpus chunk and each H.264 QP, we:
 *   1. Encode via openh264 intra-only (256 gray 256×256 frames), measure size +
 *      PSNR.
 *   2. Target that SAME byte budget with c3d via target_ratio = CHUNK_BYTES /
 *      h264_size, measure c3d size + PSNR.  (c3d's rate control lands within
 *      ~2-10% of the target, so the comparison is genuinely size-matched.)
 *   3. Report the PSNR delta at each (size, size) point.  This replaces the
 *      earlier ratio-target method which let h264 saturate at QP=51.
 */

#include "c3d.h"

#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <wels/codec_api.h>
#include <wels/codec_def.h>

#define CHUNK_SIDE   256u
#define CHUNK_BYTES  ((size_t)CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double psnr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double e = (double)a[i] - (double)b[i]; sse += e * e;
    }
    if (sse <= 0.0) return 200.0;
    return 10.0 * log10(255.0 * 255.0 / (sse / (double)n));
}

/* Encode a 256³ u8 volume as 256 intra-coded Y-plane frames at fixed QP.
 * Fills `out_yuv` with the openh264-decoded volume.  Returns encoded bytes. */
static size_t openh264_encode_decode(const uint8_t *in, int qp, uint8_t *out_yuv) {
    const int W = (int)CHUNK_SIDE, H = (int)CHUNK_SIDE;
    const size_t Y = (size_t)W * H, UV = Y / 4;

    ISVCEncoder *enc = NULL;
    if (WelsCreateSVCEncoder(&enc) != 0 || !enc) { fprintf(stderr, "enc create\n"); exit(1); }

    SEncParamExt ep = {0};
    (*enc)->GetDefaultParams(enc, &ep);
    ep.iPicWidth = W; ep.iPicHeight = H;
    ep.iRCMode = RC_OFF_MODE; ep.iTargetBitrate = 0;
    ep.fMaxFrameRate = 30;
    ep.iUsageType = SCREEN_CONTENT_REAL_TIME;  /* enables more expensive intra tools */
    ep.iSpatialLayerNum = 1;
    ep.sSpatialLayers[0].iVideoWidth  = W;
    ep.sSpatialLayers[0].iVideoHeight = H;
    ep.sSpatialLayers[0].fFrameRate   = 30;
    ep.sSpatialLayers[0].iDLayerQp    = qp;
    ep.iTemporalLayerNum = 1;
    ep.uiIntraPeriod = 0;  /* single-GOP: 1 I-frame + 255 P-frames per chunk,
                            * letting H.264 exploit z-axis (temporal) correlation
                            * the same way c3d's 3D DWT does.  All-I handicaps
                            * H.264 by discarding inter-slice redundancy. */
    ep.eSpsPpsIdStrategy = CONSTANT_ID;
    ep.iMultipleThreadIdc = 1;
    ep.iComplexityMode = HIGH_COMPLEXITY;
    ep.iMaxQp = qp; ep.iMinQp = qp;
    ep.iEntropyCodingModeFlag = 1;             /* CABAC (+10-15 % over CAVLC) */
    ep.bEnableAdaptiveQuant = true;            /* per-MB QP adaptation */
    ep.bEnableBackgroundDetection = false;     /* irrelevant for all-intra */
    ep.bEnableFrameCroppingFlag = false;

    if ((*enc)->InitializeExt(enc, &ep) != 0) { fprintf(stderr, "enc init\n"); exit(1); }
    int pf = videoFormatI420;
    (*enc)->SetOption(enc, ENCODER_OPTION_DATAFORMAT, &pf);

    uint8_t *y = malloc(Y), *u = malloc(UV), *v = malloc(UV);
    memset(u, 128, UV); memset(v, 128, UV);

    size_t cap = 4u * 1024 * 1024, total = 0;
    uint8_t *bitstream = malloc(cap);

    SFrameBSInfo info; SSourcePicture pic = {0};
    pic.iPicWidth = W; pic.iPicHeight = H; pic.iColorFormat = videoFormatI420;
    pic.iStride[0] = W; pic.iStride[1] = W/2; pic.iStride[2] = W/2;
    pic.pData[0] = y; pic.pData[1] = u; pic.pData[2] = v;

    for (int z = 0; z < H; ++z) {
        memcpy(y, in + (size_t)z * Y, Y);
        pic.uiTimeStamp = z;
        memset(&info, 0, sizeof info);
        if ((*enc)->EncodeFrame(enc, &pic, &info) != cmResultSuccess) {
            fprintf(stderr, "EncodeFrame z=%d fail\n", z); exit(1);
        }
        for (int l = 0; l < info.iLayerNum; ++l) {
            const SLayerBSInfo *lb = &info.sLayerInfo[l];
            int lsz = 0;
            for (int n = 0; n < lb->iNalCount; ++n) lsz += lb->pNalLengthInByte[n];
            if (total + (size_t)lsz > cap) {
                while (total + (size_t)lsz > cap) cap *= 2;
                bitstream = realloc(bitstream, cap);
            }
            memcpy(bitstream + total, lb->pBsBuf, (size_t)lsz); total += lsz;
        }
    }
    (*enc)->Uninitialize(enc); WelsDestroySVCEncoder(enc);

    ISVCDecoder *dec = NULL;
    WelsCreateDecoder(&dec);
    SDecodingParam dp = {0};
    dp.uiTargetDqLayer = UCHAR_MAX;
    dp.eEcActiveIdc = ERROR_CON_SLICE_COPY;
    dp.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
    (*dec)->Initialize(dec, &dp);

    uint8_t *decbuf[3] = {0,0,0}; SBufferInfo bi;
    size_t scan = 0, prev_start = 0, decoded = 0; bool have_prev = false;
    while (scan + 3 < total) {
        bool sc = (bitstream[scan] == 0 && bitstream[scan+1] == 0 &&
                   bitstream[scan+2] == 0 && bitstream[scan+3] == 1);
        if (sc) {
            if (have_prev) {
                memset(&bi, 0, sizeof bi);
                (*dec)->DecodeFrameNoDelay(dec, bitstream + prev_start, (int)(scan - prev_start), decbuf, &bi);
                if (bi.iBufferStatus == 1) {
                    int sy = bi.UsrData.sSystemBuffer.iStride[0];
                    for (int r = 0; r < H; ++r)
                        memcpy(out_yuv + decoded * Y + r * W, decbuf[0] + r * sy, W);
                    decoded++;
                }
            }
            prev_start = scan; have_prev = true; scan += 4;
        } else scan++;
    }
    if (have_prev) {
        memset(&bi, 0, sizeof bi);
        (*dec)->DecodeFrameNoDelay(dec, bitstream + prev_start, (int)(total - prev_start), decbuf, &bi);
        if (bi.iBufferStatus == 1) {
            int sy = bi.UsrData.sSystemBuffer.iStride[0];
            for (int r = 0; r < H; ++r)
                memcpy(out_yuv + decoded * Y + r * W, decbuf[0] + r * sy, W);
            decoded++;
        }
    }
    memset(&bi, 0, sizeof bi);
    (*dec)->FlushFrame(dec, decbuf, &bi);
    while (bi.iBufferStatus == 1 && decoded < (size_t)H) {
        int sy = bi.UsrData.sSystemBuffer.iStride[0];
        for (int r = 0; r < H; ++r)
            memcpy(out_yuv + decoded * Y + r * W, decbuf[0] + r * sy, W);
        decoded++;
        memset(&bi, 0, sizeof bi);
        (*dec)->FlushFrame(dec, decbuf, &bi);
    }
    (*dec)->Uninitialize(dec); WelsDestroyDecoder(dec);

    if (decoded != (size_t)H) fprintf(stderr, "warning: openh264 decoded %zu/%d at qp=%d\n", decoded, H, qp);

    free(y); free(u); free(v); free(bitstream);
    return total;
}

/* QP sweep: span low (near-lossless) through high (aggressive). */
static const int g_qps[] = { 18, 24, 30, 36, 42, 48 };
#define N_QPS (sizeof g_qps / sizeof g_qps[0])

int main(int argc, char **argv) {
    const char *dir_path = (argc > 1) ? argv[1] : getenv("C3D_CORPUS");
    if (!dir_path) { fprintf(stderr, "usage: %s <corpus_dir>\n", argv[0]); return 2; }
    DIR *dir = opendir(dir_path);
    if (!dir) { perror(dir_path); return 1; }

    uint8_t *in       = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *dec      = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *enc      = aligned_alloc(32, c3d_chunk_encode_max_size());
    uint8_t *h_dec    = aligned_alloc(32, CHUNK_BYTES);
    if (!in || !dec || !enc || !h_dec) { fprintf(stderr, "oom\n"); return 1; }

    /* Reusable c3d encoder/decoder — shaves ~50 ms/chunk of malloc churn. */
    c3d_encoder *enc_ctx = c3d_encoder_new();
    c3d_decoder *dec_ctx = c3d_decoder_new();

    /* Aggregate over the corpus: for each QP, sum h264/c3d size+PSNR. */
    double h_sz_sum[N_QPS] = {0}, h_p_sum[N_QPS] = {0};
    double c_sz_sum[N_QPS] = {0}, c_p_sum[N_QPS] = {0};
    double c_enc_s[N_QPS] = {0}, c_dec_s[N_QPS] = {0};
    size_t n_chunks = 0;

    printf("Chunk-by-chunk (c3d matched to h264 byte budget):\n");
    printf("%-40s | %s\n", "chunk", "  QP   h264_sz  h264_PSNR | c3d_sz  c3d_PSNR  dec_MB/s  Δ");
    printf("%-40s | %s\n", "-----", "  --   -------  ---------   -----   --------  --------  -");

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof full, "%s/%s", dir_path, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode) || (size_t)st.st_size != CHUNK_BYTES) continue;
        FILE *fp = fopen(full, "rb");
        if (!fp || fread(in, 1, CHUNK_BYTES, fp) != CHUNK_BYTES) {
            if (fp) fclose(fp); continue;
        }
        fclose(fp);

        for (size_t q = 0; q < N_QPS; ++q) {
            int qp = g_qps[q];
            size_t h_sz = openh264_encode_decode(in, qp, h_dec);
            double h_p  = psnr_u8(in, h_dec, CHUNK_BYTES);

            /* Match c3d's byte budget to h264's output size. */
            float target_r = (float)((double)CHUNK_BYTES / (double)h_sz);
            if (target_r <= 1.01f) target_r = 1.01f;   /* degenerate near-lossless */
            double t0 = now_s();
            size_t c_sz = c3d_encoder_chunk_encode(enc_ctx, in, target_r, NULL, enc, c3d_chunk_encode_max_size());
            double t_enc = now_s() - t0;

            t0 = now_s();
            c3d_decoder_chunk_decode(dec_ctx, enc, c_sz, NULL, dec);
            double t_dec = now_s() - t0;
            double c_p  = psnr_u8(in, dec, CHUNK_BYTES);

            double c_dec_mbps = ((double)CHUNK_BYTES / (1024.0*1024.0)) / t_dec;

            printf("%-40s | QP%2d %8zu   %7.2f     %7zu  %7.2f   %7.1f  %+.2f\n",
                   ent->d_name, qp, h_sz, h_p, c_sz, c_p, c_dec_mbps, c_p - h_p);

            h_sz_sum[q] += (double)h_sz; h_p_sum[q] += h_p;
            c_sz_sum[q] += (double)c_sz; c_p_sum[q] += c_p;
            c_enc_s[q]  += t_enc;        c_dec_s[q] += t_dec;
        }
        n_chunks++;
    }
    closedir(dir);

    if (n_chunks == 0) { fprintf(stderr, "no chunks in %s\n", dir_path); return 1; }

    printf("\nSummary averaged over %zu chunks:\n", n_chunks);
    printf("  QP    h264_avg_sz  h264_avg_PSNR | c3d_avg_sz  c3d_avg_PSNR  enc_MB/s  dec_MB/s | Δ_PSNR\n");
    printf("  --    -----------  -------------   ----------  ------------  --------  -------- | ------\n");
    for (size_t q = 0; q < N_QPS; ++q) {
        double h_sz = h_sz_sum[q] / n_chunks;
        double h_p  = h_p_sum[q]  / n_chunks;
        double c_sz = c_sz_sum[q] / n_chunks;
        double c_p  = c_p_sum[q]  / n_chunks;
        double enc_mbps = ((double)n_chunks * (double)CHUNK_BYTES) / (c_enc_s[q] * 1024.0 * 1024.0);
        double dec_mbps = ((double)n_chunks * (double)CHUNK_BYTES) / (c_dec_s[q] * 1024.0 * 1024.0);
        double ratio_h = (double)CHUNK_BYTES / h_sz;
        printf("  %2d  %11.0f  %13.2f   %10.0f  %12.2f    %6.1f    %6.1f  | %+6.2f  (~%.1f:1)\n",
               g_qps[q], h_sz, h_p, c_sz, c_p, enc_mbps, dec_mbps, c_p - h_p, ratio_h);
    }
    c3d_encoder_free(enc_ctx); c3d_decoder_free(dec_ctx);
    free(in); free(dec); free(enc); free(h_dec);
    return 0;
}
