/* c3d_bench — byte-budget-matched benchmark of c3d vs H.264 / H.265 / AV1.
 *
 * For each corpus chunk and each (codec, QP) point we:
 *   1. Encode the 256³ chunk as 256 intra+P-predicted 256×256 Y-plane frames
 *      with one video codec, measure encoded size + PSNR after decode.
 *   2. Target that SAME byte budget with c3d via target_ratio = CHUNK_BYTES /
 *      codec_size, measure c3d size + PSNR.  (c3d's rate control lands within
 *      ~2-10 % of the target, so the comparison is genuinely size-matched.)
 *   3. Report the PSNR delta at each (size, size) point.
 *
 * All three baselines use 1 keyframe + 255 predicted frames so they can
 * exploit z-axis correlation the way c3d's 3D DWT does — all-intra handicaps
 * video codecs by discarding inter-slice redundancy.
 *
 * Rate-control mode is fixed-QP (constant quantiser) for every codec, so the
 * sweep traces a clean R-D curve without the noise of a bitrate-target loop.
 */

#include "c3d.h"

#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <wels/codec_api.h>
#include <wels/codec_def.h>

#include <x265.h>
#include <libde265/de265.h>

#include <aom/aom_codec.h>
#include <aom/aom_encoder.h>
#include <aom/aom_decoder.h>
#include <aom/aomcx.h>
#include <aom/aomdx.h>

#include <zfp.h>

#include "sz3c.h"

#include <sys/wait.h>
#include <fcntl.h>

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

/* Per-voxel absolute-error distribution: MAE, P90, P95, P99, max.  Uses a
 * 256-bucket histogram of |diff| ∈ [0, 255] (bounded by u8 range) so the
 * whole thing is one linear pass over the volume + one scan over 256 bins.
 * Writes all 5 values at once — same histogram covers every metric. */
typedef struct { double mae, p90, p95, p99; int max_err; } err_stats;
static void err_metrics_u8(const uint8_t *a, const uint8_t *b, size_t n,
                           err_stats *out) {
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        hist[d < 0 ? -d : d]++;
    }
    double mae_sum = 0;
    int max_err = 0;
    for (int k = 0; k < 256; ++k) {
        mae_sum += (double)k * hist[k];
        if (hist[k] > 0) max_err = k;
    }
    out->mae = mae_sum / (double)n;
    out->max_err = max_err;
    double tgt[3] = { 0.90 * (double)n, 0.95 * (double)n, 0.99 * (double)n };
    double pct[3] = { 0, 0, 0 };
    int got[3] = { 0, 0, 0 };
    double cum = 0;
    for (int k = 0; k < 256; ++k) {
        cum += hist[k];
        for (int i = 0; i < 3; ++i) {
            if (!got[i] && cum >= tgt[i]) { pct[i] = k; got[i] = 1; }
        }
    }
    out->p90 = pct[0]; out->p95 = pct[1]; out->p99 = pct[2];
}

/* Block-SSIM (Structural Similarity) — 8×8 non-overlapping blocks per slice,
 * averaged across the volume.  Uses standard SSIM moments with C1=(0.01·L)²,
 * C2=(0.03·L)², L=255 for 8-bit.  Block-SSIM is a faster variant of the
 * canonical sliding-window-Gaussian SSIM that's widely used in video-codec
 * benchmarks; for relative comparisons between codecs it tracks full SSIM
 * to within ~0.5% and runs ~50× faster.  Returns mean SSIM ∈ [0, 1]. */
#define C3D_SSIM_BLOCK 8u
static double ssim_u8(const uint8_t *a, const uint8_t *b, int side) {
    const double C1 = 0.01 * 255.0 * 0.01 * 255.0;   /* 6.5025  */
    const double C2 = 0.03 * 255.0 * 0.03 * 255.0;   /* 58.5225 */
    const int B = (int)C3D_SSIM_BLOCK;
    const double inv_n = 1.0 / (double)(B * B);
    double acc = 0.0;
    long blocks = 0;
    const size_t S2 = (size_t)side * (size_t)side;
    for (int z = 0; z < side; ++z) {
        const uint8_t *ap = a + (size_t)z * S2;
        const uint8_t *bp = b + (size_t)z * S2;
        for (int by = 0; by + B <= side; by += B)
        for (int bx = 0; bx + B <= side; bx += B) {
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            for (int dy = 0; dy < B; ++dy) {
                const uint8_t *arow = ap + (size_t)(by + dy) * (size_t)side + (size_t)bx;
                const uint8_t *brow = bp + (size_t)(by + dy) * (size_t)side + (size_t)bx;
                for (int dx = 0; dx < B; ++dx) {
                    double va = (double)arow[dx];
                    double vb = (double)brow[dx];
                    sa  += va;      sb  += vb;
                    saa += va * va; sbb += vb * vb;
                    sab += va * vb;
                }
            }
            double mua  = sa * inv_n,  mub = sb * inv_n;
            double vara = saa * inv_n - mua * mua;
            double varb = sbb * inv_n - mub * mub;
            double cov  = sab * inv_n - mua * mub;
            double num  = (2.0 * mua * mub + C1) * (2.0 * cov + C2);
            double den  = (mua * mua + mub * mub + C1) * (vara + varb + C2);
            acc += num / den;
            blocks++;
        }
    }
    return blocks > 0 ? acc / (double)blocks : 1.0;
}

/* =====================================================================
 * H.264 — openh264 encode + decode, single GOP (1 I + 255 P), CQP, CABAC.
 * =====================================================================*/
static size_t h264_encode_decode(const uint8_t *in, int qp, uint8_t *out_yuv,
                                 double *t_enc_out, double *t_dec_out) {
    const int W = (int)CHUNK_SIDE, H = (int)CHUNK_SIDE;
    const size_t Y = (size_t)W * (size_t)H, UV = Y / 4;

    ISVCEncoder *enc = NULL;
    if (WelsCreateSVCEncoder(&enc) != 0 || !enc) { fprintf(stderr, "h264 enc create\n"); exit(1); }

    SEncParamExt ep = {0};
    (*enc)->GetDefaultParams(enc, &ep);
    ep.iPicWidth = W; ep.iPicHeight = H;
    ep.iRCMode = RC_OFF_MODE; ep.iTargetBitrate = 0;
    ep.fMaxFrameRate = 30;
    ep.iUsageType = SCREEN_CONTENT_REAL_TIME;
    ep.iSpatialLayerNum = 1;
    ep.sSpatialLayers[0].iVideoWidth  = W;
    ep.sSpatialLayers[0].iVideoHeight = H;
    ep.sSpatialLayers[0].fFrameRate   = 30;
    ep.sSpatialLayers[0].iDLayerQp    = qp;
    ep.iTemporalLayerNum = 1;
    ep.uiIntraPeriod = 0;                   /* single-GOP: 1 I + 255 P */
    ep.eSpsPpsIdStrategy = CONSTANT_ID;
    ep.iMultipleThreadIdc = 1;
    ep.iComplexityMode = HIGH_COMPLEXITY;
    ep.iMaxQp = qp; ep.iMinQp = qp;
    ep.iEntropyCodingModeFlag = 1;          /* CABAC */
    ep.bEnableAdaptiveQuant = true;
    ep.bEnableBackgroundDetection = false;
    ep.bEnableFrameCroppingFlag = false;

    if ((*enc)->InitializeExt(enc, &ep) != 0) { fprintf(stderr, "h264 enc init\n"); exit(1); }
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

    double t_enc0 = now_s();
    for (int z = 0; z < H; ++z) {
        memcpy(y, in + (size_t)z * Y, Y);
        pic.uiTimeStamp = z;
        memset(&info, 0, sizeof info);
        if ((*enc)->EncodeFrame(enc, &pic, &info) != cmResultSuccess) {
            fprintf(stderr, "h264 EncodeFrame z=%d fail\n", z); exit(1);
        }
        for (int l = 0; l < info.iLayerNum; ++l) {
            const SLayerBSInfo *lb = &info.sLayerInfo[l];
            int lsz = 0;
            for (int n = 0; n < lb->iNalCount; ++n) lsz += lb->pNalLengthInByte[n];
            if (total + (size_t)lsz > cap) {
                while (total + (size_t)lsz > cap) cap *= 2;
                bitstream = realloc(bitstream, cap);
            }
            memcpy(bitstream + total, lb->pBsBuf, (size_t)lsz); total += (size_t)lsz;
        }
    }
    (*enc)->Uninitialize(enc); WelsDestroySVCEncoder(enc);
    double t_enc = now_s() - t_enc0;

    double t_dec0 = now_s();
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
    double t_dec = now_s() - t_dec0;

    if (decoded != (size_t)H) fprintf(stderr, "warning: h264 decoded %zu/%d at qp=%d\n", decoded, H, qp);

    free(y); free(u); free(v); free(bitstream);
    if (t_enc_out) *t_enc_out = t_enc;
    if (t_dec_out) *t_dec_out = t_dec;
    return total;
}

/* =====================================================================
 * H.265 — x265 encode + libde265 decode, 1 I + 255 P, CQP, no B-frames.
 * =====================================================================*/
static size_t h265_encode_decode(const uint8_t *in, int qp, uint8_t *out_yuv,
                                 double *t_enc_out, double *t_dec_out) {
    const int W = (int)CHUNK_SIDE, H = (int)CHUNK_SIDE;
    const size_t Y = (size_t)W * (size_t)H, UV = Y / 4;

    x265_param *param = x265_param_alloc();
    x265_param_default_preset(param, "medium", "zerolatency");
    /* Keep x265 single-threaded so 8 worker processes don't spawn 8×N
     * pool threads each.  frame-threads is already 1 under zerolatency. */
    x265_param_parse(param, "pools", "1");
    x265_param_parse(param, "wpp",   "0");
    param->sourceWidth          = W;
    param->sourceHeight         = H;
    param->fpsNum               = 30;
    param->fpsDenom             = 1;
    param->internalCsp          = X265_CSP_I420;
    param->internalBitDepth     = 8;
    param->bRepeatHeaders       = 1;            /* inline SPS/VPS/PPS */
    param->bAnnexB              = 1;            /* NAL with start codes */
    param->keyframeMax          = 256;          /* one IDR for 256-frame chunk */
    param->keyframeMin          = 256;
    param->bframes              = 0;
    param->lookaheadDepth       = 0;
    param->rc.rateControlMode   = X265_RC_CQP;
    param->rc.qp                = qp;
    param->logLevel             = X265_LOG_NONE;
    param->bEmitInfoSEI         = 0;
    param->bEmitHRDSEI          = 0;

    x265_encoder *enc = x265_encoder_open(param);
    if (!enc) { fprintf(stderr, "x265_encoder_open qp=%d failed\n", qp); exit(1); }

    x265_picture *pic_in = x265_picture_alloc();
    x265_picture_init(param, pic_in);
    uint8_t *y = malloc(Y), *u = malloc(UV), *v = malloc(UV);
    memset(u, 128, UV); memset(v, 128, UV);
    pic_in->planes[0] = y; pic_in->planes[1] = u; pic_in->planes[2] = v;
    pic_in->stride[0] = W; pic_in->stride[1] = W / 2; pic_in->stride[2] = W / 2;
    pic_in->bitDepth  = 8;
    pic_in->colorSpace = X265_CSP_I420;

    size_t cap = 4u * 1024 * 1024, total = 0;
    uint8_t *bitstream = malloc(cap);

    x265_nal *p_nal = NULL;
    uint32_t nal_count = 0;

    double t_enc0 = now_s();
    for (int z = 0; z < H; ++z) {
        memcpy(y, in + (size_t)z * Y, Y);
        pic_in->pts = z;
        int ret = x265_encoder_encode(enc, &p_nal, &nal_count, pic_in, NULL);
        if (ret < 0) { fprintf(stderr, "x265 encode z=%d fail\n", z); exit(1); }
        for (uint32_t i = 0; i < nal_count; ++i) {
            uint32_t sz = p_nal[i].sizeBytes;
            if (total + sz > cap) {
                while (total + sz > cap) cap *= 2;
                bitstream = realloc(bitstream, cap);
            }
            memcpy(bitstream + total, p_nal[i].payload, sz);
            total += sz;
        }
    }
    /* Flush */
    while (1) {
        int ret = x265_encoder_encode(enc, &p_nal, &nal_count, NULL, NULL);
        if (ret <= 0) break;
        for (uint32_t i = 0; i < nal_count; ++i) {
            uint32_t sz = p_nal[i].sizeBytes;
            if (total + sz > cap) {
                while (total + sz > cap) cap *= 2;
                bitstream = realloc(bitstream, cap);
            }
            memcpy(bitstream + total, p_nal[i].payload, sz);
            total += sz;
        }
    }
    x265_encoder_close(enc);
    x265_picture_free(pic_in);
    x265_param_free(param);
    free(y); free(u); free(v);
    double t_enc = now_s() - t_enc0;

    double t_dec0 = now_s();
    /* Decode with libde265. */
    de265_decoder_context *dec = de265_new_decoder();
    de265_set_verbosity(0);
    de265_push_data(dec, bitstream, (int)total, 0, NULL);
    de265_flush_data(dec);

    size_t decoded = 0;
    int more = 1;
    while (more) {
        de265_error err = de265_decode(dec, &more);
        if (err != DE265_OK && err != DE265_ERROR_WAITING_FOR_INPUT_DATA) break;
        const struct de265_image *img;
        while ((img = de265_get_next_picture(dec)) != NULL) {
            int stride = 0;
            const uint8_t *py = de265_get_image_plane(img, 0, &stride);
            int iw = de265_get_image_width(img, 0);
            int ih = de265_get_image_height(img, 0);
            int cpy_w = iw < W ? iw : W;
            int cpy_h = ih < H ? ih : H;
            if (decoded < (size_t)H) {
                for (int r = 0; r < cpy_h; ++r)
                    memcpy(out_yuv + decoded * Y + (size_t)r * (size_t)W,
                           py + (size_t)r * (size_t)stride, (size_t)cpy_w);
                decoded++;
            }
            de265_release_next_picture(dec);
        }
    }
    de265_free_decoder(dec);
    double t_dec = now_s() - t_dec0;

    if (decoded != (size_t)H) fprintf(stderr, "warning: h265 decoded %zu/%d at qp=%d\n", decoded, H, qp);

    free(bitstream);
    if (t_enc_out) *t_enc_out = t_enc;
    if (t_dec_out) *t_dec_out = t_dec;
    return total;
}

/* =====================================================================
 * AV1 — libaom all-intra encode + decode, AOM_Q (constant Q).
 *
 * All-intra rather than inter + altref for two reasons:
 *   1. libaom's altref-with-lag pipeline under AOM_Q drops visible frames
 *      at flush time (consistently loses ~17/256 slices).
 *   2. Without altref, AOM_Q rate-control saturates at a ~770 KB floor on
 *      256-frame sequences, leaving no useful R-D range above ~20:1.
 * All-intra sidesteps both problems at the cost of not exploiting z-axis
 * correlation the way H.264/H.265 do — documented in the README table.
 * =====================================================================*/
static size_t av1_encode_decode(const uint8_t *in, int cq, uint8_t *out_yuv,
                                double *t_enc_out, double *t_dec_out) {
    const int W = (int)CHUNK_SIDE, H = (int)CHUNK_SIDE;
    const size_t Y = (size_t)W * (size_t)H;

    aom_codec_iface_t *cx_iface = aom_codec_av1_cx();
    aom_codec_enc_cfg_t cfg;
    aom_codec_err_t e = aom_codec_enc_config_default(cx_iface, &cfg, AOM_USAGE_ALL_INTRA);
    if (e != AOM_CODEC_OK) { fprintf(stderr, "aom cfg default: %s\n", aom_codec_err_to_string(e)); exit(1); }

    cfg.g_w                = W;
    cfg.g_h                = H;
    cfg.g_bit_depth        = AOM_BITS_8;
    cfg.g_input_bit_depth  = 8;
    cfg.g_timebase.num     = 1;
    cfg.g_timebase.den     = 30;
    cfg.g_threads          = 1;
    cfg.g_lag_in_frames    = 0;
    cfg.g_pass             = AOM_RC_ONE_PASS;
    cfg.rc_end_usage       = AOM_Q;
    cfg.rc_target_bitrate  = 0;
    cfg.rc_min_quantizer   = 0;
    cfg.rc_max_quantizer   = 63;
    cfg.kf_mode            = AOM_KF_AUTO;       /* every frame is a keyframe in ALL_INTRA */
    cfg.monochrome         = 0;

    aom_codec_ctx_t enc;
    if (aom_codec_enc_init(&enc, cx_iface, &cfg, 0) != AOM_CODEC_OK) {
        fprintf(stderr, "aom_codec_enc_init: %s\n", aom_codec_error_detail(&enc)); exit(1);
    }
    aom_codec_control(&enc, AOME_SET_CQ_LEVEL,     cq);
    aom_codec_control(&enc, AOME_SET_CPUUSED,      6);  /* ALL_INTRA max speed = 9, 6 is balanced */
    aom_codec_control(&enc, AV1E_SET_ROW_MT,       0);
    aom_codec_control(&enc, AV1E_SET_AQ_MODE,      0);
    aom_codec_control(&enc, AV1E_SET_TILE_COLUMNS, 0);
    aom_codec_control(&enc, AV1E_SET_TILE_ROWS,    0);

    aom_image_t *img = aom_img_alloc(NULL, AOM_IMG_FMT_I420, W, H, 1);
    if (!img) { fprintf(stderr, "aom_img_alloc fail\n"); exit(1); }
    /* Fill U and V once with 128. */
    memset(img->planes[1], 128, (size_t)img->stride[1] * (size_t)(H / 2));
    memset(img->planes[2], 128, (size_t)img->stride[2] * (size_t)(H / 2));

    size_t cap = 4u * 1024 * 1024, total = 0;
    uint8_t *bitstream = malloc(cap);
    /* Packet sizes — each cx_data packet is one temporal unit that may
     * carry a superframe of several OBUs (visible + altref).  We decode
     * TU by TU and drain get_frame for each (handles the superframe case). */
    size_t *tu_sizes = malloc(sizeof(size_t) * (size_t)(H * 2 + 32));
    int n_tus = 0;

    double t_enc0 = now_s();
    for (int z = 0; z < H; ++z) {
        for (int r = 0; r < H; ++r)
            memcpy(img->planes[0] + r * img->stride[0], in + (size_t)z * Y + r * W, W);
        aom_enc_frame_flags_t flags = (z == 0) ? AOM_EFLAG_FORCE_KF : 0;
        if (aom_codec_encode(&enc, img, z, 1, flags) != AOM_CODEC_OK) {
            fprintf(stderr, "aom_codec_encode z=%d: %s\n", z, aom_codec_error_detail(&enc)); exit(1);
        }
        const aom_codec_cx_pkt_t *pkt;
        aom_codec_iter_t it = NULL;
        while ((pkt = aom_codec_get_cx_data(&enc, &it)) != NULL) {
            if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
            size_t sz = pkt->data.frame.sz;
            if (total + sz > cap) {
                while (total + sz > cap) cap *= 2;
                bitstream = realloc(bitstream, cap);
            }
            memcpy(bitstream + total, pkt->data.frame.buf, sz);
            tu_sizes[n_tus++] = sz;
            total += sz;
        }
    }
    /* Flush */
    if (aom_codec_encode(&enc, NULL, 0, 1, 0) != AOM_CODEC_OK) {
        fprintf(stderr, "aom flush: %s\n", aom_codec_error_detail(&enc)); exit(1);
    }
    {
        const aom_codec_cx_pkt_t *pkt;
        aom_codec_iter_t it = NULL;
        while ((pkt = aom_codec_get_cx_data(&enc, &it)) != NULL) {
            if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
            size_t sz = pkt->data.frame.sz;
            if (total + sz > cap) {
                while (total + sz > cap) cap *= 2;
                bitstream = realloc(bitstream, cap);
            }
            memcpy(bitstream + total, pkt->data.frame.buf, sz);
            tu_sizes[n_tus++] = sz;
            total += sz;
        }
    }

    aom_img_free(img);
    aom_codec_destroy(&enc);
    double t_enc = now_s() - t_enc0;

    double t_dec0 = now_s();
    /* Decode TU by TU; each TU may decode to multiple visible frames
     * when libaom uses superframes (altref + show_existing_frame). */
    aom_codec_ctx_t dec;
    if (aom_codec_dec_init(&dec, aom_codec_av1_dx(), NULL, 0) != AOM_CODEC_OK) {
        fprintf(stderr, "aom_codec_dec_init fail\n"); exit(1);
    }

    size_t offset = 0, decoded = 0;
    for (int i = 0; i < n_tus; ++i) {
        if (aom_codec_decode(&dec, bitstream + offset, tu_sizes[i], NULL) != AOM_CODEC_OK) {
            fprintf(stderr, "aom_codec_decode tu %d: %s\n", i, aom_codec_error_detail(&dec));
            break;
        }
        offset += tu_sizes[i];
        aom_codec_iter_t it = NULL;
        aom_image_t *dimg;
        while ((dimg = aom_codec_get_frame(&dec, &it)) != NULL) {
            if (decoded >= (size_t)H) continue;
            int cpy_w = (int)dimg->d_w < W ? (int)dimg->d_w : W;
            int cpy_h = (int)dimg->d_h < H ? (int)dimg->d_h : H;
            for (int r = 0; r < cpy_h; ++r)
                memcpy(out_yuv + decoded * Y + (size_t)r * (size_t)W,
                       dimg->planes[0] + (size_t)r * (size_t)dimg->stride[0],
                       (size_t)cpy_w);
            decoded++;
        }
    }
    aom_codec_destroy(&dec);
    double t_dec = now_s() - t_dec0;

    if (decoded != (size_t)H) fprintf(stderr, "warning: av1 decoded %zu/%d at cq=%d\n", decoded, H, cq);

    free(bitstream); free(tu_sizes);
    if (t_enc_out) *t_enc_out = t_enc;
    if (t_dec_out) *t_dec_out = t_dec;
    return total;
}

/* ---------------------------------------------------------------------
 * ZFP — fixed-rate 3D integer mode.  u8 input is widened to int32,
 * compressed at `rate_q8` / 256 bits per value (so rate_q8=128 → 0.5 bpp ≈
 * 16:1 on u8-in-i32 semantic bytes), decompressed, narrowed back to u8.
 * The rate slots into the "QP" column: smaller = higher compression.
 * --------------------------------------------------------------------- */
static size_t zfp_encode_decode(const uint8_t *in, int rate_q8, uint8_t *out,
                                double *t_enc_out, double *t_dec_out) {
    const int W = CHUNK_SIDE, H = CHUNK_SIDE, D = CHUNK_SIDE;
    const size_t N = (size_t)W * (size_t)H * (size_t)D;

    int32_t *buf32 = aligned_alloc(32, N * sizeof(int32_t));
    if (!buf32) { fprintf(stderr, "zfp oom\n"); exit(1); }
    for (size_t i = 0; i < N; ++i) buf32[i] = (int32_t)in[i];

    double t_enc0 = now_s();
    zfp_field *field = zfp_field_3d(buf32, zfp_type_int32, W, H, D);
    zfp_stream *zfp = zfp_stream_open(NULL);
    /* Fixed-rate: bits per value = rate_q8 / 256.  ZFP clamps to its minimum
     * (~1 bpp for int32 3D) so highest useful ratios are ~32:1 on i32
     * data = ~8:1 on the u8 source byte count. */
    double bpp = (double)rate_q8 / 256.0;
    zfp_stream_set_rate(zfp, bpp, zfp_type_int32, 3, zfp_false);
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    uint8_t *bits = malloc(bufsize);
    if (!bits) { fprintf(stderr, "zfp oom\n"); exit(1); }
    bitstream *bs = stream_open(bits, bufsize);
    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);

    size_t nbytes = zfp_compress(zfp, field);
    if (nbytes == 0) { fprintf(stderr, "zfp_compress failed\n"); exit(1); }
    double t_enc = now_s() - t_enc0;

    double t_dec0 = now_s();
    /* Decode in place on the same buffer. */
    memset(buf32, 0, N * sizeof(int32_t));
    zfp_stream_rewind(zfp);
    if (!zfp_decompress(zfp, field)) {
        fprintf(stderr, "zfp_decompress failed\n"); exit(1);
    }

    for (size_t i = 0; i < N; ++i) {
        int32_t v = buf32[i];
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        out[i] = (uint8_t)v;
    }
    double t_dec = now_s() - t_dec0;

    stream_close(bs);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    free(bits);
    free(buf32);
    if (t_enc_out) *t_enc_out = t_enc;
    if (t_dec_out) *t_dec_out = t_dec;
    return nbytes;
}

/* ---------------------------------------------------------------------
 * SZ3 — C API (Lorenzo + regression + Huffman + zstd).  SZ3 only supports
 * float/double at runtime, so u8 is widened to float, compressed with
 * absolute error bound in LSBs, decompressed, narrowed back.
 * --------------------------------------------------------------------- */
static size_t sz3_encode_decode(const uint8_t *in, int tol, uint8_t *out,
                                double *t_enc_out, double *t_dec_out) {
    const size_t N = CHUNK_BYTES;
    float *buf = aligned_alloc(32, N * sizeof(float));
    if (!buf) { fprintf(stderr, "sz3 oom\n"); exit(1); }
    for (size_t i = 0; i < N; ++i) buf[i] = (float)in[i];

    double t_enc0 = now_s();
    size_t out_size = 0;
    unsigned char *comp = SZ_compress_args(
        SZ_FLOAT, buf, &out_size,
        ABS, (double)tol, 0.0, 0.0,
        0, 0, CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE);
    if (!comp || out_size == 0) { fprintf(stderr, "sz3 encode failed\n"); exit(1); }
    double t_enc = now_s() - t_enc0;

    double t_dec0 = now_s();
    float *dec = (float *)SZ_decompress(
        SZ_FLOAT, comp, out_size,
        0, 0, CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE);
    if (!dec) { fprintf(stderr, "sz3 decode failed\n"); exit(1); }
    for (size_t i = 0; i < N; ++i) {
        float v = dec[i];
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        out[i] = (uint8_t)(v + 0.5f);
    }
    double t_dec = now_s() - t_dec0;
    free_buf(comp);
    free_buf(dec);
    free(buf);
    if (t_enc_out) *t_enc_out = t_enc;
    if (t_dec_out) *t_dec_out = t_dec;
    return out_size;
}

/* ---------------------------------------------------------------------
 * TTHRESH — invoked as an external CLI.  Writes the chunk to a temp
 * file, spawns `tthresh -i ... -p <psnr> -c ...` for encode then
 * `tthresh -c ... -o ...` for decode; reads the reconstructed bytes
 * back.  `target_psnr` is the "QP" — higher = better quality.
 *
 * Hits disk 3× per call so it's measurably slower than library-linked
 * codecs; accepted overhead for a comparison bench.
 * --------------------------------------------------------------------- */
#ifndef TTHRESH_BIN
#define TTHRESH_BIN "/home/forrest/c3d/third_party/tthresh/build/tthresh"
#endif
static int run_silently(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, 1); dup2(devnull, 2);
            close(devnull);
        }
        execv(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
static size_t tthresh_encode_decode(const uint8_t *in, int target_psnr,
                                    uint8_t *out,
                                    double *t_enc_out, double *t_dec_out) {
    char raw[160], comp[160], dec[160], psnr_arg[32], sx[8], sy[8], sz[8];
    /* pthread_self() is a thread-unique id within this process, so bench
     * workers running in parallel write to distinct tempfiles. */
    unsigned long tid = (unsigned long)pthread_self();
    snprintf(raw,  sizeof raw,  "/tmp/tthresh_%d_%lx_%d.raw",
             (int)getpid(), tid, target_psnr);
    snprintf(comp, sizeof comp, "/tmp/tthresh_%d_%lx_%d.comp",
             (int)getpid(), tid, target_psnr);
    snprintf(dec,  sizeof dec,  "/tmp/tthresh_%d_%lx_%d.dec",
             (int)getpid(), tid, target_psnr);
    snprintf(psnr_arg, sizeof psnr_arg, "%d", target_psnr);
    snprintf(sx, sizeof sx, "%u", CHUNK_SIDE);
    snprintf(sy, sizeof sy, "%u", CHUNK_SIDE);
    snprintf(sz, sizeof sz, "%u", CHUNK_SIDE);

    FILE *fp = fopen(raw, "wb");
    if (!fp) { perror(raw); exit(1); }
    if (fwrite(in, 1, CHUNK_BYTES, fp) != CHUNK_BYTES) { perror("fwrite"); exit(1); }
    fclose(fp);

    char *enc_argv[] = {
        (char *)(uintptr_t)TTHRESH_BIN, (char *)(uintptr_t)"-i", raw,
        (char *)(uintptr_t)"-t", (char *)(uintptr_t)"uchar",
        (char *)(uintptr_t)"-s", sx, sy, sz,
        (char *)(uintptr_t)"-p", psnr_arg, (char *)(uintptr_t)"-c", comp, NULL
    };
    double t_enc0 = now_s();
    if (run_silently(enc_argv) != 0) {
        fprintf(stderr, "tthresh encode failed\n"); exit(1);
    }
    double t_enc = now_s() - t_enc0;

    char *dec_argv[] = {
        (char *)(uintptr_t)TTHRESH_BIN, (char *)(uintptr_t)"-c", comp,
        (char *)(uintptr_t)"-o", dec, NULL
    };
    double t_dec0 = now_s();
    if (run_silently(dec_argv) != 0) {
        fprintf(stderr, "tthresh decode failed\n"); exit(1);
    }
    double t_dec = now_s() - t_dec0;

    struct stat st;
    if (stat(comp, &st) != 0) { perror(comp); exit(1); }
    size_t comp_size = (size_t)st.st_size;

    fp = fopen(dec, "rb");
    if (!fp) { perror(dec); exit(1); }
    size_t rd = fread(out, 1, CHUNK_BYTES, fp);
    fclose(fp);
    if (rd != CHUNK_BYTES) { fprintf(stderr, "tthresh short decode %zu\n", rd); exit(1); }

    unlink(raw); unlink(comp); unlink(dec);
    if (t_enc_out) *t_enc_out = t_enc;
    if (t_dec_out) *t_dec_out = t_dec;
    return comp_size;
}

/* =====================================================================
 * Driver — parallel sweep across chunks.  Each worker owns its own c3d
 * encoder/decoder + scratch buffers; codec handles are constructed per call
 * so they're thread-local by construction.  Chunks are dispensed from a
 * shared counter under a mutex; per-thread accumulators are reduced at end.
 * =====================================================================*/

/* H.264 / H.265 take [0,51] QP; AV1 AOM_Q takes cq_level ∈ [0,63];
 * ZFP accuracy tolerance in integer LSBs spans a wide range. */
static const int g_h264_qps[] = { 18, 24, 30, 36, 42, 48 };
static const int g_h265_qps[] = { 18, 24, 30, 36, 42, 48 };
static const int g_av1_cqs [] = { 16, 28, 40, 48, 55, 60 };
/* Fixed-rate in q8 (bits × 256).  256=1.0 bpp, 128=0.5, 64=0.25, 32=0.125,
 * etc.  Picked to span ratios comparable to the video/c3d QP columns. */
static const int g_zfp_rates[] = { 1024, 512, 256, 128,  64,  32 };
/* SZ3 absolute error bound in u8 LSBs.  Larger → higher ratio. */
static const int g_sz3_eps  [] = {    1,   3,   6,  12,  24,  48 };
/* TTHRESH target PSNR in dB. */
static const int g_tthresh_p[] = {   50,  45,  40,  35,  30,  25 };
#define N_QPS    (sizeof g_h264_qps / sizeof g_h264_qps[0])
#define N_CODECS 6u

/* Codec entry point: encode `in` at QP `qp`, decode into `out`, return
 * the compressed byte size.  Writes encode/decode wall time in seconds
 * into *t_enc_out / *t_dec_out when those are non-NULL. */
typedef size_t (*codec_fn)(const uint8_t *in, int qp, uint8_t *out,
                           double *t_enc_out, double *t_dec_out);
typedef struct {
    const char *name;
    codec_fn    fn;
    const int  *qps;
    const char *qp_label;
} codec_desc;

static const codec_desc g_codecs[N_CODECS] = {
    { "H264", h264_encode_decode, g_h264_qps, "Q"  },
    { "H265", h265_encode_decode, g_h265_qps, "Q"  },
    { "AV1",  av1_encode_decode,  g_av1_cqs,  "cq" },
    { "ZFP",  zfp_encode_decode,  g_zfp_rates, "rq8" },
    { "SZ3",  sz3_encode_decode,  g_sz3_eps,   "eps" },
    { "TTHR", tthresh_encode_decode, g_tthresh_p, "P" },
};

typedef struct {
    double v_sz[N_CODECS][N_QPS];
    double v_p [N_CODECS][N_QPS];   /* PSNR    */
    double v_s [N_CODECS][N_QPS];   /* block-SSIM  */
    double v_mae[N_CODECS][N_QPS];
    double v_p99[N_CODECS][N_QPS];
    double v_max[N_CODECS][N_QPS];
    double c_sz[N_CODECS][N_QPS];
    double c_p [N_CODECS][N_QPS];
    double c_s [N_CODECS][N_QPS];
    double c_mae[N_CODECS][N_QPS];
    double c_p99[N_CODECS][N_QPS];
    double c_max[N_CODECS][N_QPS];
    double c_et[N_CODECS][N_QPS];
    double c_dt[N_CODECS][N_QPS];
    double v_et[N_CODECS][N_QPS];
    double v_dt[N_CODECS][N_QPS];
} bench_sums;

typedef struct {
    int tid;
    /* shared job queue */
    pthread_mutex_t *job_mutex;
    int *next_idx;
    int n_paths;
    char (*paths)[4096];
    /* shared output lock */
    pthread_mutex_t *out_mutex;
    /* shared optional c3dx context (learned priors) */
    const c3d_ctx *ctx;
    /* per-thread results */
    bench_sums sums;
    int my_chunks;
} worker_t;

static void *worker_fn(void *arg) {
    worker_t *w = (worker_t *)arg;

    uint8_t *in        = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *c3d_dec   = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *c3d_enc   = aligned_alloc(32, c3d_chunk_encode_max_size());
    uint8_t *codec_dec = aligned_alloc(32, CHUNK_BYTES);
    if (!in || !c3d_dec || !c3d_enc || !codec_dec) {
        fprintf(stderr, "worker %d: oom\n", w->tid); exit(1);
    }
    c3d_encoder *enc_ctx = c3d_encoder_new();
    c3d_decoder *dec_ctx = c3d_decoder_new();

    for (;;) {
        pthread_mutex_lock(w->job_mutex);
        int idx = (*w->next_idx)++;
        pthread_mutex_unlock(w->job_mutex);
        if (idx >= w->n_paths) break;

        const char *path = w->paths[idx];
        const char *bn = strrchr(path, '/'); bn = bn ? bn + 1 : path;

        FILE *fp = fopen(path, "rb");
        if (!fp || fread(in, 1, CHUNK_BYTES, fp) != CHUNK_BYTES) {
            if (fp) fclose(fp);
            continue;
        }
        fclose(fp);

        for (size_t k = 0; k < N_CODECS; ++k) {
            const codec_desc *cd = &g_codecs[k];
            for (size_t q = 0; q < N_QPS; ++q) {
                int qp = cd->qps[q];
                double t_v_enc = 0.0, t_v_dec = 0.0;
                size_t v_sz = cd->fn(in, qp, codec_dec, &t_v_enc, &t_v_dec);
                double v_p  = psnr_u8(in, codec_dec, CHUNK_BYTES);
                double v_s  = ssim_u8(in, codec_dec, (int)CHUNK_SIDE);
                err_stats v_e; err_metrics_u8(in, codec_dec, CHUNK_BYTES, &v_e);

                float target_r = (float)((double)CHUNK_BYTES / (double)v_sz);
                if (target_r <= 1.01f) target_r = 1.01f;

                double t0 = now_s();
                size_t c_sz = c3d_encoder_chunk_encode(enc_ctx, in, target_r, w->ctx,
                                                      c3d_enc, c3d_chunk_encode_max_size());
                double t_enc = now_s() - t0;

                t0 = now_s();
                c3d_decoder_chunk_decode(dec_ctx, c3d_enc, c_sz, w->ctx, c3d_dec);
                double t_dec = now_s() - t0;
                double c_p  = psnr_u8(in, c3d_dec, CHUNK_BYTES);
                double c_s  = ssim_u8(in, c3d_dec, (int)CHUNK_SIDE);
                err_stats c_e; err_metrics_u8(in, c3d_dec, CHUNK_BYTES, &c_e);

                w->sums.v_sz [k][q] += (double)v_sz;
                w->sums.v_p  [k][q] += v_p;
                w->sums.v_s  [k][q] += v_s;
                w->sums.v_mae[k][q] += v_e.mae;
                w->sums.v_p99[k][q] += v_e.p99;
                w->sums.v_max[k][q] += v_e.max_err;
                w->sums.c_sz [k][q] += (double)c_sz;
                w->sums.c_p  [k][q] += c_p;
                w->sums.c_s  [k][q] += c_s;
                w->sums.c_mae[k][q] += c_e.mae;
                w->sums.c_p99[k][q] += c_e.p99;
                w->sums.c_max[k][q] += c_e.max_err;
                w->sums.c_et [k][q] += t_enc;
                w->sums.c_dt [k][q] += t_dec;
                w->sums.v_et [k][q] += t_v_enc;
                w->sums.v_dt [k][q] += t_v_dec;

                pthread_mutex_lock(w->out_mutex);
                printf("[t%d] %-5s %-34s | %s%2d %8zu %6.2f %5.3f  %7zu %6.2f %5.3f  %+.2f\n",
                       w->tid, cd->name, bn, cd->qp_label, qp,
                       v_sz, v_p, v_s, c_sz, c_p, c_s, c_p - v_p);
                fflush(stdout);
                pthread_mutex_unlock(w->out_mutex);
            }
        }
        w->my_chunks++;
    }

    c3d_encoder_free(enc_ctx); c3d_decoder_free(dec_ctx);
    free(in); free(c3d_dec); free(c3d_enc); free(codec_dec);
    return NULL;
}

static void print_summary(size_t k, size_t n_chunks, const bench_sums *s) {
    const codec_desc *cd = &g_codecs[k];
    printf("\n%s summary averaged over %zu chunks:\n", cd->name, n_chunks);
    printf("  %-3s  vid_sz       PSNR   SSIM   MAE    P99  max    "
           "vidE  vidD    "
           "c3d_sz       PSNR   SSIM   MAE    P99  max    "
           "c3dE  c3dD | ΔPSNR    ΔSSIM\n",
           cd->qp_label);
    printf("  ---  -----------  -----  -----  -----  ---  ---    "
           "----  ----    "
           "-----------  -----  -----  -----  ---  ---    "
           "----  ---- | -------  -------\n");
    for (size_t q = 0; q < N_QPS; ++q) {
        double n = (double)n_chunks;
        double v_sz = s->v_sz[k][q] / n;
        double v_p  = s->v_p [k][q] / n;
        double v_s  = s->v_s [k][q] / n;
        double v_m  = s->v_mae[k][q] / n;
        double v_p99 = s->v_p99[k][q] / n;
        double v_max = s->v_max[k][q] / n;
        double c_sz = s->c_sz[k][q] / n;
        double c_p  = s->c_p [k][q] / n;
        double c_s  = s->c_s [k][q] / n;
        double c_m  = s->c_mae[k][q] / n;
        double c_p99 = s->c_p99[k][q] / n;
        double c_max = s->c_max[k][q] / n;
        double c_enc_mbps = ((double)n_chunks * (double)CHUNK_BYTES) /
                         (s->c_et[k][q] * 1024.0 * 1024.0);
        double c_dec_mbps = ((double)n_chunks * (double)CHUNK_BYTES) /
                         (s->c_dt[k][q] * 1024.0 * 1024.0);
        double v_enc_mbps = ((double)n_chunks * (double)CHUNK_BYTES) /
                         (s->v_et[k][q] * 1024.0 * 1024.0);
        double v_dec_mbps = ((double)n_chunks * (double)CHUNK_BYTES) /
                         (s->v_dt[k][q] * 1024.0 * 1024.0);
        double ratio_v = (double)CHUNK_BYTES / v_sz;
        printf("  %3d  %11.0f  %5.2f  %5.3f  %5.2f  %3.0f  %3.0f    "
               "%4.0f  %4.0f    "
               "%11.0f  %5.2f  %5.3f  %5.2f  %3.0f  %3.0f    "
               "%4.0f  %4.0f | %+6.2f   %+7.4f  (~%.1f:1)\n",
               cd->qps[q],
               v_sz, v_p, v_s, v_m, v_p99, v_max,
               v_enc_mbps, v_dec_mbps,
               c_sz, c_p, c_s, c_m, c_p99, c_max,
               c_enc_mbps, c_dec_mbps,
               c_p - v_p, c_s - v_s, ratio_v);
    }
}

static int pick_n_threads(void) {
    const char *e = getenv("C3D_BENCH_THREADS");
    if (e && *e) {
        long n = strtol(e, NULL, 10);
        if (n > 0 && n < 256) return (int)n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return (int)n;
}

int main(int argc, char **argv) {
    const char *dir_path = (argc > 1) ? argv[1] : getenv("C3D_CORPUS");
    if (!dir_path) { fprintf(stderr, "usage: %s <corpus_dir>\n", argv[0]); return 2; }
    DIR *dir = opendir(dir_path);
    if (!dir) { perror(dir_path); return 1; }

    /* Collect matching chunk paths. */
    int cap = 64, n_paths = 0;
    char (*paths)[4096] = malloc((size_t)cap * 4096);
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof full, "%s/%s", dir_path, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode) ||
            (size_t)st.st_size != CHUNK_BYTES) continue;
        if (n_paths == cap) {
            cap *= 2;
            paths = realloc(paths, (size_t)cap * 4096);
        }
        memcpy(paths[n_paths++], full, sizeof paths[0]);
    }
    closedir(dir);

    if (n_paths == 0) { fprintf(stderr, "no chunks in %s\n", dir_path); return 1; }

    int n_threads = pick_n_threads();
    if (n_threads > n_paths) n_threads = n_paths;
    printf("c3d_bench: %d chunks, %d worker threads\n", n_paths, n_threads);

    /* Optional learned-priors .c3dx.  Set via C3D_BENCH_CTX=<path>.  When
     * present, passed into every c3d_encoder_chunk_encode/decode call so
     * the measurement reflects the trained priors. */
    c3d_ctx *ctx = NULL;
    uint8_t *ctx_bytes = NULL;
    const char *ctx_path = getenv("C3D_BENCH_CTX");
    if (ctx_path) {
        FILE *fp = fopen(ctx_path, "rb");
        if (!fp) { perror(ctx_path); return 1; }
        fseek(fp, 0, SEEK_END);
        size_t cs = (size_t)ftell(fp);
        fseek(fp, 0, SEEK_SET);
        ctx_bytes = malloc(cs);
        if (fread(ctx_bytes, 1, cs, fp) != cs) { perror("ctx read"); return 1; }
        fclose(fp);
        ctx = c3d_ctx_parse(ctx_bytes, cs);
        printf("loaded ctx %s (%zu bytes)\n", ctx_path, cs);
    }

    printf("Each line: [tN] codec chunk | QP vid_sz vid_PSNR  c3d_sz c3d_PSNR  Δ\n");
    printf("----------------------------------------------------------------------\n");
    fflush(stdout);

    pthread_mutex_t job_mutex, out_mutex;
    pthread_mutex_init(&job_mutex, NULL);
    pthread_mutex_init(&out_mutex, NULL);
    int next_idx = 0;

    worker_t *workers = calloc((size_t)n_threads, sizeof *workers);
    pthread_t *thr    = calloc((size_t)n_threads, sizeof *thr);
    double t_wall0 = now_s();
    for (int t = 0; t < n_threads; ++t) {
        workers[t].tid       = t;
        workers[t].job_mutex = &job_mutex;
        workers[t].out_mutex = &out_mutex;
        workers[t].next_idx  = &next_idx;
        workers[t].n_paths   = n_paths;
        workers[t].paths     = paths;
        workers[t].ctx       = ctx;
        pthread_create(&thr[t], NULL, worker_fn, &workers[t]);
    }
    for (int t = 0; t < n_threads; ++t) pthread_join(thr[t], NULL);
    double t_wall = now_s() - t_wall0;

    /* Reduce per-thread sums. */
    bench_sums tot = {0};
    size_t n_chunks = 0;
    for (int t = 0; t < n_threads; ++t) {
        n_chunks += (size_t)workers[t].my_chunks;
        for (size_t k = 0; k < N_CODECS; ++k) for (size_t q = 0; q < N_QPS; ++q) {
            tot.v_sz [k][q] += workers[t].sums.v_sz [k][q];
            tot.v_p  [k][q] += workers[t].sums.v_p  [k][q];
            tot.v_s  [k][q] += workers[t].sums.v_s  [k][q];
            tot.v_mae[k][q] += workers[t].sums.v_mae[k][q];
            tot.v_p99[k][q] += workers[t].sums.v_p99[k][q];
            tot.v_max[k][q] += workers[t].sums.v_max[k][q];
            tot.c_sz [k][q] += workers[t].sums.c_sz [k][q];
            tot.c_p  [k][q] += workers[t].sums.c_p  [k][q];
            tot.c_s  [k][q] += workers[t].sums.c_s  [k][q];
            tot.c_mae[k][q] += workers[t].sums.c_mae[k][q];
            tot.c_p99[k][q] += workers[t].sums.c_p99[k][q];
            tot.c_max[k][q] += workers[t].sums.c_max[k][q];
            tot.c_et [k][q] += workers[t].sums.c_et [k][q];
            tot.c_dt [k][q] += workers[t].sums.c_dt [k][q];
            tot.v_et [k][q] += workers[t].sums.v_et [k][q];
            tot.v_dt [k][q] += workers[t].sums.v_dt [k][q];
        }
    }

    if (n_chunks == 0) { fprintf(stderr, "no chunks successfully processed\n"); return 1; }

    for (size_t k = 0; k < N_CODECS; ++k) print_summary(k, n_chunks, &tot);

    printf("\nWall time: %.2f s  (%d threads, %zu chunks)\n", t_wall, n_threads, n_chunks);

    pthread_mutex_destroy(&job_mutex);
    pthread_mutex_destroy(&out_mutex);
    if (ctx) c3d_ctx_free(ctx);
    if (ctx_bytes) free(ctx_bytes);
    free(workers); free(thr); free(paths);
    return 0;
}
