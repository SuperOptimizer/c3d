/* c3d_progressive — demonstrate §T9 byte-truncatable decode.
 *
 * Usage: c3d_progressive <encoded.c3dc> <original.u8> [truncation_fractions...]
 *
 * Encoded chunk is a raw c3d_chunk_encode output.
 * Original raw-u8 chunk is used for PSNR/SSIM computation.  If no fractions
 * given, sweeps {1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01}.
 *
 * For each truncation point, feeds only that many bytes to
 * c3d_decoder_chunk_decode and reports size / PSNR / block-SSIM vs the
 * original.  Shows the trade-off for streaming / bandwidth-adaptive
 * callers who want to stop early.
 *
 * See LICENSE. */

#include "c3d.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define CHUNK_BYTES (256u * 256u * 256u)
#define SIDE 256

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

static double ssim_u8(const uint8_t *a, const uint8_t *b, int side) {
    const double C1 = 6.5025, C2 = 58.5225;
    const int B = 8;
    const double inv_n = 1.0 / (double)(B * B);
    double acc = 0.0; long blocks = 0;
    const size_t S2 = (size_t)side * (size_t)side;
    for (int z = 0; z < side; ++z) {
        const uint8_t *ap = a + (size_t)z * S2;
        const uint8_t *bp = b + (size_t)z * S2;
        for (int by = 0; by + B <= side; by += B)
        for (int bx = 0; bx + B <= side; bx += B) {
            double sa=0,sb=0,saa=0,sbb=0,sab=0;
            for (int dy = 0; dy < B; ++dy) {
                const uint8_t *ar = ap + (size_t)(by+dy)*(size_t)side + (size_t)bx;
                const uint8_t *br = bp + (size_t)(by+dy)*(size_t)side + (size_t)bx;
                for (int dx = 0; dx < B; ++dx) {
                    double va = ar[dx], vb = br[dx];
                    sa += va; sb += vb;
                    saa += va*va; sbb += vb*vb; sab += va*vb;
                }
            }
            double mua = sa * inv_n, mub = sb * inv_n;
            double vara = saa * inv_n - mua*mua;
            double varb = sbb * inv_n - mub*mub;
            double cov  = sab * inv_n - mua*mub;
            double num  = (2*mua*mub + C1) * (2*cov + C2);
            double den  = (mua*mua + mub*mub + C1) * (vara + varb + C2);
            acc += num / den; blocks++;
        }
    }
    return blocks > 0 ? acc / (double)blocks : 1.0;
}

static uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }
    struct stat st;
    if (stat(path, &st) != 0) { perror(path); fclose(fp); return NULL; }
    uint8_t *buf = aligned_alloc(32, ((size_t)st.st_size + 31u) & ~(size_t)31u);
    if (!buf) { fprintf(stderr, "oom\n"); fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)st.st_size, fp) != (size_t)st.st_size) {
        perror("fread"); free(buf); fclose(fp); return NULL;
    }
    fclose(fp);
    *out_size = (size_t)st.st_size;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <encoded.c3dc> <original.u8> [frac1 frac2 ...]\n"
                "Feeds progressively smaller prefixes of <encoded.c3dc> to\n"
                "c3d_decoder_chunk_decode and reports PSNR/SSIM vs <original.u8>.\n"
                "Default fractions: 1.0 0.75 0.50 0.25 0.10 0.05 0.02 0.01\n",
                argv[0]);
        return 2;
    }
    size_t enc_size, orig_size;
    uint8_t *enc = read_file(argv[1], &enc_size);
    uint8_t *orig = read_file(argv[2], &orig_size);
    if (!enc || !orig) return 1;
    if (orig_size != CHUNK_BYTES) {
        fprintf(stderr, "%s: expected %u bytes, got %zu\n",
                argv[2], CHUNK_BYTES, orig_size);
        return 1;
    }

    double fracs[16];
    size_t n_fracs;
    if (argc > 3) {
        n_fracs = (size_t)(argc - 3);
        if (n_fracs > 16) n_fracs = 16;
        for (size_t i = 0; i < n_fracs; ++i) fracs[i] = atof(argv[3 + i]);
    } else {
        static const double default_fracs[] = {
            1.00, 0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01
        };
        n_fracs = sizeof default_fracs / sizeof default_fracs[0];
        memcpy(fracs, default_fracs, sizeof default_fracs);
    }

    c3d_decoder *dec = c3d_decoder_new();
    uint8_t *out = aligned_alloc(32, CHUNK_BYTES);

    printf("chunk: %zu B encoded, ratio %.1f:1\n", enc_size,
           (double)CHUNK_BYTES / (double)enc_size);
    printf("%-10s %-12s %-10s %-10s %-10s\n",
           "frac", "bytes_fed", "dec_ms", "PSNR_dB", "SSIM");
    printf("%-10s %-12s %-10s %-10s %-10s\n",
           "----", "---------", "------", "-------", "----");

    for (size_t i = 0; i < n_fracs; ++i) {
        size_t trunc = (size_t)(fracs[i] * (double)enc_size);
        if (trunc < 388) trunc = 388;          /* header-minimum */
        if (trunc > enc_size) trunc = enc_size;
        memset(out, 0xaa, CHUNK_BYTES);
        double t0 = now_s();
        c3d_decoder_chunk_decode(dec, enc, trunc, out);
        double dt = now_s() - t0;
        double p = psnr_u8(orig, out, CHUNK_BYTES);
        double s = ssim_u8(orig, out, SIDE);
        printf("%-10.4f %-12zu %-10.2f %-10.3f %-10.4f\n",
               fracs[i], trunc, dt * 1000.0, p, s);
    }

    free(enc); free(orig); free(out);
    c3d_decoder_free(dec);
    return 0;
}
