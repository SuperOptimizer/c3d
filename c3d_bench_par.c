/* c3d_bench_par — parallel multi-chunk encode/decode demo + scaling bench.
 *
 * Demonstrates that c3d_encoder/c3d_decoder are reentrant for concurrent use:
 * each thread holds its own context.  Reads all 256³ raw u8 files from a
 * corpus directory, fans them out to N threads, encodes at a target ratio,
 * decodes, reports per-thread and aggregate throughput.
 *
 * Usage:  c3d_bench_par <corpus_dir> [n_threads] [target_ratio]
 */

#include "c3d.h"

#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define CHUNK_BYTES ((size_t)256*256*256)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    int       tid;
    char    **paths;
    size_t    n_paths;
    float     target_ratio;
    double    elapsed_s;
    size_t    total_bytes_in;
    size_t    total_bytes_enc;
    double    avg_psnr;
} worker_arg;

static double psnr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) { double e = a[i] - (double)b[i]; sse += e*e; }
    if (sse <= 0) return 200.0;
    return 10.0 * log10(255.0 * 255.0 / (sse / (double)n));
}

static void *worker(void *p) {
    worker_arg *a = p;
    c3d_encoder *enc_ctx = c3d_encoder_new();
    c3d_decoder *dec_ctx = c3d_decoder_new();
    uint8_t *in  = aligned_alloc(32, CHUNK_BYTES);
    uint8_t *enc = aligned_alloc(32, c3d_chunk_encode_max_size());
    uint8_t *dec = aligned_alloc(32, CHUNK_BYTES);

    double psnr_sum = 0; size_t enc_bytes_sum = 0;
    double t0 = now_s();
    for (size_t i = 0; i < a->n_paths; ++i) {
        FILE *fp = fopen(a->paths[i], "rb");
        if (!fp) continue;
        if (fread(in, 1, CHUNK_BYTES, fp) != CHUNK_BYTES) { fclose(fp); continue; }
        fclose(fp);
        size_t n = c3d_encoder_chunk_encode(enc_ctx, in, a->target_ratio, NULL,
                                            enc, c3d_chunk_encode_max_size());
        c3d_decoder_chunk_decode(dec_ctx, enc, n, NULL, dec);
        psnr_sum += psnr_u8(in, dec, CHUNK_BYTES);
        enc_bytes_sum += n;
    }
    a->elapsed_s = now_s() - t0;
    a->total_bytes_in = a->n_paths * CHUNK_BYTES;
    a->total_bytes_enc = enc_bytes_sum;
    a->avg_psnr = psnr_sum / (double)a->n_paths;

    c3d_encoder_free(enc_ctx); c3d_decoder_free(dec_ctx);
    free(in); free(enc); free(dec);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <corpus_dir> [n_threads=4] [target_ratio=10]\n", argv[0]);
        return 2;
    }
    int n_threads = (argc > 2) ? atoi(argv[2]) : 4;
    float target = (argc > 3) ? (float)atof(argv[3]) : 10.0f;

    /* Collect chunk paths. */
    DIR *d = opendir(argv[1]); if (!d) { perror(argv[1]); return 1; }
    char **paths = NULL; size_t n_paths = 0, paths_cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        char full[4096]; snprintf(full, sizeof full, "%s/%s", argv[1], ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode) || (size_t)st.st_size != CHUNK_BYTES) continue;
        if (n_paths == paths_cap) { paths_cap = paths_cap ? paths_cap * 2 : 16; paths = realloc(paths, paths_cap * sizeof *paths); }
        paths[n_paths++] = strdup(full);
    }
    closedir(d);
    printf("corpus: %zu chunks, %d threads, target %.1f:1\n", n_paths, n_threads, (double)target);
    if (n_paths == 0) { fprintf(stderr, "no chunks\n"); return 1; }

    pthread_t *thr = calloc(n_threads, sizeof *thr);
    worker_arg *args = calloc(n_threads, sizeof *args);
    /* Round-robin assignment. */
    size_t per = (n_paths + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        args[t].tid = t;
        args[t].paths = &paths[t * per];
        args[t].n_paths = (t * per >= n_paths) ? 0
                        : (((t + 1) * per <= n_paths) ? per : n_paths - t * per);
        args[t].target_ratio = target;
    }

    double t0 = now_s();
    for (int t = 0; t < n_threads; ++t) pthread_create(&thr[t], NULL, worker, &args[t]);
    for (int t = 0; t < n_threads; ++t) pthread_join(thr[t], NULL);
    double total = now_s() - t0;

    size_t agg_bytes_in = 0, agg_bytes_enc = 0; double psnr_sum = 0; size_t n_done = 0;
    printf("\n%-3s %-7s %12s %10s %10s %s\n",
           "tid", "chunks", "bytes_in", "bytes_enc", "MB/s_in", "PSNR_avg");
    for (int t = 0; t < n_threads; ++t) {
        if (args[t].n_paths == 0) continue;
        double mbps = (args[t].total_bytes_in / (1024.0*1024.0)) / args[t].elapsed_s;
        printf("%-3d %-7zu %12zu %10zu %10.1f %.2f\n",
               t, args[t].n_paths, args[t].total_bytes_in, args[t].total_bytes_enc,
               mbps, args[t].avg_psnr);
        agg_bytes_in  += args[t].total_bytes_in;
        agg_bytes_enc += args[t].total_bytes_enc;
        psnr_sum += args[t].avg_psnr * args[t].n_paths;
        n_done += args[t].n_paths;
    }
    double agg_mbps = (agg_bytes_in / (1024.0*1024.0)) / total;
    printf("---\n");
    printf("aggregate: %zu chunks in %.2fs, %.1f MB/s in (%.2fx scale-up vs ideal=%dx)\n",
           n_done, total, agg_mbps,
           agg_mbps / ((agg_bytes_in / (1024.0*1024.0)) / args[0].elapsed_s) /* approx scale */,
           n_threads);
    printf("compression: %.1f:1, avg PSNR %.2f dB\n",
           (double)agg_bytes_in / agg_bytes_enc, psnr_sum / n_done);

    for (size_t i = 0; i < n_paths; ++i) free(paths[i]);
    free(paths); free(thr); free(args);
    return 0;
}
