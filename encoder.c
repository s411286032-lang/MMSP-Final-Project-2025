// encoder.c - MMSP JPEG Final - Methods 0/1/2/3
// Build: gcc -O2 encoder.c -lm -o encoder

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;        // positive => bottom-up
    uint16_t biPlanes;
    uint16_t biBitCount;      // 24
    uint32_t biCompression;   // 0
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

static void die(const char *msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

static int row_size_bytes_24(int w) {
    int raw = w * 3;
    return (raw + 3) & ~3;
}

typedef struct {
    int w, h;
    // top-down RGB channels
    uint8_t *R, *G, *B;
} ImageRGB;

// ---------- BMP I/O (24-bit BI_RGB only) ----------
static ImageRGB bmp_read_24(const char *path, BITMAPFILEHEADER *out_bfh, BITMAPINFOHEADER *out_bih) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("Cannot open BMP.");

    BITMAPFILEHEADER bfh;
    BITMAPINFOHEADER bih;

    if (fread(&bfh, sizeof(bfh), 1, fp) != 1) die("Read BITMAPFILEHEADER failed.");
    if (fread(&bih, sizeof(bih), 1, fp) != 1) die("Read BITMAPINFOHEADER failed.");

    if (bfh.bfType != 0x4D42) die("Not a BMP.");
    if (bih.biPlanes != 1) die("Unsupported BMP planes.");
    if (bih.biBitCount != 24) die("Only 24-bit BMP supported.");
    if (bih.biCompression != 0) die("Only BI_RGB BMP supported.");

    int w = bih.biWidth;
    int h = (bih.biHeight >= 0) ? bih.biHeight : -bih.biHeight;
    int bottom_up = (bih.biHeight >= 0);
    if (w <= 0 || h <= 0) die("Invalid BMP dimensions.");

    int rowSize = row_size_bytes_24(w);

    uint8_t *R = (uint8_t*)malloc((size_t)w * h);
    uint8_t *G = (uint8_t*)malloc((size_t)w * h);
    uint8_t *B = (uint8_t*)malloc((size_t)w * h);
    if (!R || !G || !B) die("OOM (RGB).");

    if (fseek(fp, (long)bfh.bfOffBits, SEEK_SET) != 0) die("Seek pixel data failed.");

    uint8_t *row = (uint8_t*)malloc((size_t)rowSize);
    if (!row) die("OOM (row).");

    for (int file_row = 0; file_row < h; file_row++) {
        if (fread(row, 1, (size_t)rowSize, fp) != (size_t)rowSize) die("Read row failed.");
        int img_row = bottom_up ? (h - 1 - file_row) : file_row;
        for (int x = 0; x < w; x++) {
            uint8_t b = row[x*3 + 0];
            uint8_t g = row[x*3 + 1];
            uint8_t r = row[x*3 + 2];
            size_t idx = (size_t)img_row * w + x;
            R[idx] = r; G[idx] = g; B[idx] = b;
        }
    }

    free(row);
    fclose(fp);

    if (out_bfh) *out_bfh = bfh;
    if (out_bih) *out_bih = bih;

    ImageRGB img = { w, h, R, G, B };
    return img;
}

static void bmp_write_24(const char *path, const ImageRGB *img) {
    int w = img->w, h = img->h;
    int rowSize = row_size_bytes_24(w);
    uint32_t imageSize = (uint32_t)rowSize * (uint32_t)h;

    BITMAPFILEHEADER bfh;
    BITMAPINFOHEADER bih;

    bfh.bfType = 0x4D42;
    bfh.bfOffBits = (uint32_t)(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER));
    bfh.bfSize = bfh.bfOffBits + imageSize;
    bfh.bfReserved1 = 0;
    bfh.bfReserved2 = 0;

    memset(&bih, 0, sizeof(bih));
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biWidth = w;
    bih.biHeight = h;               // bottom-up output
    bih.biPlanes = 1;
    bih.biBitCount = 24;
    bih.biCompression = 0;
    bih.biSizeImage = imageSize;

    FILE *fp = fopen(path, "wb");
    if (!fp) die("Cannot write BMP.");

    if (fwrite(&bfh, sizeof(bfh), 1, fp) != 1) die("Write bfh failed.");
    if (fwrite(&bih, sizeof(bih), 1, fp) != 1) die("Write bih failed.");

    uint8_t *row = (uint8_t*)malloc((size_t)rowSize);
    if (!row) die("OOM row write.");

    for (int file_row = 0; file_row < h; file_row++) {
        int img_row = h - 1 - file_row; // bottom-up
        for (int x = 0; x < w; x++) {
            size_t idx = (size_t)img_row * w + x;
            row[x*3 + 0] = img->B[idx];
            row[x*3 + 1] = img->G[idx];
            row[x*3 + 2] = img->R[idx];
        }
        for (int p = w*3; p < rowSize; p++) row[p] = 0;
        if (fwrite(row, 1, (size_t)rowSize, fp) != (size_t)rowSize) die("Write row failed.");
    }

    free(row);
    fclose(fp);
}

// ---------- Utility: write/read dim ----------
static void dim_write(const char *path, int w, int h) {
    FILE *f = fopen(path, "w");
    if (!f) die("Cannot write dim.txt");
    fprintf(f, "%d %d\n", w, h);
    fclose(f);
}

// ---------- RGB <-> YCbCr (full range) ----------
static inline double clamp255(double x) {
    if (x < 0.0) return 0.0;
    if (x > 255.0) return 255.0;
    return x;
}

static void rgb_to_ycbcr(const ImageRGB *rgb, double *Y, double *Cb, double *Cr) {
    // ITU-R BT.601 (full-range style)
    int w = rgb->w, h = rgb->h;
    for (int i = 0; i < w*h; i++) {
        double R = (double)rgb->R[i];
        double G = (double)rgb->G[i];
        double B = (double)rgb->B[i];
        double y  =  0.299    * R + 0.587    * G + 0.114    * B;
        double cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128.0;
        double cr =  0.5      * R - 0.418688 * G - 0.081312 * B + 128.0;
        Y[i]  = y;
        Cb[i] = cb;
        Cr[i] = cr;
    }
}

// ---------- Padding to multiple of 8 (edge replicate) ----------
static void pad_channel_to8(const double *in, int w, int h, double **out, int *W8, int *H8) {
    int pw = (w + 7) / 8 * 8;
    int ph = (h + 7) / 8 * 8;
    double *p = (double*)malloc((size_t)pw * ph * sizeof(double));
    if (!p) die("OOM pad.");

    for (int y = 0; y < ph; y++) {
        int sy = (y < h) ? y : (h - 1);
        for (int x = 0; x < pw; x++) {
            int sx = (x < w) ? x : (w - 1);
            p[y*pw + x] = in[sy*w + sx];
        }
    }
    *out = p; *W8 = pw; *H8 = ph;
}

// ---------- 8x8 DCT/IDCT (double) ----------
static double c8(int u) { return (u == 0) ? (1.0 / sqrt(2.0)) : 1.0; }

static void dct8x8(const double in[64], double out[64]) {
    // DCT-II, orthonormal scaling
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            double sum = 0.0;
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    double s = in[y*8 + x];
                    sum += s
                         * cos(((2*x + 1) * u * M_PI) / 16.0)
                         * cos(((2*y + 1) * v * M_PI) / 16.0);
                }
            }
            out[v*8 + u] = 0.25 * c8(u) * c8(v) * sum;
        }
    }
}

static void idct8x8(const double in[64], double out[64]) {
    // IDCT (inverse of above)
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            double sum = 0.0;
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    double F = in[v*8 + u];
                    sum += c8(u)*c8(v) * F
                         * cos(((2*x + 1) * u * M_PI) / 16.0)
                         * cos(((2*y + 1) * v * M_PI) / 16.0);
                }
            }
            out[y*8 + x] = 0.25 * sum;
        }
    }
}

// ---------- Quantization tables (standard JPEG) ----------
static const int QTY_std[64] = {
    16,11,10,16,24,40,51,61,
    12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,
    24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103,99
};

static const int QTC_std[64] = {
    17,18,24,47,99,99,99,99,
    18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99,
    47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99
};

static void write_qtable_txt(const char *path, const int Qt[64]) {
    FILE *f = fopen(path, "w");
    if (!f) die("Cannot write Qt txt.");
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            fprintf(f, "%d%s", Qt[r*8 + c], (c==7)?"":" ");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// ---------- ZigZag order ----------
static const int zigzag[64] = {
     0, 1, 5, 6,14,15,27,28,
     2, 4, 7,13,16,26,29,42,
     3, 8,12,17,25,30,41,43,
     9,11,18,24,31,40,44,53,
    10,19,23,32,39,45,52,54,
    20,22,33,38,46,51,55,60,
    21,34,37,47,50,56,59,61,
    35,36,48,49,57,58,62,63
};

static void zigzag_scan(const int16_t in[64], int16_t out[64]) {
    for (int i = 0; i < 64; i++) out[i] = in[zigzag[i]];
}

static void zigzag_unscan(const int16_t in[64], int16_t out[64]) {
    for (int i = 0; i < 64; i++) out[zigzag[i]] = in[i];
}

// ---------- Method 0 ----------
static void encoder_method0(int argc, char *argv[]) {
    if (argc != 7) die("Usage: encoder 0 in.bmp R.txt G.txt B.txt dim.txt");
    const char *in_bmp = argv[2];
    const char *Rtxt   = argv[3];
    const char *Gtxt   = argv[4];
    const char *Btxt   = argv[5];
    const char *dimtxt = argv[6];

    BITMAPFILEHEADER bfh; BITMAPINFOHEADER bih;
    ImageRGB img = bmp_read_24(in_bmp, &bfh, &bih);

    dim_write(dimtxt, img.w, img.h);

    FILE *fr = fopen(Rtxt, "w");
    FILE *fg = fopen(Gtxt, "w");
    FILE *fb = fopen(Btxt, "w");
    if (!fr || !fg || !fb) die("Cannot write R/G/B txt.");

    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            size_t idx = (size_t)y*img.w + x;
            fprintf(fr, "%u%s", (unsigned)img.R[idx], (x==img.w-1)?"":" ");
            fprintf(fg, "%u%s", (unsigned)img.G[idx], (x==img.w-1)?"":" ");
            fprintf(fb, "%u%s", (unsigned)img.B[idx], (x==img.w-1)?"":" ");
        }
        fprintf(fr, "\n"); fprintf(fg, "\n"); fprintf(fb, "\n");
    }
    fclose(fr); fclose(fg); fclose(fb);

    free(img.R); free(img.G); free(img.B);
}

// ---------- Helper: compute DCT+Quant for padded channel ----------
typedef struct {
    int w, h;        // padded dims
    int bx, by;      // blocks in x/y
    // per-block quantized coefficients (int16) length = blocks*64
    int16_t *qF;
    // per-block quantization error in frequency domain (float) length = blocks*64
    float   *eF;
    // per-frequency accum for SQNR
    double  sigPow[64];
    double  errPow[64];
    uint64_t count; // number of blocks accumulated
} ChanFreq;

static void init_chanfreq(ChanFreq *cf, int w, int h, int need_eF) {
    cf->w = w; cf->h = h;
    cf->bx = w/8; cf->by = h/8;
    size_t blocks = (size_t)cf->bx * cf->by;
    cf->qF = (int16_t*)malloc(blocks * 64 * sizeof(int16_t));
    if (!cf->qF) die("OOM qF.");
    cf->eF = need_eF ? (float*)malloc(blocks * 64 * sizeof(float)) : NULL;
    if (need_eF && !cf->eF) die("OOM eF.");
    for (int k=0;k<64;k++){ cf->sigPow[k]=0.0; cf->errPow[k]=0.0; }
    cf->count = 0;
}

static void free_chanfreq(ChanFreq *cf) {
    free(cf->qF);
    if (cf->eF) free(cf->eF);
}

static void dct_quant_channel(const double *padded, int w, int h, const int Qt[64], ChanFreq *out, int store_eF) {
    init_chanfreq(out, w, h, store_eF);
    size_t blocks = (size_t)out->bx * out->by;

    for (int by = 0; by < out->by; by++) {
        for (int bx = 0; bx < out->bx; bx++) {
            double block[64];
            double F[64];

            // load 8x8 and level shift by -128
            for (int yy=0; yy<8; yy++) {
                for (int xx=0; xx<8; xx++) {
                    int x = bx*8 + xx;
                    int y = by*8 + yy;
                    block[yy*8 + xx] = padded[y*w + x] - 128.0;
                }
            }

            dct8x8(block, F);

            size_t bidx = (size_t)(by*out->bx + bx);
            size_t base = bidx * 64;

            for (int k=0;k<64;k++) {
                double q = (double)Qt[k];
                double qv = nearbyint(F[k] / q);      // round to nearest
                int16_t qshort;
                if (qv < -32768) qshort = -32768;
                else if (qv > 32767) qshort = 32767;
                else qshort = (int16_t)qv;

                out->qF[base + k] = qshort;

                double recon = (double)qshort * q;
                double err = F[k] - recon;

                out->sigPow[k] += F[k]*F[k];
                out->errPow[k] += err*err;

                if (store_eF) out->eF[base + k] = (float)err;
            }
        }
    }
    out->count = blocks;
}

static void print_sqnr_3x64(const ChanFreq *Y, const ChanFreq *Cb, const ChanFreq *Cr) {
    // SQNR_k = 10log10( sum(F^2)/sum(err^2) )
    for (int ch = 0; ch < 3; ch++) {
        const ChanFreq *C = (ch==0)?Y:(ch==1)?Cb:Cr;
        for (int k=0;k<64;k++) {
            double s = C->sigPow[k];
            double e = C->errPow[k];
            double sqnr;
            if (e <= 0.0) sqnr = 999.0;
            else sqnr = 10.0 * log10(s / e);
            printf("%8.3f%s", sqnr, (k==63)?"":" ");
        }
        printf("\n");
    }
}

// ---------- Method 1 (encoder) ----------
static void encoder_method1(int argc, char *argv[]) {
    // encoder 1 Kimberly.bmp Qt_Y.txt Qt_Cb.txt Qt_Cr.txt dim.txt qF_Y.raw qF_Cb.raw qF_Cr.raw eF_Y.raw eF_Cb.raw eF_Cr.raw
    if (argc != 13) die("Usage: encoder 1 in.bmp Qt_Y.txt Qt_Cb.txt Qt_Cr.txt dim.txt qF_Y.raw qF_Cb.raw qF_Cr.raw eF_Y.raw eF_Cb.raw eF_Cr.raw");

    const char *in_bmp  = argv[2];
    const char *QtY_txt = argv[3];
    const char *QtCb_txt= argv[4];
    const char *QtCr_txt= argv[5];
    const char *dim_txt = argv[6];
    const char *qFY_raw = argv[7];
    const char *qFCb_raw= argv[8];
    const char *qFCr_raw= argv[9];
    const char *eFY_raw = argv[10];
    const char *eFCb_raw= argv[11];
    const char *eFCr_raw= argv[12];

    ImageRGB img = bmp_read_24(in_bmp, NULL, NULL);
    dim_write(dim_txt, img.w, img.h);

    write_qtable_txt(QtY_txt, QTY_std);
    write_qtable_txt(QtCb_txt, QTC_std);
    write_qtable_txt(QtCr_txt, QTC_std);

    double *Y = (double*)malloc((size_t)img.w*img.h*sizeof(double));
    double *Cb= (double*)malloc((size_t)img.w*img.h*sizeof(double));
    double *Cr= (double*)malloc((size_t)img.w*img.h*sizeof(double));
    if (!Y||!Cb||!Cr) die("OOM YCbCr.");
    rgb_to_ycbcr(&img, Y, Cb, Cr);

    double *Yp,*Cbp,*Crp; int W8,H8;
    pad_channel_to8(Y, img.w, img.h, &Yp, &W8, &H8);
    int W8b,H8b;
    pad_channel_to8(Cb, img.w, img.h, &Cbp, &W8b, &H8b);
    int W8c,H8c;
    pad_channel_to8(Cr, img.w, img.h, &Crp, &W8c, &H8c);
    // all padded dims should match
    if (W8!=W8b || W8!=W8c || H8!=H8b || H8!=H8c) die("Pad mismatch.");

    ChanFreq FY, FCb, FCr;
    dct_quant_channel(Yp, W8, H8, QTY_std, &FY, 1);
    dct_quant_channel(Cbp,W8, H8, QTC_std, &FCb,1);
    dct_quant_channel(Crp,W8, H8, QTC_std, &FCr,1);

    // Write qF raw as int16 in block row-major, inside block row-major
    FILE *fqY = fopen(qFY_raw, "wb");
    FILE *fqCb= fopen(qFCb_raw,"wb");
    FILE *fqCr= fopen(qFCr_raw,"wb");
    if (!fqY||!fqCb||!fqCr) die("Cannot write qF raw.");
    fwrite(FY.qF,  sizeof(int16_t), (size_t)FY.bx*FY.by*64, fqY);
    fwrite(FCb.qF, sizeof(int16_t), (size_t)FCb.bx*FCb.by*64, fqCb);
    fwrite(FCr.qF, sizeof(int16_t), (size_t)FCr.bx*FCr.by*64, fqCr);
    fclose(fqY); fclose(fqCb); fclose(fqCr);

    // Write eF raw as float
    FILE *feY = fopen(eFY_raw, "wb");
    FILE *feCb= fopen(eFCb_raw,"wb");
    FILE *feCr= fopen(eFCr_raw,"wb");
    if (!feY||!feCb||!feCr) die("Cannot write eF raw.");
    fwrite(FY.eF,  sizeof(float), (size_t)FY.bx*FY.by*64, feY);
    fwrite(FCb.eF, sizeof(float), (size_t)FCb.bx*FCb.by*64, feCb);
    fwrite(FCr.eF, sizeof(float), (size_t)FCr.bx*FCr.by*64, feCr);
    fclose(feY); fclose(feCb); fclose(feCr);

    // Print 3x64 SQNR (Y/Cb/Cr)
    print_sqnr_3x64(&FY, &FCb, &FCr);

    // cleanup
    free(img.R); free(img.G); free(img.B);
    free(Y); free(Cb); free(Cr);
    free(Yp); free(Cbp); free(Crp);
    free_chanfreq(&FY); free_chanfreq(&FCb); free_chanfreq(&FCr);
}

// ---------- Method 2: DPCM + ZigZag + RLE ----------
static void apply_dc_dpcm_inplace(int16_t *qF, size_t blocks) {
    int16_t prev = 0;
    for (size_t b = 0; b < blocks; b++) {
        int16_t dc = qF[b*64 + 0];
        int16_t diff = (int16_t)(dc - prev);
        qF[b*64 + 0] = diff;
        prev = dc;
    }
}

static void undo_dc_dpcm_inplace(int16_t *qF, size_t blocks) {
    int16_t prev = 0;
    for (size_t b = 0; b < blocks; b++) {
        int16_t diff = qF[b*64 + 0];
        int16_t dc = (int16_t)(prev + diff);
        qF[b*64 + 0] = dc;
        prev = dc;
    }
}

typedef struct { uint8_t skip; int16_t val; } Pair;

static size_t rle_encode_zigzag(const int16_t coeff64[64], Pair *pairs, size_t maxPairs) {
    // encode full 64 in zigzag order into (skip,value) list, skipping zeros; trailing zeros omitted
    int16_t zz[64];
    zigzag_scan(coeff64, zz);
    size_t n = 0;
    int skip = 0;
    for (int i = 0; i < 64; i++) {
        if (zz[i] == 0) { skip++; continue; }
        if (n >= maxPairs) die("RLE pairs overflow.");
        pairs[n].skip = (uint8_t)skip;
        pairs[n].val  = zz[i];
        n++;
        skip = 0;
    }
    return n;
}

static void rle_decode_zigzag(Pair *pairs, size_t nPairs, int16_t out64[64]) {
    // rebuild zigzag array from pairs, remaining zeros
    int16_t zz[64];
    for (int i=0;i<64;i++) zz[i]=0;
    int pos = 0;
    for (size_t i=0;i<nPairs;i++) {
        pos += pairs[i].skip;
        if (pos < 0 || pos >= 64) die("RLE decode pos out of range.");
        zz[pos] = pairs[i].val;
        pos++;
    }
    zigzag_unscan(zz, out64);
}

static void encoder_method2(int argc, char *argv[]) {
    // encoder 2 Kimberly.bmp ascii rle_code.txt
    // encoder 2 Kimberly.bmp binary rle_code.bin
    if (argc != 5) die("Usage: encoder 2 in.bmp ascii|binary out");

    const char *in_bmp = argv[2];
    const char *fmt    = argv[3];
    const char *out    = argv[4];

    // First compute qF like method1 (no need eF)
    ImageRGB img = bmp_read_24(in_bmp, NULL, NULL);

    double *Y = (double*)malloc((size_t)img.w*img.h*sizeof(double));
    double *Cb= (double*)malloc((size_t)img.w*img.h*sizeof(double));
    double *Cr= (double*)malloc((size_t)img.w*img.h*sizeof(double));
    if (!Y||!Cb||!Cr) die("OOM YCbCr.");
    rgb_to_ycbcr(&img, Y, Cb, Cr);

    double *Yp,*Cbp,*Crp; int W8,H8;
    pad_channel_to8(Y, img.w, img.h, &Yp, &W8, &H8);
    int W8b,H8b; pad_channel_to8(Cb, img.w, img.h, &Cbp, &W8b, &H8b);
    int W8c,H8c; pad_channel_to8(Cr, img.w, img.h, &Crp, &W8c, &H8c);

    ChanFreq FY, FCb, FCr;
    dct_quant_channel(Yp, W8, H8, QTY_std, &FY, 0);
    dct_quant_channel(Cbp,W8, H8, QTC_std, &FCb,0);
    dct_quant_channel(Crp,W8, H8, QTC_std, &FCr,0);

    size_t blocks = (size_t)FY.bx * FY.by;

    // DPCM on DC
    apply_dc_dpcm_inplace(FY.qF, blocks);
    apply_dc_dpcm_inplace(FCb.qF, blocks);
    apply_dc_dpcm_inplace(FCr.qF, blocks);

    if (strcmp(fmt, "ascii") == 0) {
        FILE *f = fopen(out, "w");
        if (!f) die("Cannot write rle_code.txt");

        // first row: original image size
        fprintf(f, "%d %d\n", img.w, img.h);

        Pair pairs[128];
        for (int m=0;m<FY.by;m++) {
            for (int n=0;n<FY.bx;n++) {
                size_t b = (size_t)m*FY.bx + n;

                // Y
                size_t base = b*64;
                size_t np = rle_encode_zigzag(&FY.qF[base], pairs, 128);
                fprintf(f, "($%d,$%d, Y)", m, n);
                for (size_t i=0;i<np;i++) fprintf(f, " %u %d", pairs[i].skip, (int)pairs[i].val);
                fprintf(f, "\n");

                // Cb
                base = b*64;
                np = rle_encode_zigzag(&FCb.qF[base], pairs, 128);
                fprintf(f, "($%d,$%d, Cb)", m, n);
                for (size_t i=0;i<np;i++) fprintf(f, " %u %d", pairs[i].skip, (int)pairs[i].val);
                fprintf(f, "\n");

                // Cr
                base = b*64;
                np = rle_encode_zigzag(&FCr.qF[base], pairs, 128);
                fprintf(f, "($%d,$%d, Cr)", m, n);
                for (size_t i=0;i<np;i++) fprintf(f, " %u %d", pairs[i].skip, (int)pairs[i].val);
                fprintf(f, "\n");
            }
        }
        fclose(f);
        printf("Method 2 ascii done: %s\n", out);

    } else if (strcmp(fmt, "binary") == 0) {
        FILE *f = fopen(out, "wb");
        if (!f) die("Cannot write rle_code.bin");

        // simple binary format:
        // magic 'RLE2' (0x32454C52), w(int32), h(int32), bx(int32), by(int32)
        // then for each block (row-major), for each channel Y,Cb,Cr:
        // nPairs(uint16), then repeated: skip(uint8), val(int16)
        uint32_t magic = 0x32454C52;
        int32_t w = img.w, h = img.h;
        int32_t bx = FY.bx, by = FY.by;
        fwrite(&magic, 4, 1, f);
        fwrite(&w, 4, 1, f);
        fwrite(&h, 4, 1, f);
        fwrite(&bx,4, 1, f);
        fwrite(&by,4, 1, f);

        Pair pairs[256];
        for (size_t b=0;b<blocks;b++) {
            for (int ch=0; ch<3; ch++) {
                int16_t *src = (ch==0)?FY.qF:((ch==1)?FCb.qF:FCr.qF);
                size_t base = b*64;
                size_t np = rle_encode_zigzag(&src[base], pairs, 256);
                if (np > 65535) die("np too big.");
                uint16_t np16 = (uint16_t)np;
                fwrite(&np16, 2, 1, f);
                for (size_t i=0;i<np;i++) {
                    fwrite(&pairs[i].skip, 1, 1, f);
                    fwrite(&pairs[i].val,  2, 1, f);
                }
            }
        }
        fclose(f);

        // compression rate (approx)
        // original BMP size:
        int rowSize = row_size_bytes_24(img.w);
        double origBytes = (double)(sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER) + rowSize*img.h);
        // encoded bytes:
        FILE *fb = fopen(out, "rb");
        fseek(fb, 0, SEEK_END);
        double encBytes = (double)ftell(fb);
        fclose(fb);

        printf("Compression rate Y/Cb/Cr (overall): %.6f (encoded/original)\n", encBytes/origBytes);

    } else {
        die("fmt must be ascii or binary.");
    }

    // cleanup
    free(img.R); free(img.G); free(img.B);
    free(Y); free(Cb); free(Cr);
    free(Yp); free(Cbp); free(Crp);
    free_chanfreq(&FY); free_chanfreq(&FCb); free_chanfreq(&FCr);
}

// ---------- Huffman coding for Method 3 ----------
typedef struct HuffNode {
    int isLeaf;
    int32_t sym;
    uint64_t freq;
    struct HuffNode *l, *r;
} HuffNode;

typedef struct {
    int32_t sym;
    uint64_t freq;
} SymFreq;

typedef struct {
    int32_t sym;
    uint32_t code;   // up to 32 bits for simplicity
    uint8_t  len;    // code length
} Code;

static HuffNode* new_node(int isLeaf, int32_t sym, uint64_t freq, HuffNode* l, HuffNode* r) {
    HuffNode *n = (HuffNode*)malloc(sizeof(HuffNode));
    if (!n) die("OOM HuffNode.");
    n->isLeaf=isLeaf; n->sym=sym; n->freq=freq; n->l=l; n->r=r;
    return n;
}

static void free_tree(HuffNode *n) {
    if (!n) return;
    free_tree(n->l);
    free_tree(n->r);
    free(n);
}

static HuffNode* build_huffman_tree(SymFreq *arr, size_t n) {
    // naive O(n^2): okay for assignment scale
    HuffNode **nodes = (HuffNode**)malloc(n*sizeof(HuffNode*));
    if (!nodes) die("OOM nodes.");
    for (size_t i=0;i<n;i++) nodes[i] = new_node(1, arr[i].sym, arr[i].freq, NULL, NULL);
    size_t m = n;

    while (m > 1) {
        size_t a=0,b=1;
        if (nodes[b]->freq < nodes[a]->freq) { size_t t=a;a=b;b=t; }
        for (size_t i=2;i<m;i++) {
            if (nodes[i]->freq < nodes[a]->freq) { b=a; a=i; }
            else if (nodes[i]->freq < nodes[b]->freq) { b=i; }
        }
        HuffNode *na = nodes[a];
        HuffNode *nb = nodes[b];
        HuffNode *parent = new_node(0, 0, na->freq + nb->freq, na, nb);

        if (a > b) { size_t t=a;a=b;b=t; }
        nodes[a] = parent;
        nodes[b] = nodes[m-1];
        m--;
    }
    HuffNode *root = nodes[0];
    free(nodes);
    return root;
}

static void gen_codes_rec(HuffNode *n, uint32_t code, uint8_t len, Code *out, size_t *idx, size_t cap) {
    if (!n) return;
    if (n->isLeaf) {
        if (*idx >= cap) die("Code table overflow.");
        out[*idx].sym = n->sym;
        out[*idx].code = code;
        out[*idx].len = (len==0)?1:len; // avoid zero-length
        (*idx)++;
        return;
    }
    gen_codes_rec(n->l, (code<<1),     (uint8_t)(len+1), out, idx, cap);
    gen_codes_rec(n->r, (code<<1)|1u,  (uint8_t)(len+1), out, idx, cap);
}

static int cmp_symfreq(const void *a, const void *b) {
    const SymFreq *x = (const SymFreq*)a;
    const SymFreq *y = (const SymFreq*)b;
    if (x->sym < y->sym) return -1;
    if (x->sym > y->sym) return 1;
    return 0;
}

static int find_code(const Code *codes, size_t n, int32_t sym, uint32_t *code, uint8_t *len) {
    for (size_t i=0;i<n;i++) {
        if (codes[i].sym == sym) { *code=codes[i].code; *len=codes[i].len; return 1; }
    }
    return 0;
}

typedef struct {
    uint8_t *buf;
    size_t cap;
    size_t nbytes;
    uint8_t bitpos; // 0..7
} BitWriter;

static void bw_init(BitWriter *bw) {
    bw->cap = 1024;
    bw->buf = (uint8_t*)malloc(bw->cap);
    if (!bw->buf) die("OOM bitbuf.");
    bw->nbytes = 0;
    bw->bitpos = 0;
}

static void bw_push_bit(BitWriter *bw, int bit) {
    if (bw->nbytes == bw->cap) {
        bw->cap *= 2;
        bw->buf = (uint8_t*)realloc(bw->buf, bw->cap);
        if (!bw->buf) die("OOM realloc bitbuf.");
    }
    if (bw->bitpos == 0) bw->buf[bw->nbytes] = 0;
    if (bit) bw->buf[bw->nbytes] |= (uint8_t)(1u << (7 - bw->bitpos));
    bw->bitpos++;
    if (bw->bitpos == 8) { bw->bitpos = 0; bw->nbytes++; }
}

static void bw_push_code(BitWriter *bw, uint32_t code, uint8_t len) {
    // code is in MSB-first in our recursion; we stored as bits in 'code' from root path
    // len bits: output from MSB to LSB
    for (int i = len-1; i >= 0; i--) {
        int bit = (code >> i) & 1;
        bw_push_bit(bw, bit);
    }
}

static size_t bw_finish(BitWriter *bw) {
    if (bw->bitpos != 0) bw->nbytes++; // flush last partial byte
    return bw->nbytes;
}

static void bw_free(BitWriter *bw) {
    free(bw->buf);
}

static int32_t pack_symbol(uint8_t skip, int16_t val) {
    // pack (skip,val) into int32
    return (int32_t)(((uint32_t)skip << 16) | (uint16_t)val);
}
static int32_t sym_EOB(void) { return (int32_t)0x7FFFFFFF; } // end of block marker

static void encoder_method3(int argc, char *argv[]) {
    // encoder 3 Kimberly.bmp ascii codebook.txt huffman_code.txt
    // encoder 3 Kimberly.bmp binary codebook.txt huffman_code.bin
    if (argc != 6) die("Usage: encoder 3 in.bmp ascii|binary codebook.txt huffman_code.(txt|bin)");

    const char *in_bmp = argv[2];
    const char *fmt    = argv[3];
    const char *codebook_txt = argv[4];
    const char *out = argv[5];

    // Produce qF same as method2 (with DPCM)
    ImageRGB img = bmp_read_24(in_bmp, NULL, NULL);

    double *Y = (double*)malloc((size_t)img.w*img.h*sizeof(double));
    double *Cb= (double*)malloc((size_t)img.w*img.h*sizeof(double));
    double *Cr= (double*)malloc((size_t)img.w*img.h*sizeof(double));
    if (!Y||!Cb||!Cr) die("OOM YCbCr.");
    rgb_to_ycbcr(&img, Y, Cb, Cr);

    double *Yp,*Cbp,*Crp; int W8,H8;
    pad_channel_to8(Y, img.w, img.h, &Yp, &W8, &H8);
    int W8b,H8b; pad_channel_to8(Cb, img.w, img.h, &Cbp, &W8b, &H8b);
    int W8c,H8c; pad_channel_to8(Cr, img.w, img.h, &Crp, &W8c, &H8c);

    ChanFreq FY, FCb, FCr;
    dct_quant_channel(Yp, W8, H8, QTY_std, &FY, 0);
    dct_quant_channel(Cbp,W8, H8, QTC_std, &FCb,0);
    dct_quant_channel(Crp,W8, H8, QTC_std, &FCr,0);

    size_t blocks = (size_t)FY.bx * FY.by;

    apply_dc_dpcm_inplace(FY.qF, blocks);
    apply_dc_dpcm_inplace(FCb.qF, blocks);
    apply_dc_dpcm_inplace(FCr.qF, blocks);

    // Gather symbols frequency from RLE pairs + EOB
    // We'll build dynamic array of SymFreq by simple map (linear search ok for assignment).
    SymFreq *sf = NULL; size_t nsf = 0; size_t cap = 0;

    auto void add_sym(int32_t sym) {
        for (size_t i=0;i<nsf;i++) {
            if (sf[i].sym == sym) { sf[i].freq++; return; }
        }
        if (nsf == cap) {
            cap = (cap==0)?1024:(cap*2);
            sf = (SymFreq*)realloc(sf, cap*sizeof(SymFreq));
            if (!sf) die("OOM symfreq realloc.");
        }
        sf[nsf].sym = sym;
        sf[nsf].freq = 1;
        nsf++;
    }

    Pair pairs[256];

    for (size_t b=0;b<blocks;b++) {
        for (int ch=0; ch<3; ch++) {
            int16_t *src = (ch==0)?FY.qF:((ch==1)?FCb.qF:FCr.qF);
            size_t base = b*64;
            size_t np = rle_encode_zigzag(&src[base], pairs, 256);
            for (size_t i=0;i<np;i++) add_sym(pack_symbol(pairs[i].skip, pairs[i].val));
            add_sym(sym_EOB());
        }
    }

    // Sort by sym for stable codebook print
    qsort(sf, nsf, sizeof(SymFreq), cmp_symfreq);

    // Build Huffman tree and codes
    HuffNode *root = build_huffman_tree(sf, nsf);
    Code *codes = (Code*)malloc(nsf * sizeof(Code));
    if (!codes) die("OOM codes.");
    size_t idx = 0;
    gen_codes_rec(root, 0u, 0u, codes, &idx, nsf);
    size_t nCodes = idx;

    // Write codebook.txt (symbol count codeword)
    FILE *fc = fopen(codebook_txt, "w");
    if (!fc) die("Cannot write codebook.txt");
    fprintf(fc, "symbol\tcount\tcodeword\n");
    for (size_t i=0;i<nsf;i++) {
        int32_t sym = sf[i].sym;
        uint32_t code=0; uint8_t len=0;
        if (!find_code(codes, nCodes, sym, &code, &len)) die("Missing code.");
        // print codeword as bits
        fprintf(fc, "%d\t%llu\t", (int)sym, (unsigned long long)sf[i].freq);
        for (int b=len-1;b>=0;b--) fprintf(fc, "%d", (code>>b)&1);
        fprintf(fc, "\n");
    }
    fclose(fc);

    // Encode bitstream
    BitWriter bw; bw_init(&bw);

    for (size_t b=0;b<blocks;b++) {
        for (int ch=0; ch<3; ch++) {
            int16_t *src = (ch==0)?FY.qF:((ch==1)?FCb.qF:FCr.qF);
            size_t base = b*64;
            size_t np = rle_encode_zigzag(&src[base], pairs, 256);
            for (size_t i=0;i<np;i++) {
                int32_t sym = pack_symbol(pairs[i].skip, pairs[i].val);
                uint32_t code=0; uint8_t len=0;
                if (!find_code(codes, nCodes, sym, &code, &len)) die("Missing code.");
                bw_push_code(&bw, code, len);
            }
            // EOB
            {
                int32_t sym = sym_EOB();
                uint32_t code=0; uint8_t len=0;
                if (!find_code(codes, nCodes, sym, &code, &len)) die("Missing code EOB.");
                bw_push_code(&bw, code, len);
            }
        }
    }
    size_t nbytes = bw_finish(&bw);

    if (strcmp(fmt, "ascii") == 0) {
        FILE *fo = fopen(out, "w");
        if (!fo) die("Cannot write huffman_code.txt");
        fprintf(fo, "size: %d %d\n", img.w, img.h);
        fprintf(fo, "bitstream: ");
        // print bits as 0/1 (can be huge)
        for (size_t i=0;i<nbytes;i++) {
            for (int b=0;b<8;b++) {
                int bit = (bw.buf[i] >> (7-b)) & 1;
                fprintf(fo, "%d", bit);
            }
        }
        fprintf(fo, "\n");
        fclose(fo);
    } else if (strcmp(fmt, "binary") == 0) {
        FILE *fo = fopen(out, "wb");
        if (!fo) die("Cannot write huffman_code.bin");
        uint32_t magic = 0x33465548; // 'HUF3'
        fwrite(&magic, 4, 1, fo);
        int32_t w = img.w, h = img.h;
        fwrite(&w, 4, 1, fo);
        fwrite(&h, 4, 1, fo);
        uint32_t bitlen = (uint32_t)(nbytes * 8);
        fwrite(&bitlen, 4, 1, fo);
        fwrite(bw.buf, 1, nbytes, fo);
        fclose(fo);

        // compression rate
        int rowSize = row_size_bytes_24(img.w);
        double origBytes = (double)(sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER) + rowSize*img.h);
        FILE *fb = fopen(out, "rb");
        fseek(fb, 0, SEEK_END);
        double encBytes = (double)ftell(fb);
        fclose(fb);
        printf("Compression rate overall: %.6f (encoded/original)\n", encBytes/origBytes);
    } else {
        die("fmt must be ascii or binary.");
    }

    bw_free(&bw);
    free_tree(root);
    free(codes);
    free(sf);

    // cleanup
    free(img.R); free(img.G); free(img.B);
    free(Y); free(Cb); free(Cr);
    free(Yp); free(Cbp); free(Crp);
    free_chanfreq(&FY); free_chanfreq(&FCb); free_chanfreq(&FCr);
}

int main(int argc, char *argv[]) {
    if (argc < 2) die("Usage: encoder <0|1|2|3> ...");
    int mode = atoi(argv[1]);
    switch (mode) {
        case 0: encoder_method0(argc, argv); break;
        case 1: encoder_method1(argc, argv); break;
        case 2: encoder_method2(argc, argv); break;
        case 3: encoder_method3(argc, argv); break;
        default: die("Invalid mode (0~3).");
    }
    return 0;
}
