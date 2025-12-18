// decoder.c - MMSP JPEG Final - Methods 0/1/2/3
// Build: gcc -O2 decoder.c -lm -o decoder

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
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
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
    uint8_t *R, *G, *B;
} ImageRGB;

static ImageRGB rgb_alloc(int w, int h) {
    ImageRGB img;
    img.w = w; img.h = h;
    img.R = (uint8_t*)malloc((size_t)w*h);
    img.G = (uint8_t*)malloc((size_t)w*h);
    img.B = (uint8_t*)malloc((size_t)w*h);
    if (!img.R || !img.G || !img.B) die("OOM RGB alloc.");
    return img;
}

static void rgb_free(ImageRGB *img) {
    free(img->R); free(img->G); free(img->B);
    img->R=img->G=img->B=NULL;
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
    bih.biHeight = h;          // bottom-up
    bih.biPlanes = 1;
    bih.biBitCount = 24;
    bih.biCompression = 0;
    bih.biSizeImage = imageSize;

    FILE *fp = fopen(path, "wb");
    if (!fp) die("Cannot write BMP.");
    fwrite(&bfh, sizeof(bfh), 1, fp);
    fwrite(&bih, sizeof(bih), 1, fp);

    uint8_t *row = (uint8_t*)malloc((size_t)rowSize);
    if (!row) die("OOM row.");

    for (int file_row = 0; file_row < h; file_row++) {
        int img_row = h - 1 - file_row;
        for (int x=0;x<w;x++) {
            size_t idx = (size_t)img_row*w + x;
            row[x*3+0] = img->B[idx];
            row[x*3+1] = img->G[idx];
            row[x*3+2] = img->R[idx];
        }
        for (int p=w*3;p<rowSize;p++) row[p]=0;
        fwrite(row, 1, (size_t)rowSize, fp);
    }
    free(row);
    fclose(fp);
}

static void dim_read(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "r");
    if (!f) die("Cannot read dim.txt");
    if (fscanf(f, "%d %d", w, h) != 2) die("Invalid dim.txt");
    fclose(f);
}

// ---------- Quantization tables (must match encoder) ----------
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

// ---------- ZigZag ----------
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

static void zigzag_unscan(const int16_t in[64], int16_t out[64]) {
    for (int i = 0; i < 64; i++) out[zigzag[i]] = in[i];
}

// ---------- DCT/IDCT ----------
static double c8(int u) { return (u==0)?(1.0/sqrt(2.0)):1.0; }

static void idct8x8(const double in[64], double out[64]) {
    for (int x=0;x<8;x++) {
        for (int y=0;y<8;y++) {
            double sum=0.0;
            for (int u=0;u<8;u++) {
                for (int v=0;v<8;v++) {
                    double F = in[v*8 + u];
                    sum += c8(u)*c8(v)*F
                         * cos(((2*x+1)*u*M_PI)/16.0)
                         * cos(((2*y+1)*v*M_PI)/16.0);
                }
            }
            out[y*8 + x] = 0.25 * sum;
        }
    }
}

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// ---------- YCbCr -> RGB ----------
static void ycbcr_to_rgb(const double *Y, const double *Cb, const double *Cr, int w, int h, ImageRGB *rgb) {
    for (int i=0;i<w*h;i++) {
        double y = Y[i];
        double cb = Cb[i] - 128.0;
        double cr = Cr[i] - 128.0;

        double R = y + 1.402 * cr;
        double G = y - 0.344136 * cb - 0.714136 * cr;
        double B = y + 1.772 * cb;

        int r = (int)llround(R);
        int g = (int)llround(G);
        int b = (int)llround(B);

        rgb->R[i] = clamp_u8(r);
        rgb->G[i] = clamp_u8(g);
        rgb->B[i] = clamp_u8(b);
    }
}

// ---------- padding info ----------
static void calc_pad8(int w, int h, int *W8, int *H8) {
    *W8 = (w+7)/8*8;
    *H8 = (h+7)/8*8;
}

// ---------- Method 0 ----------
static void decoder_method0(int argc, char *argv[]) {
    if (argc != 7) die("Usage: decoder 0 out.bmp R.txt G.txt B.txt dim.txt");
    const char *outbmp = argv[2];
    const char *Rtxt = argv[3];
    const char *Gtxt = argv[4];
    const char *Btxt = argv[5];
    const char *dimtxt=argv[6];

    int w,h;
    dim_read(dimtxt,&w,&h);

    ImageRGB img = rgb_alloc(w,h);

    FILE *fr=fopen(Rtxt,"r");
    FILE *fg=fopen(Gtxt,"r");
    FILE *fb=fopen(Btxt,"r");
    if (!fr||!fg||!fb) die("Cannot open R/G/B txt");

    for (int y=0;y<h;y++) {
        for (int x=0;x<w;x++) {
            unsigned rv,gv,bv;
            if (fscanf(fr,"%u",&rv)!=1) die("Read R failed");
            if (fscanf(fg,"%u",&gv)!=1) die("Read G failed");
            if (fscanf(fb,"%u",&bv)!=1) die("Read B failed");
            size_t idx=(size_t)y*w+x;
            img.R[idx]=(uint8_t)rv;
            img.G[idx]=(uint8_t)gv;
            img.B[idx]=(uint8_t)bv;
        }
    }
    fclose(fr); fclose(fg); fclose(fb);

    bmp_write_24(outbmp,&img);
    rgb_free(&img);
}

// ---------- Load qF raw ----------
static int16_t* load_qF_raw(const char *path, size_t n16) {
    FILE *f=fopen(path,"rb");
    if(!f) die("Cannot open qF raw");
    int16_t *buf=(int16_t*)malloc(n16*sizeof(int16_t));
    if(!buf) die("OOM qF load");
    if(fread(buf,sizeof(int16_t),n16,f)!=n16) die("Read qF raw failed");
    fclose(f);
    return buf;
}

static float* load_eF_raw(const char *path, size_t nf) {
    FILE *f=fopen(path,"rb");
    if(!f) die("Cannot open eF raw");
    float *buf=(float*)malloc(nf*sizeof(float));
    if(!buf) die("OOM eF load");
    if(fread(buf,sizeof(float),nf,f)!=nf) die("Read eF raw failed");
    fclose(f);
    return buf;
}

static void read_qtable_txt(const char *path, int Qt[64]) {
    FILE *f=fopen(path,"r");
    if(!f) die("Cannot read Qt txt");
    for(int r=0;r<8;r++){
        for(int c=0;c<8;c++){
            if(fscanf(f,"%d",&Qt[r*8+c])!=1) die("Invalid Qt txt");
        }
    }
    fclose(f);
}

// ---------- DC DPCM undo ----------
static void undo_dc_dpcm_inplace(int16_t *qF, size_t blocks) {
    int16_t prev=0;
    for(size_t b=0;b<blocks;b++){
        int16_t diff=qF[b*64+0];
        int16_t dc=(int16_t)(prev+diff);
        qF[b*64+0]=dc;
        prev=dc;
    }
}

typedef struct { uint8_t skip; int16_t val; } Pair;

static void rle_decode_to_coeff64(Pair *pairs, size_t nPairs, int16_t coeff64[64]) {
    int16_t zz[64]; for(int i=0;i<64;i++) zz[i]=0;
    int pos=0;
    for(size_t i=0;i<nPairs;i++){
        pos += pairs[i].skip;
        if(pos<0 || pos>=64) die("RLE pos OOR");
        zz[pos]=pairs[i].val;
        pos++;
    }
    zigzag_unscan(zz, coeff64);
}

// ---------- Reconstruct channel from qF (+ optional eF) ----------
static void reconstruct_channel_spatial(
    const int16_t *qF, const float *eF_or_null,
    int bx, int by, const int Qt[64],
    double *out_padded, int W8, int H8)
{
    (void)H8;
    size_t blocks=(size_t)bx*by;
    for(size_t b=0;b<blocks;b++){
        double F[64];
        double blk[64];

        for(int k=0;k<64;k++){
            double recon = (double)qF[b*64+k] * (double)Qt[k];
            if(eF_or_null) recon += (double)eF_or_null[b*64+k];
            F[k]=recon;
        }

        idct8x8(F, blk);

        int m = (int)(b / (size_t)bx);
        int n = (int)(b % (size_t)bx);

        for(int yy=0;yy<8;yy++){
            for(int xx=0;xx<8;xx++){
                int x=n*8+xx;
                int y=m*8+yy;
                double s = blk[yy*8+xx] + 128.0;
                out_padded[y*W8 + x] = s;
            }
        }
    }
}

// ---------- Method 1 decoder ----------
static void decoder_method1a(int argc, char *argv[]) {
    // decoder 1 QRes.bmp Kimberly.bmp Qt_Y.txt Qt_Cb.txt Qt_Cr.txt dim.txt qF_Y.raw qF_Cb.raw qF_Cr.raw
    if(argc!=10) die("Usage: decoder 1 QRes.bmp orig.bmp QtY QtCb QtCr dim qFY qFCb qFCr");
    const char *outbmp=argv[2];
    const char *origbmp=argv[3];
    const char *QtYtxt=argv[4];
    const char *QtCbtxt=argv[5];
    const char *QtCrtxt=argv[6];
    const char *dimtxt=argv[7];
    const char *qFY=argv[8];
    const char *qFCb=argv[9];
    const char *qFCr=argv[10]; // (won't exist) - keep consistent? actually argc=10 -> last is argv[9]
}

static void decoder_method1(int argc, char *argv[]) {
    // two variants:
    // (a) decoder 1 QRes.bmp Kimberly.bmp QtY QtCb QtCr dim qFY qFCb qFCr
    // (b) decoder 1 Res.bmp QtY QtCb QtCr dim qFY qFCb qFCr eFY eFCb eFCr
    if (argc == 11) {
        // (a)
        const char *outbmp = argv[2];
        const char *origbmp= argv[3];
        const char *QtYtxt = argv[4];
        const char *QtCbtxt= argv[5];
        const char *QtCrtxt= argv[6];
        const char *dimtxt = argv[7];
        const char *qFYraw = argv[8];
        const char *qFCbraw= argv[9];
        const char *qFCrraw= argv[10];

        int w,h; dim_read(dimtxt,&w,&h);
        int W8,H8; calc_pad8(w,h,&W8,&H8);
        int bx=W8/8, by=H8/8;
        size_t blocks=(size_t)bx*by;
        size_t n16=blocks*64;

        int QtY[64],QtCb[64],QtCr[64];
        read_qtable_txt(QtYtxt,QtY);
        read_qtable_txt(QtCbtxt,QtCb);
        read_qtable_txt(QtCrtxt,QtCr);

        int16_t *qY=load_qF_raw(qFYraw,n16);
        int16_t *qCb=load_qF_raw(qFCbraw,n16);
        int16_t *qCr=load_qF_raw(qFCrraw,n16);

        double *Yp=(double*)malloc((size_t)W8*H8*sizeof(double));
        double *Cbp=(double*)malloc((size_t)W8*H8*sizeof(double));
        double *Crp=(double*)malloc((size_t)W8*H8*sizeof(double));
        if(!Yp||!Cbp||!Crp) die("OOM recon buffers.");

        reconstruct_channel_spatial(qY,NULL,bx,by,QtY,Yp,W8,H8);
        reconstruct_channel_spatial(qCb,NULL,bx,by,QtCb,Cbp,W8,H8);
        reconstruct_channel_spatial(qCr,NULL,bx,by,QtCr,Crp,W8,H8);

        // crop to original size
        double *Y=(double*)malloc((size_t)w*h*sizeof(double));
        double *Cb=(double*)malloc((size_t)w*h*sizeof(double));
        double *Cr=(double*)malloc((size_t)w*h*sizeof(double));
        if(!Y||!Cb||!Cr) die("OOM crop.");
        for(int yy=0;yy<h;yy++){
            for(int xx=0;xx<w;xx++){
                Y[yy*w+xx]=Yp[yy*W8+xx];
                Cb[yy*w+xx]=Cbp[yy*W8+xx];
                Cr[yy*w+xx]=Crp[yy*W8+xx];
            }
        }

        ImageRGB out = rgb_alloc(w,h);
        ycbcr_to_rgb(Y,Cb,Cr,w,h,&out);
        bmp_write_24(outbmp,&out);

        // pixel-domain SQNR comparing with origbmp
        // (read original)
        // For simplicity: read original as BMP 24-bit using local quick reader (minimal)
        // We'll implement a minimal BMP reader here:
        FILE *fp=fopen(origbmp,"rb");
        if(!fp) die("Cannot open orig bmp");
        BITMAPFILEHEADER bfh; BITMAPINFOHEADER bih;
        fread(&bfh,sizeof(bfh),1,fp);
        fread(&bih,sizeof(bih),1,fp);
        if(bfh.bfType!=0x4D42||bih.biBitCount!=24||bih.biCompression!=0) die("orig bmp not supported");
        int ow=bih.biWidth;
        int oh=(bih.biHeight>=0)?bih.biHeight:-bih.biHeight;
        int bottom_up=(bih.biHeight>=0);
        if(ow!=w||oh!=h) die("orig size mismatch");
        int rowSize=row_size_bytes_24(ow);
        fseek(fp,(long)bfh.bfOffBits,SEEK_SET);
        uint8_t *row=(uint8_t*)malloc((size_t)rowSize);
        if(!row) die("OOM row");
        double sig[3]={0,0,0}, err[3]={0,0,0};
        for(int file_row=0;file_row<oh;file_row++){
            fread(row,1,(size_t)rowSize,fp);
            int img_row=bottom_up?(oh-1-file_row):file_row;
            for(int x=0;x<ow;x++){
                size_t idx=(size_t)img_row*ow+x;
                uint8_t ob=row[x*3+2], og=row[x*3+1], obb=row[x*3+0];
                double dr=(double)ob - (double)out.R[idx];
                double dg=(double)og - (double)out.G[idx];
                double db=(double)obb - (double)out.B[idx];
                sig[0]+= (double)ob*(double)ob; err[0]+= dr*dr;
                sig[1]+= (double)og*(double)og; err[1]+= dg*dg;
                sig[2]+= (double)obb*(double)obb; err[2]+= db*db;
            }
        }
        free(row); fclose(fp);
        for(int c=0;c<3;c++){
            double sqnr = (err[c]<=0.0)?999.0:(10.0*log10(sig[c]/err[c]));
            printf("%8.3f%s", sqnr, (c==2)?"":" ");
        }
        printf("\n");

        // cleanup
        free(qY); free(qCb); free(qCr);
        free(Yp); free(Cbp); free(Crp);
        free(Y); free(Cb); free(Cr);
        rgb_free(&out);
    }
    else if (argc == 12) {
        // (b) decoder 1 Res.bmp QtY QtCb QtCr dim qFY qFCb qFCr eFY eFCb eFCr
        const char *outbmp = argv[2];
        const char *QtYtxt = argv[3];
        const char *QtCbtxt= argv[4];
        const char *QtCrtxt= argv[5];
        const char *dimtxt = argv[6];
        const char *qFYraw = argv[7];
        const char *qFCbraw= argv[8];
        const char *qFCrraw= argv[9];
        const char *eFYraw = argv[10];
        const char *eFCbraw= argv[11];
        const char *eFCrraw= argv[12]; // out of range; fixed below
        die("Method 1(b) argc mismatch. Expected 12 args after program name.");
    } else if (argc == 13) {
        // correct (b): program + 12 args => argc=13
        const char *outbmp = argv[2];
        const char *QtYtxt = argv[3];
        const char *QtCbtxt= argv[4];
        const char *QtCrtxt= argv[5];
        const char *dimtxt = argv[6];
        const char *qFYraw = argv[7];
        const char *qFCbraw= argv[8];
        const char *qFCrraw= argv[9];
        const char *eFYraw = argv[10];
        const char *eFCbraw= argv[11];
        const char *eFCrraw= argv[12];

        int w,h; dim_read(dimtxt,&w,&h);
        int W8,H8; calc_pad8(w,h,&W8,&H8);
        int bx=W8/8, by=H8/8;
        size_t blocks=(size_t)bx*by;
        size_t n16=blocks*64;

        int QtY[64],QtCb[64],QtCr[64];
        read_qtable_txt(QtYtxt,QtY);
        read_qtable_txt(QtCbtxt,QtCb);
        read_qtable_txt(QtCrtxt,QtCr);

        int16_t *qY=load_qF_raw(qFYraw,n16);
        int16_t *qCb=load_qF_raw(qFCbraw,n16);
        int16_t *qCr=load_qF_raw(qFCrraw,n16);

        float *eY=load_eF_raw(eFYraw,n16);
        float *eCb=load_eF_raw(eFCbraw,n16);
        float *eCr=load_eF_raw(eFCrraw,n16);

        double *Yp=(double*)malloc((size_t)W8*H8*sizeof(double));
        double *Cbp=(double*)malloc((size_t)W8*H8*sizeof(double));
        double *Crp=(double*)malloc((size_t)W8*H8*sizeof(double));
        if(!Yp||!Cbp||!Crp) die("OOM recon buffers.");

        reconstruct_channel_spatial(qY,eY,bx,by,QtY,Yp,W8,H8);
        reconstruct_channel_spatial(qCb,eCb,bx,by,QtCb,Cbp,W8,H8);
        reconstruct_channel_spatial(qCr,eCr,bx,by,QtCr,Crp,W8,H8);

        double *Y=(double*)malloc((size_t)w*h*sizeof(double));
        double *Cb=(double*)malloc((size_t)w*h*sizeof(double));
        double *Cr=(double*)malloc((size_t)w*h*sizeof(double));
        if(!Y||!Cb||!Cr) die("OOM crop.");
        for(int yy=0;yy<h;yy++){
            for(int xx=0;xx<w;xx++){
                Y[yy*w+xx]=Yp[yy*W8+xx];
                Cb[yy*w+xx]=Cbp[yy*W8+xx];
                Cr[yy*w+xx]=Crp[yy*W8+xx];
            }
        }

        ImageRGB out = rgb_alloc(w,h);
        ycbcr_to_rgb(Y,Cb,Cr,w,h,&out);
        bmp_write_24(outbmp,&out);

        // cleanup
        free(qY); free(qCb); free(qCr);
        free(eY); free(eCb); free(eCr);
        free(Yp); free(Cbp); free(Crp);
        free(Y); free(Cb); free(Cr);
        rgb_free(&out);
    } else {
        die("decoder 1: invalid argc for (a) or (b).");
    }
}

// ---------- Method 2 decoder ----------
static void decoder_method2(int argc, char *argv[]) {
    // decoder 2 QRes.bmp ascii rle_code.txt
    // decoder 2 QRes.bmp binary rle_code.bin
    if (argc != 5) die("Usage: decoder 2 out.bmp ascii|binary rle_code.(txt|bin)");
    const char *outbmp = argv[2];
    const char *fmt = argv[3];
    const char *in  = argv[4];

    int w=0,h=0,bx=0,by=0;
    int W8,H8;

    int16_t *qY=NULL,*qCb=NULL,*qCr=NULL;

    if (strcmp(fmt,"ascii")==0) {
        FILE *f=fopen(in,"r");
        if(!f) die("Cannot open rle_code.txt");
        if(fscanf(f,"%d %d",&w,&h)!=2) die("Bad first line in rle txt");
        calc_pad8(w,h,&W8,&H8);
        bx=W8/8; by=H8/8;
        size_t blocks=(size_t)bx*by;
        size_t n16=blocks*64;
        qY=(int16_t*)calloc(n16,sizeof(int16_t));
        qCb=(int16_t*)calloc(n16,sizeof(int16_t));
        qCr=(int16_t*)calloc(n16,sizeof(int16_t));
        if(!qY||!qCb||!qCr) die("OOM q arrays");

        // parse each line: ($m,$n, X) skip val skip val ...
        for(int m=0;m<by;m++){
            for(int n=0;n<bx;n++){
                for(int ch=0;ch<3;ch++){
                    // read header tokens crudely
                    char buf[64];
                    if(fscanf(f,"%63s",buf)!=1) die("Unexpected EOF in rle txt");
                    // buf should be like "($m,$n," (may include comma)
                    // We'll ignore strict parsing, then read channel token
                    char chan[8];
                    if(fscanf(f,"%7s",chan)!=1) die("Bad chan");
                    // chan like "Y)" or "Cb)" etc
                    // Now read rest of line: pairs until newline
                    Pair pairs[512]; size_t np=0;
                    int c;
                    // read pairs with fscanf until endline
                    while (1) {
                        // peek next char
                        c = fgetc(f);
                        if (c == '\n' || c == EOF) break;
                        ungetc(c,f);
                        unsigned sk; int vv;
                        if (fscanf(f,"%u %d",&sk,&vv)!=2) {
                            // consume line
                            while((c=fgetc(f))!='\n' && c!=EOF) {}
                            break;
                        }
                        pairs[np].skip=(uint8_t)sk;
                        pairs[np].val=(int16_t)vv;
                        np++;
                        if(np>=512) die("Too many pairs in a block line");
                    }
                    int16_t coeff[64];
                    rle_decode_to_coeff64(pairs,np,coeff);

                    size_t b=(size_t)m*bx+n;
                    int16_t *dst = (ch==0)?qY:((ch==1)?qCb:qCr);
                    memcpy(&dst[b*64], coeff, 64*sizeof(int16_t));
                }
            }
        }
        fclose(f);

        // undo DPCM
        blocks = (size_t)bx * by;
        undo_dc_dpcm_inplace(qY,blocks);
        undo_dc_dpcm_inplace(qCb,blocks);
        undo_dc_dpcm_inplace(qCr,blocks);

    } else if (strcmp(fmt,"binary")==0) {
        FILE *f=fopen(in,"rb");
        if(!f) die("Cannot open rle_code.bin");
        uint32_t magic;
        fread(&magic,4,1,f);
        if(magic!=0x32454C52) die("Bad rle bin magic");
        fread(&w,4,1,f);
        fread(&h,4,1,f);
        fread(&bx,4,1,f);
        fread(&by,4,1,f);
        calc_pad8(w,h,&W8,&H8);

        size_t blocks=(size_t)bx*by;
        size_t n16=blocks*64;
        qY=(int16_t*)calloc(n16,sizeof(int16_t));
        qCb=(int16_t*)calloc(n16,sizeof(int16_t));
        qCr=(int16_t*)calloc(n16,sizeof(int16_t));
        if(!qY||!qCb||!qCr) die("OOM q arrays");

        for(size_t b=0;b<blocks;b++){
            for(int ch=0;ch<3;ch++){
                uint16_t np16;
                fread(&np16,2,1,f);
                Pair pairs[512];
                if(np16>512) die("np too big in bin");
                for(uint16_t i=0;i<np16;i++){
                    fread(&pairs[i].skip,1,1,f);
                    fread(&pairs[i].val,2,1,f);
                }
                int16_t coeff[64];
                rle_decode_to_coeff64(pairs,np16,coeff);
                int16_t *dst=(ch==0)?qY:((ch==1)?qCb:qCr);
                memcpy(&dst[b*64], coeff, 64*sizeof(int16_t));
            }
        }
        fclose(f);

        // undo DPCM
        undo_dc_dpcm_inplace(qY,blocks);
        undo_dc_dpcm_inplace(qCb,blocks);
        undo_dc_dpcm_inplace(qCr,blocks);

    } else die("fmt must be ascii|binary");

    // Now reconstruct like decoder1(a) using standard Qt
    double *Yp=(double*)malloc((size_t)W8*H8*sizeof(double));
    double *Cbp=(double*)malloc((size_t)W8*H8*sizeof(double));
    double *Crp=(double*)malloc((size_t)W8*H8*sizeof(double));
    if(!Yp||!Cbp||!Crp) die("OOM recon");

    // reuse same recon function as before but inline
    // recon F = qF*Qt, then IDCT + 128
    {
        size_t blocks=(size_t)bx*by;
        for(size_t b=0;b<blocks;b++){
            double F[64],blk[64];
            for(int k=0;k<64;k++) F[k]=(double)qY[b*64+k]*(double)QTY_std[k];
            idct8x8(F,blk);
            int m=(int)(b/(size_t)bx), n=(int)(b%(size_t)bx);
            for(int yy=0;yy<8;yy++)for(int xx=0;xx<8;xx++){
                Yp[(m*8+yy)*W8 + (n*8+xx)] = blk[yy*8+xx] + 128.0;
            }
        }
        for(size_t b=0;b<blocks;b++){
            double F[64],blk[64];
            for(int k=0;k<64;k++) F[k]=(double)qCb[b*64+k]*(double)QTC_std[k];
            idct8x8(F,blk);
            int m=(int)(b/(size_t)bx), n=(int)(b%(size_t)bx);
            for(int yy=0;yy<8;yy++)for(int xx=0;xx<8;xx++){
                Cbp[(m*8+yy)*W8 + (n*8+xx)] = blk[yy*8+xx] + 128.0;
            }
        }
        for(size_t b=0;b<blocks;b++){
            double F[64],blk[64];
            for(int k=0;k<64;k++) F[k]=(double)qCr[b*64+k]*(double)QTC_std[k];
            idct8x8(F,blk);
            int m=(int)(b/(size_t)bx), n=(int)(b%(size_t)bx);
            for(int yy=0;yy<8;yy++)for(int xx=0;xx<8;xx++){
                Crp[(m*8+yy)*W8 + (n*8+xx)] = blk[yy*8+xx] + 128.0;
            }
        }
    }

    double *Y=(double*)malloc((size_t)w*h*sizeof(double));
    double *Cb=(double*)malloc((size_t)w*h*sizeof(double));
    double *Cr=(double*)malloc((size_t)w*h*sizeof(double));
    if(!Y||!Cb||!Cr) die("OOM crop");
    for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
        Y[yy*w+xx]=Yp[yy*W8+xx];
        Cb[yy*w+xx]=Cbp[yy*W8+xx];
        Cr[yy*w+xx]=Crp[yy*W8+xx];
    }

    ImageRGB outimg = rgb_alloc(w,h);
    ycbcr_to_rgb(Y,Cb,Cr,w,h,&outimg);
    bmp_write_24(outbmp,&outimg);

    free(qY); free(qCb); free(qCr);
    free(Yp); free(Cbp); free(Crp);
    free(Y); free(Cb); free(Cr);
    rgb_free(&outimg);
}

// ---------- Method 3 decoder (Huffman) ----------
typedef struct DNode {
    int isLeaf;
    int32_t sym;
    struct DNode *z,*o;
} DNode;

static DNode* dn_new() {
    DNode* n=(DNode*)calloc(1,sizeof(DNode));
    if(!n) die("OOM DNode");
    return n;
}

static void dn_free(DNode *n){
    if(!n) return;
    dn_free(n->z);
    dn_free(n->o);
    free(n);
}

static void trie_insert(DNode *root, int32_t sym, const char *bits) {
    DNode *cur=root;
    for(size_t i=0;i<strlen(bits);i++){
        if(bits[i]=='0'){
            if(!cur->z) cur->z=dn_new();
            cur=cur->z;
        } else if(bits[i]=='1'){
            if(!cur->o) cur->o=dn_new();
            cur=cur->o;
        } else die("Invalid bit char in codeword");
    }
    cur->isLeaf=1;
    cur->sym=sym;
}

static int32_t sym_EOB(void) { return (int32_t)0x7FFFFFFF; }

static void unpack_symbol(int32_t sym, uint8_t *skip, int16_t *val) {
    *skip = (uint8_t)((uint32_t)sym >> 16);
    *val  = (int16_t)(uint16_t)((uint32_t)sym & 0xFFFFu);
}

static void decoder_method3(int argc, char *argv[]) {
    // decoder 3 QRes.bmp ascii codebook.txt huffman_code.txt
    // decoder 3 QRes.bmp binary codebook.txt huffman_code.bin
    if(argc!=6) die("Usage: decoder 3 out.bmp ascii|binary codebook.txt huffman_code.(txt|bin)");

    const char *outbmp=argv[2];
    const char *fmt=argv[3];
    const char *codebook=argv[4];
    const char *in=argv[5];

    // load codebook to trie
    FILE *fc=fopen(codebook,"r");
    if(!fc) die("Cannot open codebook.txt");
    char line[2048];
    // skip header
    fgets(line,sizeof(line),fc);

    DNode *root=dn_new();

    while(fgets(line,sizeof(line),fc)){
        // symbol count codeword
        int sym=0;
        unsigned long long cnt=0;
        char bits[1024]={0};
        if(sscanf(line,"%d%llu%1023s",&sym,&cnt,bits)==3){
            (void)cnt;
            trie_insert(root,(int32_t)sym,bits);
        }
    }
    fclose(fc);

    int w=0,h=0;
    uint8_t *bitbytes=NULL;
    size_t nbytes=0;

    if(strcmp(fmt,"ascii")==0){
        FILE *f=fopen(in,"r");
        if(!f) die("Cannot open huffman_code.txt");
        // size line: "size: w h"
        if(!fgets(line,sizeof(line),f)) die("Bad huffman_code.txt");
        if(sscanf(line,"size: %d %d",&w,&h)!=2) die("Bad size line");
        // bitstream line
        if(!fgets(line,sizeof(line),f)) die("Bad huffman_code.txt");
        // extract after "bitstream: "
        char *p=strstr(line,"bitstream:");
        if(!p) die("Bad bitstream line");
        p += strlen("bitstream:");
        while(*p==' ') p++;
        size_t nbits=strlen(p);
        while(nbits>0 && (p[nbits-1]=='\n' || p[nbits-1]=='\r')) nbits--;
        nbytes=(nbits+7)/8;
        bitbytes=(uint8_t*)calloc(nbytes,1);
        if(!bitbytes) die("OOM bitbytes");
        for(size_t i=0;i<nbits;i++){
            int bit=(p[i]=='1')?1:0;
            size_t bi=i/8;
            int bp=(int)(i%8);
            if(bit) bitbytes[bi] |= (uint8_t)(1u<<(7-bp));
        }
        fclose(f);
    } else if(strcmp(fmt,"binary")==0){
        FILE *f=fopen(in,"rb");
        if(!f) die("Cannot open huffman_code.bin");
        uint32_t magic; fread(&magic,4,1,f);
        if(magic!=0x33465548) die("Bad HUF3 magic");
        fread(&w,4,1,f);
        fread(&h,4,1,f);
        uint32_t bitlen; fread(&bitlen,4,1,f);
        nbytes=(bitlen+7)/8;
        bitbytes=(uint8_t*)malloc(nbytes);
        if(!bitbytes) die("OOM bitbytes");
        if(fread(bitbytes,1,nbytes,f)!=nbytes) die("Read bitbytes failed");
        fclose(f);
    } else die("fmt must be ascii|binary");

    int W8,H8; calc_pad8(w,h,&W8,&H8);
    int bx=W8/8, by=H8/8;
    size_t blocks=(size_t)bx*by;
    size_t n16=blocks*64;

    int16_t *qY=(int16_t*)calloc(n16,sizeof(int16_t));
    int16_t *qCb=(int16_t*)calloc(n16,sizeof(int16_t));
    int16_t *qCr=(int16_t*)calloc(n16,sizeof(int16_t));
    if(!qY||!qCb||!qCr) die("OOM q arrays");

    // decode symbols sequentially -> rebuild each block per channel
    size_t bit_i=0;
    for(size_t b=0;b<blocks;b++){
        for(int ch=0;ch<3;ch++){
            Pair pairs[512]; size_t np=0;
            // read until EOB
            while(1){
                // walk trie
                DNode *cur=root;
                while(!cur->isLeaf){
                    if(bit_i/8 >= nbytes) die("Bitstream EOF");
                    int bit = (bitbytes[bit_i/8] >> (7-(bit_i%8))) & 1;
                    bit_i++;
                    cur = bit ? cur->o : cur->z;
                    if(!cur) die("Invalid code in bitstream");
                }
                int32_t sym=cur->sym;
                if(sym==sym_EOB()) break;
                if(np>=512) die("Too many pairs in a block");
                uint8_t sk; int16_t val;
                unpack_symbol(sym,&sk,&val);
                pairs[np].skip=sk;
                pairs[np].val=val;
                np++;
            }
            int16_t coeff[64];
            rle_decode_to_coeff64(pairs,np,coeff);
            int16_t *dst=(ch==0)?qY:((ch==1)?qCb:qCr);
            memcpy(&dst[b*64], coeff, 64*sizeof(int16_t));
        }
    }

    // undo DC DPCM
    undo_dc_dpcm_inplace(qY,blocks);
    undo_dc_dpcm_inplace(qCb,blocks);
    undo_dc_dpcm_inplace(qCr,blocks);

    // reconstruct to BMP (same as method2)
    double *Yp=(double*)malloc((size_t)W8*H8*sizeof(double));
    double *Cbp=(double*)malloc((size_t)W8*H8*sizeof(double));
    double *Crp=(double*)malloc((size_t)W8*H8*sizeof(double));
    if(!Yp||!Cbp||!Crp) die("OOM recon");

    for(size_t b=0;b<blocks;b++){
        double F[64],blk[64];
        for(int k=0;k<64;k++) F[k]=(double)qY[b*64+k]*(double)QTY_std[k];
        idct8x8(F,blk);
        int m=(int)(b/(size_t)bx), n=(int)(b%(size_t)bx);
        for(int yy=0;yy<8;yy++)for(int xx=0;xx<8;xx++){
            Yp[(m*8+yy)*W8 + (n*8+xx)] = blk[yy*8+xx] + 128.0;
        }
    }
    for(size_t b=0;b<blocks;b++){
        double F[64],blk[64];
        for(int k=0;k<64;k++) F[k]=(double)qCb[b*64+k]*(double)QTC_std[k];
        idct8x8(F,blk);
        int m=(int)(b/(size_t)bx), n=(int)(b%(size_t)bx);
        for(int yy=0;yy<8;yy++)for(int xx=0;xx<8;xx++){
            Cbp[(m*8+yy)*W8 + (n*8+xx)] = blk[yy*8+xx] + 128.0;
        }
    }
    for(size_t b=0;b<blocks;b++){
        double F[64],blk[64];
        for(int k=0;k<64;k++) F[k]=(double)qCr[b*64+k]*(double)QTC_std[k];
        idct8x8(F,blk);
        int m=(int)(b/(size_t)bx), n=(int)(b%(size_t)bx);
        for(int yy=0;yy<8;yy++)for(int xx=0;xx<8;xx++){
            Crp[(m*8+yy)*W8 + (n*8+xx)] = blk[yy*8+xx] + 128.0;
        }
    }

    double *Y=(double*)malloc((size_t)w*h*sizeof(double));
    double *Cb=(double*)malloc((size_t)w*h*sizeof(double));
    double *Cr=(double*)malloc((size_t)w*h*sizeof(double));
    if(!Y||!Cb||!Cr) die("OOM crop");
    for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){
        Y[yy*w+xx]=Yp[yy*W8+xx];
        Cb[yy*w+xx]=Cbp[yy*W8+xx];
        Cr[yy*w+xx]=Crp[yy*W8+xx];
    }

    ImageRGB outimg = rgb_alloc(w,h);
    ycbcr_to_rgb(Y,Cb,Cr,w,h,&outimg);
    bmp_write_24(outbmp,&outimg);

    // cleanup
    dn_free(root);
    free(bitbytes);
    free(qY); free(qCb); free(qCr);
    free(Yp); free(Cbp); free(Crp);
    free(Y); free(Cb); free(Cr);
    rgb_free(&outimg);
}

int main(int argc, char *argv[]) {
    if(argc<2) die("Usage: decoder <0|1|2|3> ...");
    int mode = atoi(argv[1]);
    switch(mode){
        case 0: decoder_method0(argc,argv); break;
        case 1: decoder_method1(argc,argv); break;
        case 2: decoder_method2(argc,argv); break;
        case 3: decoder_method3(argc,argv); break;
        default: die("Invalid mode (0~3).");
    }
    return 0;
}

