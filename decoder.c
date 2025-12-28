#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#pragma pack(push,1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BMPHeader;

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
} DIBHeader;
#pragma pack(pop)

int main(int argc, char *argv[]) {
    if (argc != 7 || atoi(argv[1]) != 0) {
        printf("Usage: decoder 0 output.bmp R.txt G.txt B.txt dim.txt\n");
        return 1;
    }

    int width, height;
    FILE *fD = fopen(argv[6], "r");
    fscanf(fD, "%d %d", &width, &height);
    fclose(fD);

    FILE *fR = fopen(argv[3], "r");
    FILE *fG = fopen(argv[4], "r");
    FILE *fB = fopen(argv[5], "r");

    int row_bytes = width * 3;
    int padding = (4 - (row_bytes % 4)) % 4;

    BMPHeader bmp = {0};
    DIBHeader dib = {0};

    bmp.bfType = 0x4D42;
    bmp.bfOffBits = sizeof(BMPHeader) + sizeof(DIBHeader);
    bmp.bfSize = bmp.bfOffBits + (row_bytes + padding) * height;

    dib.biSize = sizeof(DIBHeader);
    dib.biWidth = width;
    dib.biHeight = height;   // bottom-up
    dib.biPlanes = 1;
    dib.biBitCount = 24;
    dib.biCompression = 0;
    dib.biSizeImage = (row_bytes + padding) * height;

    FILE *fp = fopen(argv[2], "wb");
    fwrite(&bmp, sizeof(bmp), 1, fp);
    fwrite(&dib, sizeof(dib), 1, fp);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int R, G, B;
            fscanf(fR, "%d", &R);
            fscanf(fG, "%d", &G);
            fscanf(fB, "%d", &B);

            fputc(B, fp);
            fputc(G, fp);
            fputc(R, fp);
        }
        for (int p = 0; p < padding; p++)
            fputc(0, fp);
    }

    fclose(fp);
    fclose(fR); fclose(fG); fclose(fB);
    return 0;
}
