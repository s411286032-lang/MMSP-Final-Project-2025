#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#pragma pack(push, 1)
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
        printf("Usage: encoder 0 input.bmp R.txt G.txt B.txt dim.txt\n");
        return 1;
    }

    FILE *fp = fopen(argv[2], "rb");
    if (!fp) {
        printf("Cannot open input BMP\n");
        return 1;
    }

    BMPHeader bmp;
    DIBHeader dib;
    fread(&bmp, sizeof(BMPHeader), 1, fp);
    fread(&dib, sizeof(DIBHeader), 1, fp);

    int width  = dib.biWidth;
    int height = dib.biHeight;

    int row_bytes = width * 3;
    int padding = (4 - (row_bytes % 4)) % 4;

    /* allocate RGB buffers (top-down order) */
    uint8_t **R = (uint8_t **)malloc(height * sizeof(uint8_t *));
    uint8_t **G = (uint8_t **)malloc(height * sizeof(uint8_t *));
    uint8_t **B = (uint8_t **)malloc(height * sizeof(uint8_t *));
    for (int i = 0; i < height; i++) {
        R[i] = (uint8_t *)malloc(width);
        G[i] = (uint8_t *)malloc(width);
        B[i] = (uint8_t *)malloc(width);
    }

    uint8_t *row = (uint8_t *)malloc(row_bytes);

    /* read BMP pixel data (bottom-up) and flip to top-down */
    for (int i = 0; i < height; i++) {
        fread(row, 1, row_bytes, fp);
        fseek(fp, padding, SEEK_CUR);

        int dst_row = height - 1 - i;  // ⭐ 關鍵修正點

        for (int j = 0; j < width; j++) {
            B[dst_row][j] = row[j * 3 + 0];
            G[dst_row][j] = row[j * 3 + 1];
            R[dst_row][j] = row[j * 3 + 2];
        }
    }

    fclose(fp);
    free(row);

    FILE *fR = fopen(argv[3], "w");
    FILE *fG = fopen(argv[4], "w");
    FILE *fB = fopen(argv[5], "w");
    FILE *fD = fopen(argv[6], "w");

    /* write dimension */
    fprintf(fD, "%d %d\n", width, height);

    /* write RGB txt (top-down) */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(fR, "%d ", R[i][j]);
            fprintf(fG, "%d ", G[i][j]);
            fprintf(fB, "%d ", B[i][j]);
        }
        fprintf(fR, "\n");
        fprintf(fG, "\n");
        fprintf(fB, "\n");
    }

    fclose(fR);
    fclose(fG);
    fclose(fB);
    fclose(fD);

    for (int i = 0; i < height; i++) {
        free(R[i]);
        free(G[i]);
        free(B[i]);
    }
    free(R);
    free(G);
    free(B);

    return 0;
}
