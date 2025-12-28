#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/*
 * 程式流程：
 * 1. 讀取 24-bit BMP 檔案（BI_RGB，未壓縮）
 * 2. 解析 BMP Header 與 DIB Header
 * 3. 正確處理 BMP 的 bottom-up pixel 順序與 row padding
 * 4. 將 BGR pixel 資料轉換並分離為 R / G / B 三個 channel
 * 5. 將 RGB channel 以 ASCII 格式輸出成文字檔
 */

#pragma pack(push, 1)
//BMP 檔案標頭（14 bytes） 
typedef struct {
    uint16_t bfType;      // 檔案識別碼，應為 'BM'
    uint32_t bfSize;      // 整個 BMP 檔案大小
    uint16_t bfReserved1; // 保留欄位
    uint16_t bfReserved2; // 保留欄位
    uint32_t bfOffBits;   // Pixel data 在檔案中的起始位置
} BMPHeader;

//DIB 標頭（BITMAPINFOHEADER，40 bytes） 
typedef struct {
    uint32_t biSize;          // DIB header 大小
    int32_t  biWidth;         // 影像寬度（pixel）
    int32_t  biHeight;        // 影像高度（pixel，正值表示 bottom-up）
    uint16_t biPlanes;        // 固定為 1
    uint16_t biBitCount;      // 每像素位元數（24 表示 24-bit BMP）
    uint32_t biCompression;   // 壓縮方式（0 = BI_RGB）
    uint32_t biSizeImage;     // Pixel data 大小
    int32_t  biXPelsPerMeter; // 水平解析度
    int32_t  biYPelsPerMeter; // 垂直解析度
    uint32_t biClrUsed;       // 使用顏色數
    uint32_t biClrImportant;  // 重要顏色數
} DIBHeader;
#pragma pack(pop)

int main(int argc, char *argv[]) {

    //檢查指令格式，只支援 Method 0 
    if (argc != 7 || atoi(argv[1]) != 0) {
        printf("Usage: encoder 0 input.bmp R.txt G.txt B.txt dim.txt\n");
        return 1;
    }

    //以二進位模式開啟輸入 BMP 檔案 
    FILE *fp = fopen(argv[2], "rb");
    if (!fp) {
        printf("Cannot open input BMP\n");
        return 1;
    }

    //讀取 BMP Header 與 DIB Header 
    BMPHeader bmp;
    DIBHeader dib;
    fread(&bmp, sizeof(BMPHeader), 1, fp);
    fread(&dib, sizeof(DIBHeader), 1, fp);

    //取得影像寬度與高度 
    int width  = dib.biWidth;
    int height = dib.biHeight;

    //24-bit BMP 中，每個 pixel 佔 3 bytes（B, G, R）。
    //BMP 規定每一列 scanline 必須對齊至 4-byte boundary，
    //因此需要計算每列所需的 padding bytes。
    int row_bytes = width * 3;
    int padding = (4 - (row_bytes % 4)) % 4;

    //配置 RGB 三個 channel 的記憶體空間。
    //這裡使用 top-down 的儲存方式，方便後續處理。
    uint8_t **R = (uint8_t **)malloc(height * sizeof(uint8_t *));
    uint8_t **G = (uint8_t **)malloc(height * sizeof(uint8_t *));
    uint8_t **B = (uint8_t **)malloc(height * sizeof(uint8_t *));
    for (int i = 0; i < height; i++) {
        R[i] = (uint8_t *)malloc(width);
        G[i] = (uint8_t *)malloc(width);
        B[i] = (uint8_t *)malloc(width);
    }

    //暫存單一列 pixel 資料（不包含 padding)
    uint8_t *row = (uint8_t *)malloc(row_bytes);

    //BMP pixel data 在檔案中是以 bottom-up 方式儲存，
    //因此在讀取時需將 row 順序反轉，轉為 top-down。
    for (int i = 0; i < height; i++) {
        fread(row, 1, row_bytes, fp);      // 讀取一列 pixel data
        fseek(fp, padding, SEEK_CUR);      // 跳過 padding bytes

        int dst_row = height - 1 - i;      // bottom-up → top-down

        for (int j = 0; j < width; j++) {
            B[dst_row][j] = row[j * 3 + 0];
            G[dst_row][j] = row[j * 3 + 1];
            R[dst_row][j] = row[j * 3 + 2];
        }
    }

    fclose(fp);
    free(row);

    //開啟輸出檔案（ASCII 格式） 
    FILE *fR = fopen(argv[3], "w");
    FILE *fG = fopen(argv[4], "w");
    FILE *fB = fopen(argv[5], "w");
    FILE *fD = fopen(argv[6], "w");

    //輸出影像尺寸到 dim.txt
    fprintf(fD, "%d %d\n", width, height);

    //依照 top-down 順序，將 RGB channel
    //逐列輸出為 ASCII 文字檔
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

    //釋放 RGB 記憶體空間
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
