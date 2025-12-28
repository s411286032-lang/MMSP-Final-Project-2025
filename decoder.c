#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/*
 * Method 0 驗證
 * 1. RGB channel 資料是否正確儲存
 * 2. BMP 檔案 header 與 pixel data 是否能正確重建
 */

#pragma pack(push,1)
//BMP 檔案標頭（14 bytes） 
typedef struct {
    uint16_t bfType;      // 檔案識別碼，應為 'BM'
    uint32_t bfSize;      // 整個 BMP 檔案大小
    uint16_t bfReserved1; // 保留欄位
    uint16_t bfReserved2; // 保留欄位
    uint32_t bfOffBits;   // pixel data 起始位置
} BMPHeader;

//DIB 標頭（BITMAPINFOHEADER，40 bytes） 
typedef struct {
    uint32_t biSize;          // DIB header 大小
    int32_t  biWidth;         // 影像寬度
    int32_t  biHeight;        // 影像高度（正值表示 bottom-up）
    uint16_t biPlanes;        // 固定為 1
    uint16_t biBitCount;      // 每像素位元數（24-bit）
    uint32_t biCompression;   // 壓縮方式（0 = BI_RGB）
    uint32_t biSizeImage;     // pixel data 大小
    int32_t  biXPelsPerMeter; // 水平解析度
    int32_t  biYPelsPerMeter; // 垂直解析度
    uint32_t biClrUsed;       // 使用顏色數
    uint32_t biClrImportant;  // 重要顏色數
} DIBHeader;
#pragma pack(pop)

int main(int argc, char *argv[]) {

    //檢查指令格式，只支援 Method 0 
    if (argc != 7 || atoi(argv[1]) != 0) {
        printf("Usage: decoder 0 output.bmp R.txt G.txt B.txt dim.txt\n");
        return 1;
    }

    //從 dim.txt 讀取影像尺寸
    //dim.txt 由 encoder 產生，內容為：width height
    int width, height;
    FILE *fD = fopen(argv[6], "r");
    fscanf(fD, "%d %d", &width, &height);
    fclose(fD);

    //開啟 RGB channel 文字檔 
    FILE *fR = fopen(argv[3], "r");
    FILE *fG = fopen(argv[4], "r");
    FILE *fB = fopen(argv[5], "r");

    //計算每一列 pixel data 大小與 padding
    //24-bit BMP：每 pixel 3 bytes (B, G, R)
    //每列需對齊至 4-byte boundary
    int row_bytes = width * 3;
    int padding = (4 - (row_bytes % 4)) % 4;

    //初始化 BMP 與 DIB header 
    BMPHeader bmp = {0};
    DIBHeader dib = {0};

    //設定 BMP header 欄位
    bmp.bfType = 0x4D42;  // 'BM'
    bmp.bfOffBits = sizeof(BMPHeader) + sizeof(DIBHeader);
    bmp.bfSize = bmp.bfOffBits + (row_bytes + padding) * height;

    // 設定 DIB header 欄位
    //biHeight 設為正值，表示 bottom-up BMP
    dib.biSize = sizeof(DIBHeader);
    dib.biWidth = width;
    dib.biHeight = height;
    dib.biPlanes = 1;
    dib.biBitCount = 24;
    dib.biCompression = 0; // BI_RGB
    dib.biSizeImage = (row_bytes + padding) * height;

    //以二進位模式開啟輸出 BMP 檔案 
    FILE *fp = fopen(argv[2], "wb");

    //先寫入 BMP header 與 DIB header 
    fwrite(&bmp, sizeof(bmp), 1, fp);
    fwrite(&dib, sizeof(dib), 1, fp);

    //依序從 RGB 文字檔讀取 pixel 資料，
    //並以 BGR 順序寫入 BMP pixel data。
    //encoder 輸出的 RGB 為 top-down 順序，
    //這裡直接依序寫入，對應 bottom-up BMP 的列順序。
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int R, G, B;
            fscanf(fR, "%d", &R);
            fscanf(fG, "%d", &G);
            fscanf(fB, "%d", &B);

            fputc(B, fp);  // BMP 格式為 BGR
            fputc(G, fp);
            fputc(R, fp);
        }

        //寫入 padding bytes 
        for (int p = 0; p < padding; p++)
            fputc(0, fp);
    }

    //關閉所有檔案 
    fclose(fp);
    fclose(fR);
    fclose(fG);
    fclose(fB);

    return 0;
}
