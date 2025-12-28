# MMSP 2025 Final Project  
## Method 0 – BMP I/O Verification

---

## Objective

The goal of **Method 0** is to verify correct BMP image input and output before implementing JPEG-related compression methods.

This method checks that:

- A 24-bit BMP file can be read correctly
- RGB channels can be properly extracted
- The original image can be reconstructed from the RGB data

Method 0 serves as a basic sanity check for image I/O.

---

## Program Usage (Method 0)

### Encoder

`encoder 0 Kimberly.bmp R.txt G.txt B.txt dim.txt`

- **Kimberly.bmp**: Input 24-bit BMP image  
- **R.txt / G.txt / B.txt**: ASCII files storing RGB channel values  
- **dim.txt**: Image width and height  

### Decoder

`decoder 0 ResKimberly.bmp R.txt G.txt B.txt dim.txt`

- Reconstructs a BMP image from RGB text files  
- Output image is saved as **ResKimberly.bmp**

---

## Implementation Notes

- Supports 24-bit uncompressed BMP format (BI_RGB)
- BMP pixel data is stored in BGR order
- Each scanline is padded to a 4-byte boundary
- Pixel rows are processed in bottom-up order, following the BMP specification

The encoder reads pixel data row by row and separates RGB channels.  
The decoder reads RGB values and reconstructs the BMP image accordingly.

---

## Verification

### Linux `cmp` (Official Verification)

According to the assignment requirement, verification is performed in Linux.  
In GitHub Actions (Ubuntu runner), the following command is used:

`cmp Kimberly.bmp ResKimberly.bmp`

If no output is produced, the two files are bitwise identical and Method 0 is considered correct.

### Windows Testing Note

When testing on Windows using SHA256 hash comparison, the hash values may differ due to BMP header metadata differences.  
Byte-level comparison confirmed that pixel data is correct.  
Therefore, Linux `cmp` is used as the final verification method.

---

## Conclusion

Method 0 successfully verifies that:

- BMP image I/O is correctly implemented
- RGB channel extraction and reconstruction are correct
- Pixel data is preserved through the encoder–decoder pipeline

This provides a reliable foundation for subsequent JPEG-related methods.
