#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"
#include "gputimer.h"

// Compute & Clip Histogram for a specific tile
__global__ void compute_histogram(unsigned char* data, int w, int h, unsigned char *all_luts) {
    int i = threadIdx.y * blockDim.x + threadIdx.x;
    int val, avg_inc, total_pixels, cdf = 0;
    int x_start = blockIdx.x * TILE_SIZE;
    int y_start = blockIdx.y * TILE_SIZE;
    int actual_tile_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
    int actual_tile_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
    total_pixels = actual_tile_w * actual_tile_h;
    unsigned char *lut = &(all_luts[(blockIdx.y * gridDim.x + blockIdx.x) * 256]);
    __shared__ int excess;
    __shared__ int hist[256];

    hist[i] = 0;
    __syncthreads();

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int ky = 0; ky < 2; ky++) {
        for (int kx = 0; kx < 2; kx++) {
            int gx = x_start + tx + kx * 16;
            int gy = y_start + ty + ky * 16;

            if (gx < w && gy < h && gx < x_start + TILE_SIZE && gy < y_start + TILE_SIZE) {
                unsigned char pix = data[gy * w + gx];
                atomicAdd(&(hist[pix]), 1);
            }
        }
    }

    if (i == 0) excess = 0;
    __syncthreads();

    // Clip Histogram
    if (hist[i] > CLIP_LIMIT) {
        atomicAdd(&excess,  (hist[i] - CLIP_LIMIT));
        hist[i] = CLIP_LIMIT;
    }
    __syncthreads();

    // Redistribute Excess (simplisticly)
    avg_inc = excess / 256;
    
    // Compute CDF & LUT
    for (int j = 0; j <= i; j++)
        cdf += hist[j] + avg_inc;
    
    // Calculate equalized value
    val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
    if (val > 255) 
        val = 255;
    lut[i] = val;
}

__global__ void render_clahe(unsigned char *img_in, unsigned char *img_out, int w, int h, unsigned char *all_luts) {
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;
    int x1, x2, y1, y2, tl, tr, bl, br, val;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Find relative position in the grid
    // (y / TILE_SIZE) gives the tile index, but we want the center approach
    // So we offset by 0.5 to align interpolation with tile centers
    ty_f = (float)y / TILE_SIZE - 0.5f;
    tx_f = (float)x / TILE_SIZE - 0.5f;
    
    y1 = (int)floor(ty_f);
    x1 = (int)floor(tx_f);
    y2 = y1 + 1;
    x2 = x1 + 1;

    // Weights for interpolation
    y_weight = ty_f - y1;
    x_weight = tx_f - x1;

    // Clamp tile indices to boundaries 
    // If a pixel is near the edge, it might not have 4 neighbors
    if (x1 < 0) 
        x1 = 0;
    if (x2 >= gridDim.x) 
        x2 = gridDim.x - 1;
    if (y1 < 0) 
        y1 = 0;
    if (y2 >= gridDim.y) 
        y2 = gridDim.y - 1;

    // Original pixel intensity
    val = img_in[y * w + x];
    
    // Fetch mapped values from the 4 nearest tile LUTs
    tl = all_luts[(y1 * gridDim.x + x1) * 256 + val];
    tr = all_luts[(y1 * gridDim.x + x2) * 256 + val];
    bl = all_luts[(y2 * gridDim.x + x1) * 256 + val];
    br = all_luts[(y2 * gridDim.x + x2) * 256 + val];

    // Bilinear interpolation
    top = tl * (1.0f - x_weight) + tr * x_weight;
    bot = bl * (1.0f - x_weight) + br * x_weight;
    final_val = top * (1.0f - y_weight) + bot * y_weight;

    img_out[y * w + x] = (unsigned char)(final_val + 0.5f);
}

unsigned char *d_img_in;
unsigned char *d_img_out;
unsigned char *all_luts;
cudaEvent_t start, stop;

// Core CLAHE
double d_apply_clahe(PGM_IMG img_in, PGM_IMG *img_out) {
    int w = img_in.w;
    int h = img_in.h;
    int grid_w, grid_h;
    float elapsed;

    // Calculate grid dimensions
    grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    img_out->w = w;
    img_out->h = h;
    img_out->img = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    cudaEventCreate(&start);
    CUDA_CHECK_LAST_ERROR();
    cudaEventCreate(&stop);
    CUDA_CHECK_LAST_ERROR();

    cudaEventRecord(start);
    cudaMalloc(&d_img_in, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_img_out, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&all_luts, grid_w * grid_h * 256 * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_img_in, img_in.img, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    dim3 dimGrid(grid_w, grid_h);
    dim3 dimBlock(16, 16);
    compute_histogram<<<dimGrid, dimBlock>>>(d_img_in, w, h, all_luts);
    CUDA_CHECK_LAST_ERROR();

    dimGrid = dim3(grid_w, grid_h);
    dimBlock = dim3(w > TILE_SIZE ? TILE_SIZE : w, h > TILE_SIZE ? TILE_SIZE : h);
    render_clahe<<<dimGrid, dimBlock>>>(d_img_in, d_img_out, w, h, all_luts);
    CUDA_CHECK_LAST_ERROR();

    cudaMemcpy(img_out->img, d_img_out, w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cleanUp();
    
    return elapsed;
}

void cleanUp() {
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(all_luts);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
}