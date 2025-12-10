#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"

__global__ void bilinear_interpolation(unsigned char* img_data, int* all_luts) {
    int x = threadIdx.y * blockDim.x + threadIdx.x;
    int y = threadIdx.x * blockDim.y + threadIdx.y;

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
    if (x2 >= grid_w)
        x2 = grid_w - 1;
    if (y1 < 0)
        y1 = 0;
    if (y2 >= grid_h)
        y2 = grid_h - 1;

    // Original pixel intensity
    val = img_data[y * w + x];

    // Fetch mapped values from the 4 nearest tile LUTs
    tl = all_luts[(y1 * grid_w + x1) * 256 + val];
    tr = all_luts[(y1 * grid_w + x2) * 256 + val];
    bl = all_luts[(y2 * grid_w + x1) * 256 + val];
    br = all_luts[(y2 * grid_w + x2) * 256 + val];

    // Bilinear interpolation
    top = tl * (1.0f - x_weight) + tr * x_weight;
    bot = bl * (1.0f - x_weight) + br * x_weight;
    final_val = top * (1.0f - y_weight) + bot * y_weight;

    img_data[y * w + x] = (unsigned char)(final_val + 0.5f);
}

__global__ void compute_histogram(unsigned char* img_data, int image_w, int image_h, int* all_luts) {
    __shared__ int priv_hist[256] = {0};
    __shared__ int priv_lut[256] = {0};

    int total_tile_pixels = blockDim.x * blockDim.y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Build Histogram
    // Boundary check mostly for the right/bottom edge tiles
    if (x < image_w && y < image_h) {
        atomicAdd(&(priv_hist[img_data[y * image_w + x]]), 1);
    }
    __syncthreads();

    // Clip Histogram
    // use up to 256 threads
    int excess = 0, cdf = 0;
    int i = threadIdx.y * blockDim.x + threadIdx.x;

    if (i < 256) {
        if (priv_hist[i] > CLIP_LIMIT) {
            atomicAdd(&excess, priv_hist[i] - CLIP_LIMIT);
            priv_hist[i] = CLIP_LIMIT;
        }
    }
    __syncthreads();

    // Redistribute Excess (simplisticly)
    int avg_inc;
    avg_inc = excess / 256;
    if (i < 256) {
        atomicAdd(&(priv_hist[i]), avg_inc);
    }
    __syncthreads();

    // Compute CDF & LUT
    int val;
    if (i < 256) {
        atomicAdd(&cdf, priv_hist[i]);
        // Calculate equalized value
        val = (int)((float)cdf * 255.0f / total_tile_pixels + 0.5f);
        if (val > 255)
            val = 255;
        priv_lut[i] = val;
    }
    __syncthreads();

    all_luts[threadIdx.y * blockDim.x + threadIdx.x] = priv_lut;
}

unsigned char* d_img_data_in;
unsigned char* d_img_data_out;
int *all_luts; // Big array to store LUTs for all tiles

// Core CLAHE
__host__ double d_apply_clahe(PGM_IMG img_in, PGM_IMG* img_out) {
    int w = img_in.w;
    int h = img_in.h;
    int grid_w, grid_h;
    int* current_lut_ptr;
    int ty, tx, x, y, x1, x2, y1, y2, tl, tr, bl, br, val;
    int x_start, y_start, actual_tile_w, actual_tile_h;
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;

    // Allocate output image
    img_out->w = w;
    img_out->h = h;
    img_out->img = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    // Calculate grid dimensions
    grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate memory for all LUTs: [grid_h][grid_w][256], as an 1D array
    // TODO: change to char if possible
    cudaMalloc((void **) &all_luts, grid_w * grid_h * 256 * sizeof(int));
    CUDA_CHECK_LAST_ERROR();

    cudaMalloc((void **) &d_img_data_in, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();

    // Allocate space for device output image
    cudaMalloc((void **) &d_img_data_out, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();

    GpuTimer timer = GpuTimer();
    timer.Start();

    // Transfer image from host to device
    cudaMemcpy(d_img_data_in, img_in.img, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    dim3 gridSize(grid_w, grid_h);
    dim3 blockSize(w < 32 ? w : TILE_SIZE, h < 32 ? h : TILE_SIZE);

    // Precompute all Tile LUTs ---
    compute_histogram<<<gridSize, blockSize>>>(d_img_data_in, w, h, all_luts);
    CUDA_CHECK_LAST_ERROR();

    // Render pixels using Bilinear Interpolation
    bilinear_interpolation<<<gridSize, blockSize>>>(d_img_data_out, all_luts);
    CUDA_CHECK_LAST_ERROR();

    cudaMemcpy(img_out->img, d_img_data_out, w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();

    timer.Stop();

    cleanUp();

    return 0;
}

void cleanUp() {
    cudaFree(d_img_data_in);
    cudaFree(d_img_data_out);
    cudaFree(all_luts);
    cudaDeviceReset();
}
