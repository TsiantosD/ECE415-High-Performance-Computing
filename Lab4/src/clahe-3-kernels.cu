#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "helpers.h"

// Compute & Clip Histogram for a specific tile
__global__ void compute_histogram(unsigned char* data, int w, int h, int tile_w, int tile_h, int *hist) {
    int x = 0, y = 0; 

    // Build Histogram
    if(x < w && y < h) {
        atomicAdd(&(hist[data[y * w + x]]), 1);
    }
}

__global__ void clipAndComputeLuts(int *hist, int* lut, int tile_w, int tile_h) {
    int val, avg_inc, cdf = 0, total_pixels = tile_w * tile_h; 
    int i = threadIdx.x;
    __shared__ int excess;

    // Clip Histogram
    if (hist[i] > CLIP_LIMIT) {
        atomicAdd(&excess,  (hist[i] - CLIP_LIMIT));
        hist[i] = CLIP_LIMIT;
    }
    __syncthreads();

    // Redistribute Excess (simplisticly)
    avg_inc = excess / 256;
    
    // Compute CDF & LUT
    cdf += hist[i] + avg_inc; //TODO wth this doesn't work at all 
    // Calculate equalized value
    val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
    if (val > 255) 
        val = 255;
    lut[i] = val;
}

__global__ void renderClahe(PGM_IMG *img_in, PGM_IMG *img_out, int w, int h, int *all_luts) {
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;
    int x1, x2, y1, y2, tl, tr, bl, br, val;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

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
    val = img_in->img[y * w + x];
    
    // Fetch mapped values from the 4 nearest tile LUTs
    tl = all_luts[(y1 * gridDim.x + x1) * 256 + val];
    tr = all_luts[(y1 * gridDim.x + x2) * 256 + val];
    bl = all_luts[(y2 * gridDim.x + x1) * 256 + val];
    br = all_luts[(y2 * gridDim.x + x2) * 256 + val];

    // Bilinear interpolation
    top = tl * (1.0f - x_weight) + tr * x_weight;
    bot = bl * (1.0f - x_weight) + br * x_weight;
    final_val = top * (1.0f - y_weight) + bot * y_weight;

    img_out->img[y * w + x] = (unsigned char)(final_val + 0.5f);
}

char *d_img_in;
char *d_img_out;
int *all_hist;
int *all_luts;

// Core CLAHE
double d_apply_clahe(PGM_IMG img_in, PGM_IMG *img_out) {
    int w = img_in.w;
    int h = img_in.h;
    int grid_w, grid_h;

    // Calculate grid dimensions
    grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    img_out->w = w;
    img_out->h = h;
    img_out->img = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    
    // Allocate memory for all LUTs: [grid_h][grid_w][256],
    // as an 1D array
    cudaMalloc(&d_img_in, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_img_out, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&all_hist, grid_w * grid_h * 256 * sizeof(int));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&all_luts, grid_w * grid_h * 256 * sizeof(int));
    CUDA_CHECK_LAST_ERROR();

    // Precompute all Tile LUTs ---
    // for (ty = 0; ty < grid_h; ++ty) {
    //     for (tx = 0; tx < grid_w; ++tx) {
    //         x_start = tx * TILE_SIZE;
    //         y_start = ty * TILE_SIZE;
            
    //         // Handle boundary tiles that might be smaller than TILE_SIZE
    //         actual_tile_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
    //         actual_tile_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
            
    //         // Pointer to the specific 256-entry LUT for this tile
    //         current_lut_ptr = &all_luts[(ty * grid_w + tx) * 256];
            
    //         compute_histogram(img_in.img, w, h, x_start, y_start, 
    //                           actual_tile_w, actual_tile_h, 
    //                           current_lut_ptr);
    //     }
    // }

    cleanUp();
    return 0;
}

void cleanUp() {
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(all_hist);
    cudaFree(all_luts);
    cudaDeviceReset();
}