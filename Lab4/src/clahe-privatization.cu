#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"

#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaDeviceSynchronize();                                             \
        cudaError_t _err = cudaGetLastError();                               \
        if (_err != cudaSuccess) {                                           \
            printf("CUDA Error: %s in %s, line %d\n",                        \
                   cudaGetErrorString(_err), __FILE__, __LINE__);            \
            cleanUp(DEVICE_ERROR);                                           \
        }                                                                    \
    } while (0)

__global__ void compute_histogram(unsigned char* img_data int image_w, int image_h) {
    __shared__ int priv_hist[256] = {0};
    __shared__ int priv_lut[256] = {0};

    int avg_inc, val;

    int excess = 0, cdf = 0;
    int total_tile_pixels = blockDim.x * blockDim.y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Build Histogram
    // Boundary check mostly for the right/bottom edge tiles
    if (x < image_w && y < image_h) {
        atomicAdd(priv_hist[img_data[y * image_w + x]], 1);
    }
    __syncthreads();

    // Clip Histogram
    // use up to 256 threads
    int i = threadIdx.y * blockDim.x + threadIdx.x;
    if (i < 256) {
        if (priv_hist[i] > CLIP_LIMIT) {
            atomicAdd(excess, priv_hist[i] - CLIP_LIMIT);
            priv_hist[i] = CLIP_LIMIT;
        }
    }
    __syncthreads();

    // Redistribute Excess (simplisticly)
    avg_inc = excess / 256;
    if (i < 256) {
        atomicAdd(priv_hist[i], avg_inc);
    }
    __syncthreads();

    // Compute CDF & LUT
    if (i < 256) {
        atomicAdd(cdf, priv_hist[i]);
        // Calculate equalized value
        val = (int)((float)cdf * 255.0f / total_tile_pixels + 0.5f);
        if (val > 255)
            val = 255;
        priv_lut[i] = val;
    }
    __syncthreads();

    // TODO transfer
}

// Core CLAHE
__host__ PGM_IMG apply_clahe(PGM_IMG img_in) {
    PGM_IMG img_out;
    int w = img_in.w;
    int h = img_in.h;
    int grid_w, grid_h;
    int *all_luts; // Big array to store LUTs for all tiles
    int* current_lut_ptr;
    int ty, tx, x, y, x1, x2, y1, y2, tl, tr, bl, br, val;
    int x_start, y_start, actual_tile_w, actual_tile_h;
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;

    PGM_IMG device_image_in;

    // Allocate output image
    img_out.w = w;
    img_out.h = h;
    img_out.img = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    // Calculate grid dimensions
    grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate memory for all LUTs: [grid_h][grid_w][256], as an 1D array
    // TODO: change to char if possible
    cudaMalloc((void **) &all_luts, grid_w * grid_h * 256 * sizeof(int));
    CUDA_CHECK_LAST_ERROR();

    // Transfer image from host to device
    cudaMalloc((void **) &device_image_in, w * h * sizeof(unsigned char));
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(device_image_in.img, image_in.img, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    // Precompute all Tile LUTs ---
    dim3 gridSize(grid_w, grid_h);
    dim3 blockSize(TILE_SIZE > 32 : 32 : TILE_SIZE, TILE_SIZE > 32 : 32 : TILE_SIZE);
    compute_histogram<<<gridSize, blockSize>>>(/* ... */);

    // Precompute all Tile LUTs ---
    // ...

    cudaFree(all_luts);
    return img_out;
}
