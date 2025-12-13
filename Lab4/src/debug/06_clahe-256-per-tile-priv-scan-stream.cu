#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// --- NVTX INCLUDES ---
// Use nvtx3 paths to avoid conflicts with system headers
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

#include "clahe.h"
#include "gputimer.h"

//! Lowers the number of conflicts, diminishing returns HIGH
#define NUM_BANKS 8
#define NUM_STREAMS 8

unsigned char *d_img_in = NULL;
unsigned char *d_img_out = NULL;
unsigned char *all_luts = NULL;

cudaEvent_t start, stop;
cudaStream_t streams[NUM_STREAMS];

__device__ void inclusive_scan(int* s_data, int tid, int n) {
    for (int stride = 1; stride < n; stride *= 2) {
        int val = 0;

        if (tid >= stride && tid < n)
            val = s_data[tid - stride];

        __syncthreads();

        if (tid >= stride && tid < n)
            s_data[tid] += val;

        __syncthreads();
    }
}

__global__ void compute_histogram(
    const unsigned char* __restrict__ data,
    int w, int h,
    unsigned char* __restrict__ all_luts,
    int block_offset_x,
    int block_offset_y,
    int full_grid_w
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bank_id = tid % NUM_BANKS;

    // Adjust logic to handle strips (block_offset_x will be 0, block_offset_y will be strip start)
    int global_bx = blockIdx.x + block_offset_x;
    int global_by = blockIdx.y + block_offset_y;

    int x_start = global_bx * TILE_SIZE;
    int y_start = global_by * TILE_SIZE;

    int step_x = blockDim.x;
    int step_y = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int actual_tile_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
    int actual_tile_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
    int total_pixels = actual_tile_w * actual_tile_h;

    __shared__ int p_hist[NUM_BANKS][256];
    __shared__ int hist[256];
    __shared__ int excess;

    //! Initialize private and shared histograms
    #pragma unroll
    for (int i = 0; i < NUM_BANKS; i++) p_hist[i][tid] = 0;
    hist[tid] = 0;
    __syncthreads();

    //! Add histogram values to seperate private histograms
    //* Indexing increased by blockDim (here is 16) 1 pixel for each quadrant
    for (int cur_y = ty; cur_y < TILE_SIZE; cur_y += step_y) {
        for (int cur_x = tx; cur_x < TILE_SIZE; cur_x += step_x) {
            int gx = x_start + cur_x;
            int gy = y_start + cur_y;

            if (gx < w && gy < h) {
                unsigned char pix = data[gy * w + gx];

                //! 1/<NUM_BANKS>th of the serialization 
                atomicAdd(&(p_hist[bank_id][pix]), 1);
            }
        }
    }
    __syncthreads();

    //! Sum the banks into the final result
    int bin_sum = 0;
    #pragma unroll
    for (int k = 0; k < NUM_BANKS; k++) {
        bin_sum += p_hist[k][tid];
    }
    hist[tid] = bin_sum;

    if (tid == 0) excess = 0;
    __syncthreads();

    // Clip
    if (hist[tid] > CLIP_LIMIT) {
        atomicAdd(&excess, (hist[tid] - CLIP_LIMIT));
        hist[tid] = CLIP_LIMIT;
    }
    __syncthreads();

    // Redistribute
    int avg_inc = excess / 256;
    hist[tid] += avg_inc;
    __syncthreads();

    // CDF & LUT write
    inclusive_scan(hist, tid, 256);
    int cdf = hist[tid];
    int val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
    if (val > 255) val = 255;

    // Global indexing using full_grid_w
    int lut_index = (global_by * full_grid_w + global_bx) * 256;
    all_luts[lut_index + tid] = (unsigned char)val;
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
    if (x1 < 0) x1 = 0;
    if (x2 >= gridDim.x) x2 = gridDim.x - 1;
    if (y1 < 0) y1 = 0;
    if (y2 >= gridDim.y) y2 = gridDim.y - 1;

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

double d_apply_clahe(PGM_IMG img_in, PGM_IMG *img_out) {
    // NVTX: Name the Main Thread
    nvtxNameOsThread(pthread_self(), "CLAHE Main Thread");

    int w = img_in.w;
    int h = img_in.h;
    float elapsed;

    // Calculate grid dimensions
    int grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    int strip_h = (grid_h + NUM_STREAMS - 1) / NUM_STREAMS;

    img_out->w = w;
    img_out->h = h;
    int size = w * h * sizeof(unsigned char);
    img_out->img = (unsigned char *)malloc(size);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Pin so the GPU can access it asynchronously
    cudaHostRegister(img_in.img, size, cudaHostRegisterDefault);
    cudaHostRegister(img_out->img, size, cudaHostRegisterDefault);

    // Create streams and name them
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        
        char stream_name[32];
        sprintf(stream_name, "CLAHE Stream %d", i);
        nvtxNameCuStreamA(streams[i], stream_name);
    }

    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&all_luts, grid_w * grid_h * 256 * sizeof(unsigned char));

    for (int i = 0; i < NUM_STREAMS; i++) {
        int strip_offset_y = i * strip_h;

        // The start of strip is out of the image
        if (strip_offset_y >= grid_h)
            break;

        // Last strip height check
        int current_strip_h = strip_h;

        if (strip_offset_y + current_strip_h > grid_h)
            current_strip_h = grid_h - strip_offset_y;

        // Find the start and end row of the stream
        int strip_offset_y_pixel = strip_offset_y * TILE_SIZE;
        int end_pixel_y = strip_offset_y_pixel + (current_strip_h * TILE_SIZE);

        if (end_pixel_y > h)
            end_pixel_y = h;

        int num_pixel_rows = end_pixel_y - strip_offset_y_pixel;

        if (num_pixel_rows <= 0)
            continue;

        // Find size of strip and start of strip (both in pixels)
        int chunk_size_bytes = num_pixel_rows * w * sizeof(unsigned char);
        int pixel_offset = strip_offset_y_pixel * w;

        unsigned char* h_src_ptr = img_in.img + pixel_offset;
        unsigned char* d_dst_ptr = d_img_in + pixel_offset;

        // Transfer strip H2D
        cudaMemcpyAsync(d_dst_ptr, h_src_ptr, chunk_size_bytes, cudaMemcpyHostToDevice, streams[i]);

        dim3 dimGrid(grid_w, current_strip_h);
        dim3 dimBlock(16, 16);

        compute_histogram<<<dimGrid, dimBlock, 0, streams[i]>>>(d_img_in, w, h, all_luts, 0, strip_offset_y, grid_w);
    }

    // Wait for all histograms to complete
    cudaDeviceSynchronize();

    // Render Full Image
    dim3 dimGridRender(grid_w, grid_h);
    dim3 dimBlockRender(w > TILE_SIZE ? TILE_SIZE : w, h > TILE_SIZE ? TILE_SIZE : h);

    render_clahe<<<dimGridRender, dimBlockRender>>>(d_img_in, d_img_out, w, h, all_luts);

    // Copy Back
    cudaMemcpy(img_out->img, d_img_out, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaHostUnregister(img_in.img);
    cudaHostUnregister(img_out->img);

    cleanUp();

    return elapsed;
}

void cleanUp() {
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(all_luts);
    for(int i=0; i<NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
}