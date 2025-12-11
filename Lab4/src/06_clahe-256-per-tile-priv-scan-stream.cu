#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"
#include "gputimer.h"

// --- CONFIGURATION ---
#define NUM_BANKS 8 
#define NUM_STREAMS 8

// --- GLOBALS ---
unsigned char *d_img_in = NULL;
unsigned char *d_img_out = NULL;
unsigned char *all_luts = NULL;

cudaEvent_t start, stop;
cudaStream_t streams[NUM_STREAMS];

// --- DEVICE FUNCTIONS ---
__device__ void inclusive_scan(int* s_data, int tid, int n) {
    for (int stride = 1; stride < n; stride *= 2) {
        int val = 0;
        if (tid >= stride && tid < n) val = s_data[tid - stride];
        __syncthreads();
        if (tid >= stride && tid < n) s_data[tid] += val;
        __syncthreads();
    }
}

// --- KERNELS ---
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
    
    __shared__ int p_hist[NUM_BANKS][256 + 1]; 
    __shared__ int hist[256];
    __shared__ int excess;

    #pragma unroll
    for (int i = 0; i < NUM_BANKS; i++) p_hist[i][tid] = 0;
    hist[tid] = 0;
    __syncthreads();

    // Loop over the tile pixels
    for (int cur_y = ty; cur_y < TILE_SIZE; cur_y += step_y) {
        for (int cur_x = tx; cur_x < TILE_SIZE; cur_x += step_x) {
            int gx = x_start + cur_x;
            int gy = y_start + cur_y;

            if (gx < w && gy < h) {
                unsigned char pix = data[gy * w + gx];
                atomicAdd(&(p_hist[bank_id][pix]), 1);
            }
        }
    }
    __syncthreads();

    // Sum banks
    int bin_sum = 0;
    #pragma unroll
    for (int k = 0; k < NUM_BANKS; k++) bin_sum += p_hist[k][tid];
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
    
    // Correct global indexing using the passed full_grid_w
    int lut_index = (global_by * full_grid_w + global_bx) * 256;
    all_luts[lut_index + tid] = (unsigned char)val;
}

__global__ void render_clahe(unsigned char *img_in, unsigned char *img_out, int w, int h, unsigned char *all_luts) {
    // Render logic remains exactly the same
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;
    int x1, x2, y1, y2, tl, tr, bl, br, val;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= h) return;

    ty_f = (float)y / TILE_SIZE - 0.5f;
    tx_f = (float)x / TILE_SIZE - 0.5f;
    
    y1 = (int)floor(ty_f);
    x1 = (int)floor(tx_f);
    y2 = y1 + 1;
    x2 = x1 + 1;

    y_weight = ty_f - y1;
    x_weight = tx_f - x1;

    if (x1 < 0) x1 = 0;
    if (x2 >= gridDim.x) x2 = gridDim.x - 1;
    if (y1 < 0) y1 = 0;
    if (y2 >= gridDim.y) y2 = gridDim.y - 1;

    val = img_in[y * w + x];
    
    tl = all_luts[(y1 * gridDim.x + x1) * 256 + val];
    tr = all_luts[(y1 * gridDim.x + x2) * 256 + val];
    bl = all_luts[(y2 * gridDim.x + x1) * 256 + val];
    br = all_luts[(y2 * gridDim.x + x2) * 256 + val];

    top = tl * (1.0f - x_weight) + tr * x_weight;
    bot = bl * (1.0f - x_weight) + br * x_weight;
    final_val = top * (1.0f - y_weight) + bot * y_weight;

    img_out[y * w + x] = (unsigned char)(final_val + 0.5f);
}

double d_apply_clahe(PGM_IMG img_in, PGM_IMG *img_out) {
    int w = img_in.w;
    int h = img_in.h;
    float elapsed;

    int grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_per_strip = (grid_h + NUM_STREAMS - 1) / NUM_STREAMS;

    img_out->w = w;
    img_out->h = h;
    int size = w * h * sizeof(unsigned char);
    img_out->img = (unsigned char *)malloc(size);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    
    cudaEventRecord(start);
    // Pin so the GPU can access it asynchronously.
    cudaHostRegister(img_in.img, size, cudaHostRegisterDefault);
    cudaHostRegister(img_out->img, size, cudaHostRegisterDefault);
    
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&streams[i]);
    
    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&all_luts, grid_w * grid_h * 256 * sizeof(unsigned char));
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        //TODO REMOVE
        //! --- GEMINI STARTS HERE ---
        int strip_start_grid = i * tiles_per_strip;
        
        if (strip_start_grid >= grid_h) break;

        int current_strip_h_tiles = tiles_per_strip;
        if (strip_start_grid + current_strip_h_tiles > grid_h) {
            current_strip_h_tiles = grid_h - strip_start_grid;
        }

        int start_pixel_y = strip_start_grid * TILE_SIZE;
        int end_pixel_y = start_pixel_y + (current_strip_h_tiles * TILE_SIZE);
        if (end_pixel_y > h) end_pixel_y = h;
        
        int num_pixel_rows = end_pixel_y - start_pixel_y;
        if (num_pixel_rows <= 0) continue;

        int chunk_size_bytes = num_pixel_rows * w * sizeof(unsigned char);
        int pixel_offset = start_pixel_y * w;

        unsigned char* h_src_ptr = img_in.img + pixel_offset;
        unsigned char* d_dst_ptr = d_img_in + pixel_offset;

        cudaMemcpyAsync(d_dst_ptr, h_src_ptr, chunk_size_bytes, cudaMemcpyHostToDevice, streams[i]);

        dim3 dimGrid(grid_w, current_strip_h_tiles);
        dim3 dimBlock(16, 16);

        compute_histogram<<<dimGrid, dimBlock, 0, streams[i]>>>(d_img_in, w, h, all_luts, 0, strip_start_grid, grid_w);
        //! --- GEMINI ENDS HERE ---
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

    // --- MAGIC FIX: UNREGISTER ---
    // Unlock the pages so main() can free them safely later.
    cudaHostUnregister(img_in.img);
    cudaHostUnregister(img_out->img);
    
    return elapsed;
}

// Called by main() at program exit
void cleanUp() {
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(all_luts);
    for(int i=0; i<NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
}