#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>          // Required for pthread_self()

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
cudaEvent_t hist_events[NUM_STREAMS];

// --- OPTIMIZED WARP SCAN HELPER ---
__device__ __forceinline__ int warp_scan(int val) {
    // Perform an inclusive scan within a warp (32 threads)
    // using register shuffles (no shared memory needed).
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += neighbor;
    }
    return val;
}

// --- REPLACED INCLUSIVE SCAN ---
__device__ void inclusive_scan(int* const s_data, const int tid, const int n) {
    // 1. Read current value into register
    int val = s_data[tid];

    // 2. Scan within each warp (intra-warp scan)
    val = warp_scan(val);

    // 3. Store the total sum of each warp into shared memory.
    // We reuse the first few slots of s_data to store warp sums.
    // For 256 threads, we have 8 warps.
    __syncthreads();
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // The last thread in the warp holds the full sum of that warp
    if (lane_id == 31) {
        s_data[warp_id] = val; 
    }
    __syncthreads();

    // 4. Scan the Warp Sums (Only the first warp needs to do this)
    if (warp_id == 0) {
        // Read the warp sums we just wrote
        int warp_sum = (tid < (n / 32)) ? s_data[tid] : 0;
        
        // Scan the sums
        warp_sum = warp_scan(warp_sum); 
        
        // Write the scanned sums back
        if (tid < (n / 32)) s_data[tid] = warp_sum;
    }
    __syncthreads();

    // 5. Add the base sum of previous warps to the current thread's value
    if (warp_id > 0) {
        val += s_data[warp_id - 1];
    }
    
    // 6. Write final result back to shared memory
    s_data[tid] = val;
    __syncthreads();
}

__global__ void compute_histogram(
    const unsigned char* const __restrict__ data,
    const int w, const int h,
    unsigned char* const __restrict__ all_luts,
    const int block_offset_x,
    const int block_offset_y,
    const int full_grid_w
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int bank_id = tid % NUM_BANKS;

    // Adjust logic to handle strips (block_offset_x will be 0, block_offset_y will be strip start)
    const int global_bx = blockIdx.x + block_offset_x;
    const int global_by = blockIdx.y + block_offset_y;

    const int x_start = global_bx * TILE_SIZE;
    const int y_start = global_by * TILE_SIZE;

    const int step_x = blockDim.x;
    const int step_y = blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int actual_tile_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
    const int actual_tile_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
    const int total_pixels = actual_tile_w * actual_tile_h;

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
            const int gx = x_start + cur_x;
            const int gy = y_start + cur_y;

            if (gx < w && gy < h) {
                const unsigned char pix = data[gy * w + gx];

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
    const int avg_inc = excess / 256;
    hist[tid] += avg_inc;
    __syncthreads();

    // CDF & LUT write
    inclusive_scan(hist, tid, 256);
    const int cdf = hist[tid];
    
    // Calculate final value
    int val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
    if (val > 255) val = 255; // 'val' is modified, cannot be const

    // Global indexing using full_grid_w
    const int lut_index = (global_by * full_grid_w + global_bx) * 256;
    all_luts[lut_index + tid] = (unsigned char)val;
}

__global__ void render_clahe(
    const unsigned char* const __restrict__ img_in,
    unsigned char* const __restrict__ img_out,
    const int w, const int h,
    const unsigned char* const __restrict__ all_luts,
    const int block_offset_y,
    const int full_grid_h
) {
    // Adjust Y calculation based on strip offset
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * (blockIdx.y + block_offset_y) + threadIdx.y;

    if (x >= w || y >= h) return;

    // Find relative position in the grid
    // (y / TILE_SIZE) gives the tile index, but we want the center approach
    // So we offset by 0.5 to align interpolation with tile centers
    const float ty_f = (float)y / TILE_SIZE - 0.5f;
    const float tx_f = (float)x / TILE_SIZE - 0.5f;

    // These are modified for boundary checking, cannot be const
    int y1 = (int)floor(ty_f);
    int x1 = (int)floor(tx_f);
    int y2 = y1 + 1;
    int x2 = x1 + 1;

    // Weights for interpolation
    const float y_weight = ty_f - y1;
    const float x_weight = tx_f - x1;

    // Clamp tile indices to boundaries 
    // If a pixel is near the edge, it might not have 4 neighbors
    if (x1 < 0) x1 = 0;
    if (x2 >= gridDim.x) x2 = gridDim.x - 1;  // gridDim.x is still full width
    if (y1 < 0) y1 = 0;

    // Bounds check using full grid height
    if (y2 >= full_grid_h) y2 = full_grid_h - 1;
    
    // Original pixel intensity
    const int val = img_in[y * w + x];

    // LUT lookup uses full grid width/height logic
    const int tl = all_luts[(y1 * gridDim.x + x1) * 256 + val];
    const int tr = all_luts[(y1 * gridDim.x + x2) * 256 + val];
    const int bl = all_luts[(y2 * gridDim.x + x1) * 256 + val];
    const int br = all_luts[(y2 * gridDim.x + x2) * 256 + val];

    // Bilinear interpolation
    const float top = tl * (1.0f - x_weight) + tr * x_weight;
    const float bot = bl * (1.0f - x_weight) + br * x_weight;
    const float final_val = top * (1.0f - y_weight) + bot * y_weight;

    img_out[y * w + x] = (unsigned char)(final_val + 0.5f);
}

void cleanUp() {
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(all_luts);
    for (int i=0; i<NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(hist_events[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
}

double d_apply_clahe(const PGM_IMG img_in, PGM_IMG * const img_out) {
    // NVTX: Name the Main Thread
    nvtxNameOsThread(pthread_self(), "CLAHE Main Thread");

    const int w = img_in.w;
    const int h = img_in.h;
    
    // Calculate grid dimensions
    const int grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    const int grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_per_strip = (grid_h + NUM_STREAMS - 1) / NUM_STREAMS;

    img_out->w = w;
    img_out->h = h;
    const int size = w * h * sizeof(unsigned char);
    img_out->img = (unsigned char *)malloc(size);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- TIMER START ---
    cudaEventRecord(start);

    // Pin so the GPU can access it asynchronously.
    cudaHostRegister(img_in.img, size, cudaHostRegisterDefault);
    cudaHostRegister(img_out->img, size, cudaHostRegisterDefault);

    cudaMalloc(&d_img_in, size);
    cudaMalloc(&d_img_out, size);
    cudaMalloc(&all_luts, grid_w * grid_h * 256 * sizeof(unsigned char));

    // Create streams and name them for Nsight Systems
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&hist_events[i]);

        char stream_name[32];
        sprintf(stream_name, "CLAHE Stream %d", i);
        nvtxNameCuStreamA(streams[i], stream_name);
    }

    // --- STAGE 1: Calculate Histograms ---
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int strip_start_grid = i * tiles_per_strip;

        // The start of strip is out of the image
        if (strip_start_grid >= grid_h) {
            cudaEventRecord(hist_events[i], streams[i]);
            continue;
        }

        // Last strip height check
        int current_strip_h_tiles = tiles_per_strip;

        if (strip_start_grid + current_strip_h_tiles > grid_h) {
            current_strip_h_tiles = grid_h - strip_start_grid;
        }

        const int start_pixel_y = strip_start_grid * TILE_SIZE;
        
        // Find the start and end row of the stream
        int end_pixel_y = start_pixel_y + (current_strip_h_tiles * TILE_SIZE);
        if (end_pixel_y > h) end_pixel_y = h;

        const int num_pixel_rows = end_pixel_y - start_pixel_y;
        if (num_pixel_rows <= 0) {
            cudaEventRecord(hist_events[i], streams[i]);
            continue;
        }

        const int chunk_size_bytes = num_pixel_rows * w * sizeof(unsigned char);
        const int pixel_offset = start_pixel_y * w;

        cudaMemcpyAsync(d_img_in + pixel_offset, img_in.img + pixel_offset,
                        chunk_size_bytes, cudaMemcpyHostToDevice, streams[i]);

        const dim3 dimGrid(grid_w, current_strip_h_tiles);
        const dim3 dimBlock(16, 16);

        compute_histogram<<<dimGrid, dimBlock, 0, streams[i]>>>(
            d_img_in, w, h, all_luts, 0, strip_start_grid, grid_w
        );

        cudaEventRecord(hist_events[i], streams[i]);
    }

    // --- STAGE 2: Render Image ---
    const dim3 dimBlockRender(w > TILE_SIZE ? TILE_SIZE : w, h > TILE_SIZE ? TILE_SIZE : h);

    for (int i = 0; i < NUM_STREAMS; i++) {
        const int strip_start_grid = i * tiles_per_strip;
        if (strip_start_grid >= grid_h) break;

        int current_strip_h_tiles = tiles_per_strip;
        if (strip_start_grid + current_strip_h_tiles > grid_h) {
            current_strip_h_tiles = grid_h - strip_start_grid;
        }

        // Synchronization: Wait for own histogram, and neighbors if necessary
        cudaStreamWaitEvent(streams[i], hist_events[i], 0);
        if (i > 0) cudaStreamWaitEvent(streams[i], hist_events[i-1], 0);
        if (i < NUM_STREAMS - 1) cudaStreamWaitEvent(streams[i], hist_events[i+1], 0);

        const int start_pixel_y = strip_start_grid * TILE_SIZE;
        int end_pixel_y = start_pixel_y + (current_strip_h_tiles * TILE_SIZE);
        if (end_pixel_y > h) end_pixel_y = h;

        const int num_pixel_rows = end_pixel_y - start_pixel_y;
        if (num_pixel_rows <= 0) continue;

        const int chunk_size_bytes = num_pixel_rows * w * sizeof(unsigned char);
        const int pixel_offset = start_pixel_y * w;

        const dim3 dimGridRender(grid_w, current_strip_h_tiles);
        
        render_clahe<<<dimGridRender, dimBlockRender, 0, streams[i]>>>(
            d_img_in, d_img_out, w, h, all_luts, strip_start_grid, grid_h
        );

        cudaMemcpyAsync(img_out->img + pixel_offset, d_img_out + pixel_offset,
                        chunk_size_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaHostUnregister(img_in.img);
    cudaHostUnregister(img_out->img);

    cleanUp();

    return elapsed;
}