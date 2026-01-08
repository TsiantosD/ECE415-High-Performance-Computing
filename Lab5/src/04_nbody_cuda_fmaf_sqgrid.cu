#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "helpers.h"
#include "gputimer.h"
#include <unistd.h>

#ifndef GPU_MAX
#define GPU_MAX 4
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 16 
#endif

#define THREADS_PER_BLOCK (BLOCK_DIM * BLOCK_DIM)

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

GalaxySoA systemsHost;
GalaxySoA systemsDevice[GPU_MAX];

//TODO running this with -Xptxas shows register usage going up 
//TODO although occupancy goes down this enables faster threads -> speedup
//TODO 2 is faster and 8 is 100% + slower
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) calculate_forces_kernel(GalaxySoA galaxy, int bodies_per_system, float dt) {
    int system_idx = blockIdx.y;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int body_local_idx = blockIdx.x * THREADS_PER_BLOCK + tid;
    int system_offset = system_idx * bodies_per_system;
    
    if (body_local_idx >= bodies_per_system) return;

    int global_idx = system_offset + body_local_idx;

    float my_x = galaxy.x[global_idx];
    float my_y = galaxy.y[global_idx];
    float my_z = galaxy.z[global_idx];
    
    float my_vx = galaxy.vx[global_idx];
    float my_vy = galaxy.vy[global_idx];
    float my_vz = galaxy.vz[global_idx];

    // Shared memory size is now 16*16 = 256 floats
    __shared__ float sh_x[THREADS_PER_BLOCK];
    __shared__ float sh_y[THREADS_PER_BLOCK];
    __shared__ float sh_z[THREADS_PER_BLOCK];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    // We step through the system in tiles of size 256
    int num_tiles = (bodies_per_system + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int tile = 0; tile < num_tiles; tile++) {
        int t_idx = tile * THREADS_PER_BLOCK + tid;
        
        // Load data cooperatively using the flattened ID
        if (t_idx < bodies_per_system) {
            sh_x[tid] = galaxy.x[system_offset + t_idx];
            sh_y[tid] = galaxy.y[system_offset + t_idx];
            sh_z[tid] = galaxy.z[system_offset + t_idx];
        } else {
            sh_x[tid] = 0.0f;
            sh_y[tid] = 0.0f;
            sh_z[tid] = 0.0f;
        }
        __syncthreads();

        // Iterate over the tile (0 to 255)
        #pragma unroll 16
        for (int j = 0; j < THREADS_PER_BLOCK; j++) {
            int interaction_idx = tile * THREADS_PER_BLOCK + j;
            if (interaction_idx >= bodies_per_system) break;

            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dz = sh_z[j] - my_z;
            
            float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx = fmaf(dx, invDist3, Fx);
            Fy = fmaf(dy, invDist3, Fy);
            Fz = fmaf(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    galaxy.vx[global_idx] = fmaf(dt, Fx, my_vx);
    galaxy.vy[global_idx] = fmaf(dt, Fy, my_vy);
    galaxy.vz[global_idx] = fmaf(dt, Fz, my_vz);
}

__global__ void integrate_positions_kernel(GalaxySoA galaxy, int bodies_per_system, float dt) {
    int system_idx = blockIdx.y; 
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int body_local_idx = blockIdx.x * THREADS_PER_BLOCK + tid;

    int system_offset = system_idx * bodies_per_system;
    
    if (body_local_idx >= bodies_per_system) return;

    int global_idx = system_offset + body_local_idx;

    galaxy.x[global_idx] = fmaf(galaxy.vx[global_idx], dt, galaxy.x[global_idx]);
    galaxy.y[global_idx] = fmaf(galaxy.vy[global_idx], dt, galaxy.y[global_idx]);
    galaxy.z[global_idx] = fmaf(galaxy.vz[global_idx], dt, galaxy.z[global_idx]);
}

void cleanUp(void) {
    #pragma omp critical
    {
        destroy_timer();
    
        if (systemsHost.x) { free(systemsHost.x); systemsHost.x = NULL; }
        if (systemsHost.y) { free(systemsHost.y); systemsHost.y = NULL; }
        if (systemsHost.z) { free(systemsHost.z); systemsHost.z = NULL; }
        if (systemsHost.vx) { free(systemsHost.vx); systemsHost.vx = NULL; }
        if (systemsHost.vy) { free(systemsHost.vy); systemsHost.vy = NULL; }
        if (systemsHost.vz) { free(systemsHost.vz); systemsHost.vz = NULL; }

        for (int i = 0; i < GPU_MAX; i++) {
            if (systemsDevice[i].x != NULL) {
                cudaSetDevice(i);
                
                cudaFree(systemsDevice[i].x); systemsDevice[i].x = NULL;
                cudaFree(systemsDevice[i].y); systemsDevice[i].y = NULL;
                cudaFree(systemsDevice[i].z); systemsDevice[i].z = NULL;
                cudaFree(systemsDevice[i].vx); systemsDevice[i].vx = NULL;
                cudaFree(systemsDevice[i].vy); systemsDevice[i].vy = NULL;
                cudaFree(systemsDevice[i].vz); systemsDevice[i].vz = NULL;
                
                cudaDeviceReset();
            }
        }
    }
}

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *data) {
    
    int totalBodies = bodies_per_system * num_systems;

    systemsHost.x = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.y = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.z = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.vx = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.vy = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.vz = (float *) malloc(totalBodies * sizeof(float));

    for (int curBody = 0; curBody < totalBodies; curBody++) {
        systemsHost.x[curBody] = data[curBody].x;
        systemsHost.y[curBody] = data[curBody].y;
        systemsHost.z[curBody] = data[curBody].z;
        systemsHost.vx[curBody] = data[curBody].vx;
        systemsHost.vy[curBody] = data[curBody].vy;
        systemsHost.vz[curBody] = data[curBody].vz;
    }
    
    omp_set_nested(0);
    omp_set_dynamic(0);

    int gpu_num = 0;
    cudaGetDeviceCount(&gpu_num);
    if (gpu_num == 0) return 0.0;
    CUDA_CHECK_LAST_ERROR();

    int gpu_used = gpu_num > GPU_MAX ? GPU_MAX : gpu_num; 
    printf("Running on %d GPUs.\n", gpu_used);

    omp_set_num_threads(gpu_used);
    create_timer();
    start_timer();

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int gpu_id = thread_id % gpu_used; 
        
        cudaSetDevice(gpu_id);
        CUDA_CHECK_LAST_ERROR();
        
        int systems_per_gpu = (num_systems + gpu_used - 1) / gpu_used;
        int start_sys = gpu_id * systems_per_gpu;
        int end_sys = (start_sys + systems_per_gpu < num_systems) ? start_sys + systems_per_gpu : num_systems;
        int my_system_count = end_sys - start_sys;
        int bytes = my_system_count * bodies_per_system * sizeof(float);
        
        if (my_system_count > 0) {
            cudaMalloc(&(systemsDevice[gpu_id].x), bytes);
            cudaMalloc(&(systemsDevice[gpu_id].y), bytes);
            cudaMalloc(&(systemsDevice[gpu_id].z), bytes);
            cudaMalloc(&(systemsDevice[gpu_id].vx), bytes);
            cudaMalloc(&(systemsDevice[gpu_id].vy), bytes);
            cudaMalloc(&(systemsDevice[gpu_id].vz), bytes);
            CUDA_CHECK_LAST_ERROR();

            cudaMemcpy(systemsDevice[gpu_id].x, &(systemsHost.x[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(systemsDevice[gpu_id].y, &(systemsHost.y[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(systemsDevice[gpu_id].z, &(systemsHost.z[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(systemsDevice[gpu_id].vx, &(systemsHost.vx[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(systemsDevice[gpu_id].vy, &(systemsHost.vy[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(systemsDevice[gpu_id].vz, &(systemsHost.vz[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            CUDA_CHECK_LAST_ERROR();
            
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            dim3 grid((bodies_per_system + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, my_system_count);

            for (int iter = 0; iter < nIters; iter++) {
                calculate_forces_kernel<<<grid, threads>>>(
                    systemsDevice[gpu_id],
                    bodies_per_system, 
                    dt
                );
                
                integrate_positions_kernel<<<grid, threads>>>(
                    systemsDevice[gpu_id],
                    bodies_per_system, 
                    dt
                );
            }
            
            cudaDeviceSynchronize();
            CUDA_CHECK_LAST_ERROR();

            cudaMemcpy(&(systemsHost.x[start_sys * bodies_per_system]), systemsDevice[gpu_id].x, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&(systemsHost.y[start_sys * bodies_per_system]), systemsDevice[gpu_id].y, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&(systemsHost.z[start_sys * bodies_per_system]), systemsDevice[gpu_id].z, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&(systemsHost.vx[start_sys * bodies_per_system]), systemsDevice[gpu_id].vx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&(systemsHost.vy[start_sys * bodies_per_system]), systemsDevice[gpu_id].vy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&(systemsHost.vz[start_sys * bodies_per_system]), systemsDevice[gpu_id].vz, bytes, cudaMemcpyDeviceToHost);
            CUDA_CHECK_LAST_ERROR();
        }
    }

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for (int i = 0; i < totalBodies; i++) {
        OMP_PRINT_NUM_THREADS("in write back", i == 0, omp_get_thread_num());
        data[i].x = systemsHost.x[i];
        data[i].y = systemsHost.y[i];
        data[i].z = systemsHost.z[i];
        data[i].vx = systemsHost.vx[i];
        data[i].vy = systemsHost.vy[i];
        data[i].vz = systemsHost.vz[i];
    }

    stop_timer();
    double total_time = (double) get_timer_ms() / 1000.0f;
    cleanUp();

    return total_time;
}