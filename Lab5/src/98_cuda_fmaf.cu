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

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

GalaxySoA systemsHost;
GalaxySoA systemsDevice[GPU_MAX];

__global__ void __launch_bounds__(BLOCK_SIZE, 2) calculate_forces_kernel(GalaxySoA galaxy, int bodies_per_system, float dt, int my_system) {
    //! Find system and position of threads in a range of [0, BLOCK_SIZE] 
    int system_idx = my_system; 
    int body_local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int system_offset = system_idx * bodies_per_system;
    
    //! Global memory index for the current block of bodies (of size BLOCK_SIZE)
    int global_idx = system_offset + body_local_idx;

    float my_x = __ldg(&galaxy.x[global_idx]);
    float my_y = __ldg(&galaxy.y[global_idx]);
    float my_z = __ldg(&galaxy.z[global_idx]);

    __shared__ float sh_x[BLOCK_SIZE];
    __shared__ float sh_y[BLOCK_SIZE];
    __shared__ float sh_z[BLOCK_SIZE];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    int num_tiles = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //! Iterate upon chunks/tiles of BLOCK_SIZE
    //! This is possible due to velocity calculations being associative
    for (int tile = 0; tile < num_tiles; tile++) {
        int j_local = threadIdx.x;
        int t_idx = tile * BLOCK_SIZE + j_local;
        
        //! Load that chunk into shared memory for all threads to access
        if (t_idx < bodies_per_system) {
            sh_x[threadIdx.x] = __ldg(&galaxy.x[system_offset + t_idx]);
            sh_y[threadIdx.x] = __ldg(&galaxy.y[system_offset + t_idx]);
            sh_z[threadIdx.x] = __ldg(&galaxy.z[system_offset + t_idx]);
        } else {
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_z[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        //! Each thread computes its data
        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
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

    galaxy.vx[global_idx] += dt * Fx;
    galaxy.vy[global_idx] += dt * Fy;
    galaxy.vz[global_idx] += dt * Fz;
}

__global__ void integrate_positions_kernel(GalaxySoA galaxy, int bodies_per_system, float dt, int my_system) {
    //! Same indexing as forces kernel
    int system_idx = my_system; 
    int body_local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int system_offset = system_idx * bodies_per_system;
    
    int global_idx = system_offset + body_local_idx;

    galaxy.x[global_idx] += galaxy.vx[global_idx] * dt;
    galaxy.y[global_idx] += galaxy.vy[global_idx] * dt;
    galaxy.z[global_idx] += galaxy.vz[global_idx] * dt;
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

cudaStream_t *streams;
cudaStream_t *streams_all[GPU_MAX];

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *data) {
    
    int totalBodies = bodies_per_system * num_systems;

    //! Allocate host memory for body information
    systemsHost.x = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.y = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.z = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.vx = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.vy = (float *) malloc(totalBodies * sizeof(float));
    systemsHost.vz = (float *) malloc(totalBodies * sizeof(float));

    //! Transform AoS (Body *) to SoA (GalaxySoA *)
    for (int curBody = 0; curBody < totalBodies; curBody++) {
        systemsHost.x[curBody] = data[curBody].x;
        systemsHost.y[curBody] = data[curBody].y;
        systemsHost.z[curBody] = data[curBody].z;
        systemsHost.vx[curBody] = data[curBody].vx;
        systemsHost.vy[curBody] = data[curBody].vy;
        systemsHost.vz[curBody] = data[curBody].vz;
    }
    
    int gpu_num = 0;
    cudaGetDeviceCount(&gpu_num);
    if (gpu_num == 0) return 0.0;
    //CUDA_CHECK_LAST_ERROR();

    int gpu_used = gpu_num > GPU_MAX ? GPU_MAX : gpu_num; 
    printf("Running on %d GPUs.\n", gpu_used);

    for (int g = 0; g < gpu_num; g++) cudaInitDevice(g, 0, 0);

    create_timer();
    start_timer();

    #pragma omp parallel num_threads(gpu_num)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);
        //CUDA_CHECK_LAST_ERROR();
        
        int systems_per_gpu = (num_systems + gpu_used - 1) / gpu_used;
        int start_sys = gpu_id * systems_per_gpu;
        int end_sys = (start_sys + systems_per_gpu < num_systems) ? start_sys + systems_per_gpu : num_systems;
        int my_system_count = end_sys - start_sys;
        int bytes = my_system_count * bodies_per_system * sizeof(float);
        
        if (my_system_count > 0) {
            
            //! Allocate cuda memory for kernel calls for this' gpu workload
            cudaMalloc(&(systemsDevice[gpu_id].x), bytes);
            //CUDA_CHECK_LAST_ERROR();
            cudaMalloc(&(systemsDevice[gpu_id].y), bytes);
            //CUDA_CHECK_LAST_ERROR();
            cudaMalloc(&(systemsDevice[gpu_id].z), bytes);
            //CUDA_CHECK_LAST_ERROR();
            cudaMalloc(&(systemsDevice[gpu_id].vx), bytes);
            //CUDA_CHECK_LAST_ERROR();
            cudaMalloc(&(systemsDevice[gpu_id].vy), bytes);
            //CUDA_CHECK_LAST_ERROR();
            cudaMalloc(&(systemsDevice[gpu_id].vz), bytes);
            //CUDA_CHECK_LAST_ERROR();

            streams_all[gpu_id] = (cudaStream_t*)malloc(my_system_count * sizeof(cudaStream_t));
            for (int i = 0; i < my_system_count; i++) cudaStreamCreate(&streams_all[gpu_id][i]);

            cudaMemcpy(systemsDevice[gpu_id].x, &(systemsHost.x[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].y, &(systemsHost.y[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].z, &(systemsHost.z[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].vx, &(systemsHost.vx[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].vy, &(systemsHost.vy[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].vz, &(systemsHost.vz[start_sys * bodies_per_system]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            
            dim3 threads(BLOCK_SIZE);
            dim3 grid((bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE);

            for (int iter = 0; iter < nIters; iter++) {
                for (int curSystem = 0; curSystem < my_system_count; curSystem++) {
                    calculate_forces_kernel<<<grid, threads, 0, streams_all[gpu_id][curSystem]>>>(
                        systemsDevice[gpu_id],
                        bodies_per_system, 
                        dt,
                        curSystem
                    );
                    
                    integrate_positions_kernel<<<grid, threads, 0, streams_all[gpu_id][curSystem]>>>(
                        systemsDevice[gpu_id],
                        bodies_per_system, 
                        dt,
                        curSystem
                    );
                }
            }
            
            cudaDeviceSynchronize();
            //CUDA_CHECK_LAST_ERROR();

            cudaMemcpy(&(systemsHost.x[start_sys * bodies_per_system]), systemsDevice[gpu_id].x, bytes, cudaMemcpyDeviceToHost);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(&(systemsHost.y[start_sys * bodies_per_system]), systemsDevice[gpu_id].y, bytes, cudaMemcpyDeviceToHost);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(&(systemsHost.z[start_sys * bodies_per_system]), systemsDevice[gpu_id].z, bytes, cudaMemcpyDeviceToHost);
            //CUDA_CHECK_LAST_ERROR();
        }
    }

    stop_timer();
    double total_time = (double) get_timer_ms() / 1000.0f;

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

    for (int g = 0; g < gpu_num; g++) {
        cudaSetDevice(g);

        int systems_per_gpu = (num_systems + gpu_num - 1) / gpu_num;
        int start_sys = g * systems_per_gpu;
        int end_sys = min(start_sys + systems_per_gpu, num_systems);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (my_systems > 0) {
            for (int i = 0; i < my_systems; i++) {
                cudaStreamDestroy(streams_all[g][i]);
            }
            free(streams_all[g]);
        }
    }
    cleanUp();

    return total_time;
}