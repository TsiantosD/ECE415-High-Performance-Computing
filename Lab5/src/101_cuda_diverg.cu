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
#define COARSENING 4

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

GalaxySoA systemsHost;
GalaxySoA systemsDevice[GPU_MAX];

//TODO running this with -Xptxas shows register usage going up 
//TODO although occupancy goes down this enables faster threads -> speedup
//TODO 2 is faster and 8 is 100% + slower
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) calculate_forces_kernel(GalaxySoA galaxy, int bodies_padded, float dt, int my_system) {
    int system_idx = my_system;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int system_offset = system_idx * bodies_padded;

    //! Initialise dyamic coarsening variables (uses local mem) 
    float my_x[COARSENING], my_y[COARSENING], my_z[COARSENING];
    float Fx[COARSENING], Fy[COARSENING], Fz[COARSENING];

    //! Select COARSENING num of bodies to calculate 
    #pragma unroll
    for (int c = 0; c < COARSENING; c++) {
        int idx = (blockIdx.x * COARSENING + c) * THREADS_PER_BLOCK + tid;
        Fx[c] = 0.0f; Fy[c] = 0.0f; Fz[c] = 0.0f;
        int global_idx = system_offset + idx;
        my_x[c] = __ldg(&galaxy.x[global_idx]);
        my_y[c] = __ldg(&galaxy.y[global_idx]);
        my_z[c] = __ldg(&galaxy.z[global_idx]);
    }
    
    __shared__ float sh_x[THREADS_PER_BLOCK];
    __shared__ float sh_y[THREADS_PER_BLOCK];
    __shared__ float sh_z[THREADS_PER_BLOCK];

    int num_tiles = bodies_padded / THREADS_PER_BLOCK;

    for (int tile = 0; tile < num_tiles; tile++) {
        int t_idx = tile * THREADS_PER_BLOCK + tid;
        
        //! No need to check since we added padding
        sh_x[tid] = __ldg(&galaxy.x[system_offset + t_idx]);
        sh_y[tid] = __ldg(&galaxy.y[system_offset + t_idx]);
        sh_z[tid] = __ldg(&galaxy.z[system_offset + t_idx]);
        __syncthreads();

        #pragma unroll 16
        for (int j = 0; j < THREADS_PER_BLOCK; j++) {
            float sx = sh_x[j]; 
            float sy = sh_y[j]; 
            float sz = sh_z[j];

            //! Calculate all COARSENING number of elements assigned to me
            #pragma unroll
            for (int c = 0; c < COARSENING; c++) {
                float dx = sx - my_x[c];
                float dy = sy - my_y[c];
                float dz = sz - my_z[c];
                
                float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                Fx[c] = fmaf(dx, invDist3, Fx[c]);
                Fy[c] = fmaf(dy, invDist3, Fy[c]);
                Fz[c] = fmaf(dz, invDist3, Fz[c]);
            }
        }
        __syncthreads();
    }


    //! Write back all COARSENING number of elements assigned to me
    #pragma unroll
    for (int c = 0; c < COARSENING; c++) {
        int idx = (blockIdx.x * COARSENING + c) * THREADS_PER_BLOCK + tid;
        //! No need to check since we added padding
        int global_idx = system_offset + idx;
        galaxy.vx[global_idx] += dt * Fx[c];
        galaxy.vy[global_idx] += dt * Fy[c];
        galaxy.vz[global_idx] += dt * Fz[c];
    }
}

__global__ void integrate_positions_kernel(GalaxySoA galaxy, int bodies_padded, float dt, int my_system) {
    int system_idx = my_system; 
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int body_local_idx = blockIdx.x * THREADS_PER_BLOCK + tid;
    int system_offset = system_idx * bodies_padded;

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


cudaStream_t *streams;
cudaStream_t *streams_all[GPU_MAX];

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *data) {
    
    int factor = THREADS_PER_BLOCK * COARSENING;
    int n_padded = ((bodies_per_system + factor - 1) / factor) * factor;
    int totalBodiesPadded = n_padded * num_systems;

    //! Use calloc since padding must be 0
    systemsHost.x = (float *) calloc(totalBodiesPadded, sizeof(float));
    systemsHost.y = (float *) calloc(totalBodiesPadded, sizeof(float));
    systemsHost.z = (float *) calloc(totalBodiesPadded, sizeof(float));
    systemsHost.vx = (float *) calloc(totalBodiesPadded, sizeof(float));
    systemsHost.vy = (float *) calloc(totalBodiesPadded, sizeof(float));
    systemsHost.vz = (float *) calloc(totalBodiesPadded, sizeof(float));

    for (int sys = 0; sys < num_systems; sys++) {
        for (int b = 0; b < bodies_per_system; b++) {
            int src = sys * bodies_per_system + b;
            int dst = sys * n_padded + b;
            systemsHost.x[dst] = data[src].x;
            systemsHost.y[dst] = data[src].y;
            systemsHost.z[dst] = data[src].z;
            systemsHost.vx[dst] = data[src].vx;
            systemsHost.vy[dst] = data[src].vy;
            systemsHost.vz[dst] = data[src].vz;
        }
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
        int bytes = my_system_count * n_padded * sizeof(float);
        
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

            cudaMemcpy(systemsDevice[gpu_id].x, &(systemsHost.x[start_sys * n_padded]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].y, &(systemsHost.y[start_sys * n_padded]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].z, &(systemsHost.z[start_sys * n_padded]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].vx, &(systemsHost.vx[start_sys * n_padded]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].vy, &(systemsHost.vy[start_sys * n_padded]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(systemsDevice[gpu_id].vz, &(systemsHost.vz[start_sys * n_padded]), bytes, cudaMemcpyHostToDevice);
            //CUDA_CHECK_LAST_ERROR();
            
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            dim3 gridForce(n_padded / (THREADS_PER_BLOCK * COARSENING));
            dim3 gridIntegrate(n_padded / THREADS_PER_BLOCK);

            for (int iter = 0; iter < nIters; iter++) {
                for (int curSystem = 0; curSystem < my_system_count; curSystem++) {
                    calculate_forces_kernel<<<gridForce, threads, 0, streams_all[gpu_id][curSystem]>>>(
                        systemsDevice[gpu_id],
                        n_padded, 
                        dt,
                        curSystem
                    );
                    
                    integrate_positions_kernel<<<gridIntegrate, threads, 0, streams_all[gpu_id][curSystem]>>>(
                        systemsDevice[gpu_id],
                        n_padded, 
                        dt,
                        curSystem
                    );
                }
            }
                
            cudaDeviceSynchronize();
            //CUDA_CHECK_LAST_ERROR();

            cudaMemcpy(&(systemsHost.x[start_sys * n_padded]), systemsDevice[gpu_id].x, bytes, cudaMemcpyDeviceToHost);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(&(systemsHost.y[start_sys * n_padded]), systemsDevice[gpu_id].y, bytes, cudaMemcpyDeviceToHost);
            //CUDA_CHECK_LAST_ERROR();
            cudaMemcpy(&(systemsHost.z[start_sys * n_padded]), systemsDevice[gpu_id].z, bytes, cudaMemcpyDeviceToHost);
            //CUDA_CHECK_LAST_ERROR();
        }
    }

    stop_timer();
    double total_time = (double) get_timer_ms() / 1000.0f;

    omp_set_num_threads(omp_get_max_threads());

    #pragma omp parallel for
    for (int sys = 0; sys < num_systems; sys++) {
        OMP_PRINT_NUM_THREADS("in write back", sys == 0, omp_get_thread_num());
        for (int b = 0; b < bodies_per_system; b++) {
            //! Skip paddded spaces
            int src = sys * n_padded + b;
            int dst = sys * bodies_per_system + b;
            data[dst].x = systemsHost.x[src];
            data[dst].y = systemsHost.y[src];
            data[dst].z = systemsHost.z[src];
            data[dst].vx = systemsHost.vx[src];
            data[dst].vy = systemsHost.vy[src];
            data[dst].vz = systemsHost.vz[src];
        }
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
    free(streams);
    cleanUp();

    return total_time;
}