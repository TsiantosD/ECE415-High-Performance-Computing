#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <time.h>
#include "helpers.h"
#include "timer.h"

//! Upper limit
#define NUM_GPU 8
#define BLOCK_SIZE 32   //TODO think why this <<< better (more computes/thread better?)

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

__global__ static void bodyForceKernel(GalaxySoA device_system, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    
    const float xi = device_system.x[i];
    const float yi = device_system.y[i];
    const float zi = device_system.z[i];

    // TODO memory tiling ?
    for (int j = 0; j < n; j++) {
        const float dx = device_system.x[j] - xi;
        const float dy = device_system.y[j] - yi;
        const float dz = device_system.z[j] - zi;
        const float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        //! CUDA reverse sqrt (no need this time Jason i gotchu)
        const float invDist = rsqrtf(distSqr); 
        const float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    device_system.vx[i] += dt * Fx;
    device_system.vy[i] += dt * Fy;
    device_system.vz[i] += dt * Fz;

    __syncthreads();
    
    device_system.x[i] += device_system.vx[i] * dt;
    device_system.y[i] += device_system.vy[i] * dt;
    device_system.z[i] += device_system.vz[i] * dt;
}

extern "C" double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, 
                          const float dt, Body *data) {

    GalaxySoA *systems = (GalaxySoA *) malloc(num_systems * sizeof(GalaxySoA));
    for (int s = 0; s < num_systems; s++) {
        systems[s].x = (float *) malloc(bodies_per_system * sizeof(float));
        systems[s].y = (float *) malloc(bodies_per_system * sizeof(float));
        systems[s].z = (float *) malloc(bodies_per_system * sizeof(float));
        systems[s].vx = (float *) malloc(bodies_per_system * sizeof(float));
        systems[s].vy = (float *) malloc(bodies_per_system * sizeof(float));
        systems[s].vz = (float *) malloc(bodies_per_system * sizeof(float));

        for (int i = 0; i < bodies_per_system; i++) {
            int idx = s * bodies_per_system + i;
            systems[s].x[i] = data[idx].x;
            systems[s].y[i] = data[idx].y;
            systems[s].z[i] = data[idx].z;
            systems[s].vx[i] = data[idx].vx;
            systems[s].vy[i] = data[idx].vy;
            systems[s].vz[i] = data[idx].vz;
        }
    }

    printf("Running CUDA simulation on 4 GPUs...\n");

    StartTimer();

    //! Ask for NUM_GPU threads
    //TODO maybe this is not that smart?
    omp_set_num_threads(NUM_GPU);

    #pragma omp parallel
    {
        //TODO OpenMP stuff
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int gpu_id = thread_id % NUM_GPU;
        
        //! Set working gpu for this thread
        cudaError_t err = cudaSetDevice(gpu_id);

        //! Try for NUM_GPUs print error msg for failed ones 
        //! (work will be done by distributed to however many are available)
        if (err != cudaSuccess) {
            fprintf(stderr, "Thread %d failed to set device %d\n", thread_id, gpu_id);
        }

        //! Split workload for for GPUs (thread_id is the corresponding GPU)
        int chunk_size = num_systems / num_threads;
        int start_sys = thread_id * chunk_size;
        int end_sys = start_sys + chunk_size;
        
        //! Handle remainder for last GPU if systems arent mult of chunk_size
        if (thread_id == num_threads - 1) end_sys = num_systems;

        for (int s = start_sys; s < end_sys; s++) {
            float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
            size_t bytes = bodies_per_system * sizeof(float);
            
            //! Allocate Device Memory
            cudaMalloc(&d_x, bytes);
            cudaMalloc(&d_y, bytes);
            cudaMalloc(&d_z, bytes);
            cudaMalloc(&d_vx, bytes);
            cudaMalloc(&d_vy, bytes);
            cudaMalloc(&d_vz, bytes);

            //! Copy Host -> Device
            cudaMemcpy(d_x, systems[s].x, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, systems[s].y, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_z, systems[s].z, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vx, systems[s].vx, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vy, systems[s].vy, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vz, systems[s].vz, bytes, cudaMemcpyHostToDevice);

            GalaxySoA device_system;
            device_system.x  = d_x;
            device_system.y  = d_y;
            device_system.z  = d_z;
            device_system.vx = d_vx;
            device_system.vy = d_vy;
            device_system.vz = d_vz;

            int grid_size = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int iter = 0; iter < nIters; iter++) {
                bodyForceKernel<<<grid_size, BLOCK_SIZE>>> (device_system, dt, bodies_per_system);
            }

            //! Copy Device -> Host
            cudaMemcpy(systems[s].x, d_x, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(systems[s].y, d_y, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(systems[s].z, d_z, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(systems[s].vx, d_vx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(systems[s].vy, d_vy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(systems[s].vz, d_vz, bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_x);
            cudaFree(d_y);
            cudaFree(d_z);
            cudaFree(d_vx);
            cudaFree(d_vy);
            cudaFree(d_vz);
        }
    } 

    double totalTime = GetTimer() / 1000.0;

    for (int s = 0; s < num_systems; s++) {
        // Move data back to data array
        for (int i = 0; i < bodies_per_system; i++) {
            int idx = s * bodies_per_system + i;
            data[idx].x  = systems[s].x[i];
            data[idx].y  = systems[s].y[i];
            data[idx].z  = systems[s].z[i];
            data[idx].vx = systems[s].vx[i];
            data[idx].vy = systems[s].vy[i];
            data[idx].vz = systems[s].vz[i];
        }
    }

    for (int s = 0; s < num_systems; s++) {
        free(systems[s].x); 
        free(systems[s].y); 
        free(systems[s].z);
        free(systems[s].vx); 
        free(systems[s].vy);
        free(systems[s].vz);
    }
    free(systems);
    cudaDeviceReset();

    return totalTime;
}
