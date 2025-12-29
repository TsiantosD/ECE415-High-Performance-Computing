#include <math.h>
#include <cuda_runtime.h>
#include "helpers.h"

#define BLOCK_SIZE 32

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
__global__ static void bodyForceKernel(Body *p, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f, dx, dy, dz, distSqr, invDist, invDist3;
    float myX = p[i].x;
    float myY = p[i].y;
    float myZ = p[i].z;

    if (i < n) {
        for (int j = 0; j < n; j++) {
            dx = p[j].x - myX;
            dy = p[j].y - myY;
            dz = p[j].z - myZ;
            
            distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            
            invDist = rsqrtf(distSqr); 
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
__global__ void integrateKernel(Body *p, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

Body *d_data;
cudaEvent_t start, stop;

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *h_data) {
    int total_bodies = num_systems * bodies_per_system;
    size_t data_size = total_bodies * sizeof(Body);
    float time;

    cudaEventCreate(&start);
    CUDA_CHECK_LAST_ERROR();
    cudaEventCreate(&stop);
    CUDA_CHECK_LAST_ERROR();
    cudaEventRecord(start);
    CUDA_CHECK_LAST_ERROR();

    cudaMalloc((void**)&d_data, data_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    Body *system_ptr;
    int iter, sys;
    int gridSize = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (iter = 1; iter <= nIters; iter++) {
        for (sys = 0; sys < num_systems; sys++) {
	    system_ptr = &d_data[sys * bodies_per_system];
            bodyForceKernel<<<gridSize, BLOCK_SIZE>>>(system_ptr, dt, bodies_per_system);
            integrateKernel<<<gridSize, BLOCK_SIZE>>>(system_ptr, dt, bodies_per_system);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    CUDA_CHECK_LAST_ERROR();

    cleanUp();

    return (double) time / 1000.0f;
}

void cleanUp() {
    cudaFree(d_data);
}
