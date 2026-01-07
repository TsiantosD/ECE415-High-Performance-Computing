#include "helpers.h"
#include "gputimer.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

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

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *h_data) {
    int total_bodies = num_systems * bodies_per_system;
    size_t data_size = total_bodies * sizeof(Body);
    double time;

    create_timer();
    start_timer();
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
    cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    stop_timer();
    time = (double) get_timer_ms() / 1000.0f;

    cleanUp();

    return time;
}

void cleanUp() {
    destroy_timer();
    cudaFree(d_data);
    cudaDeviceReset();
}
