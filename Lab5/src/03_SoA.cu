#include "helpers.h"
#include "gputimer.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
__global__ void bodyForceKernel(GalaxySoA soa, float dt, int n, int sys_idx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;

    int offset = sys_idx * n + i;
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f, dx, dy, dz, distSqr, invDist, invDist3;
    float myX = soa.x[offset];
    float myY = soa.y[offset];
    float myZ = soa.z[offset];

    for (int j = 0; j < n; j++) {
        int target = sys_idx * n + j;

        dx = soa.x[target] - myX;
        dy = soa.y[target] - myY;
        dz = soa.z[target] - myZ;
        
        distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        
        invDist = rsqrtf(distSqr); 
        invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    soa.vx[offset] += dt * Fx;
    soa.vy[offset] += dt * Fy;
    soa.vz[offset] += dt * Fz;
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
__global__ void integrateKernel(GalaxySoA soa, float dt, int n, int sys_idx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = sys_idx * n + i;

    if (i < n) {
        soa.x[offset] += soa.vx[offset] * dt;
        soa.y[offset] += soa.vy[offset] * dt;
        soa.z[offset] += soa.vz[offset] * dt;
    }
}

GalaxySoA d_data;
cudaStream_t *streams;

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *h_data) {
    int total_bodies = num_systems * bodies_per_system;
    size_t float_size = total_bodies * sizeof(float);;
    double time;

    create_timer();
    start_timer();
    cudaMalloc(&d_data.x, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_data.y, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_data.z, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_data.vx, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_data.vy, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_data.vz, float_size);
    CUDA_CHECK_LAST_ERROR();

    streams = (cudaStream_t *)malloc(num_systems * sizeof(cudaStream_t));
    float *tmp_x = (float*)malloc(float_size);
    float *tmp_y = (float*)malloc(float_size);
    float *tmp_z = (float*)malloc(float_size);
    float *tmp_vx = (float*)malloc(float_size);
    float *tmp_vy = (float*)malloc(float_size);
    float *tmp_vz = (float*)malloc(float_size);

    for (int i = 0; i < num_systems; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < total_bodies; i++) {
        tmp_x[i] = h_data[i].x;
        tmp_y[i] = h_data[i].y;
        tmp_z[i] = h_data[i].z;
        tmp_vx[i] = h_data[i].vx;
        tmp_vy[i] = h_data[i].vy;
        tmp_vz[i] = h_data[i].vz;
    }

    for (int i = 0; i < num_systems; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < total_bodies; i++) {
        tmp_x[i] = h_data[i].x;
        tmp_y[i] = h_data[i].y;
        tmp_z[i] = h_data[i].z;
        tmp_vx[i] = h_data[i].vx;
        tmp_vy[i] = h_data[i].vy;
        tmp_vz[i] = h_data[i].vz;
    }

    cudaMemcpy(d_data.x, tmp_x, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_data.y, tmp_y, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_data.z, tmp_z, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_data.vx, tmp_vx, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_data.vy, tmp_vy, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_data.vz, tmp_vz, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    int iter, sys;
    int gridSize = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (iter = 1; iter <= nIters; iter++) {
        for (sys = 0; sys < num_systems; sys++) {
            bodyForceKernel<<<gridSize, BLOCK_SIZE, 0, streams[sys]>>>(d_data, dt, bodies_per_system, sys);
            integrateKernel<<<gridSize, BLOCK_SIZE, 0, streams[sys]>>>(d_data, dt, bodies_per_system, sys);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(tmp_x, d_data.x, float_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(tmp_y, d_data.y, float_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(tmp_z, d_data.z, float_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    for (int i = 0; i < total_bodies; i++) {
        h_data[i].x = tmp_x[i];
        h_data[i].y = tmp_y[i];
        h_data[i].z = tmp_z[i];
    }
    stop_timer();
    time = (double) get_timer_ms() / 1000.0f;

    for (int i = 0; i < num_systems; i++) {
        cudaStreamDestroy(streams[i]);
    }

    free(tmp_x); free(tmp_y); free(tmp_z);
    free(tmp_vx); free(tmp_vy); free(tmp_vz);

    cleanUp();

    return time;
}

void cleanUp() {
    destroy_timer();
    cudaFree(d_data.x);
    cudaFree(d_data.y);
    cudaFree(d_data.z);
    cudaFree(d_data.vx);
    cudaFree(d_data.vy);
    cudaFree(d_data.vz);
    free(streams);
    cudaDeviceReset();
}