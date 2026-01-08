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
    int offset = sys_idx * n + i;
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f, dx, dy, dz, distSqr, invDist, invDist3;
    float myX = soa.x[offset];
    float myY = soa.y[offset];
    float myZ = soa.z[offset];

    if (i < n) {
        for (int j = 0; j < n; j++) {
            float dx = soa.x[target] - myX;
            float dy = soa.y[target] - myY;
            float dz = soa.z[target] - myZ;
            
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
    cudaMalloc(&d_soa.x, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_soa.y, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_soa.z, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_soa.vx, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_soa.vy, float_size);
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc(&d_soa.vz, float_size);
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

    cudaMemcpy(d_soa.x, tmp_x, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_soa.y, tmp_y, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_soa.z, tmp_z, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_soa.vx, tmp_vx, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_soa.vy, tmp_vy, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_soa.vz, tmp_vz, float_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    Body *system_ptr;
    int iter, sys;
    int gridSize = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (iter = 1; iter <= nIters; iter++) {
        for (sys = 0; sys < num_systems; sys++) {
	    system_ptr = &d_data[sys * bodies_per_system];
            bodyForceKernel<<<gridSize, BLOCK_SIZE, 0, streams[sys]>>>(d_soa, dt, bodies_per_system, sys);
            integrateKernel<<<gridSize, BLOCK_SIZE, 0, streams[sys]>>>(d_soa, dt, bodies_per_system, sys);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(tmp_x, d_soa.x, float_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(tmp_y, d_soa.y, float_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(tmp_z, d_soa.z, float_size, cudaMemcpyDeviceToHost);
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
    cudaFree(d_data);
    free(streams);
    cudaDeviceReset();
}