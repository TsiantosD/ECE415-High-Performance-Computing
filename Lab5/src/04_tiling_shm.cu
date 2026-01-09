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
    int system_offset = sys_idx * n;

    float myX = 0, myY = 0, myZ = 0;
    if (i < n) {
        myX = soa.x[system_offset + i];
        myY = soa.y[system_offset + i];
        myZ = soa.z[system_offset + i];
    }

    __shared__ float shX[BLOCK_SIZE];
    __shared__ float shY[BLOCK_SIZE];
    __shared__ float shZ[BLOCK_SIZE];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; tile++) {

        int j_local = threadIdx.x;
        int j_global = tile * BLOCK_SIZE + j_local;

        if (j_global < n) {
            shX[j_local] = soa.x[system_offset + j_global];
            shY[j_local] = soa.y[system_offset + j_global];
            shZ[j_local] = soa.z[system_offset + j_global];
        } else {
            shX[j_local] = 0.0f; shY[j_local] = 0.0f; shZ[j_local] = 0.0f;
        }

        __syncthreads();

        if (i < n) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (tile * BLOCK_SIZE + j < n) {
                    float dx = shX[j] - myX;
                    float dy = shY[j] - myY;
                    float dz = shZ[j] - myZ;
                    
                    float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                    float invDist = rsqrtf(distSqr); 
                    float invDist3 = invDist * invDist * invDist;

                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
        }

        __syncthreads();
    }

    if (i < n) {
        int offset = system_offset + i;
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

    float *tmp_x = (float*)malloc(float_size);
    float *tmp_y = (float*)malloc(float_size);
    float *tmp_z = (float*)malloc(float_size);
    float *tmp_vx = (float*)malloc(float_size);
    float *tmp_vy = (float*)malloc(float_size);
    float *tmp_vz = (float*)malloc(float_size);

    streams = (cudaStream_t *)malloc(num_systems * sizeof(cudaStream_t));
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