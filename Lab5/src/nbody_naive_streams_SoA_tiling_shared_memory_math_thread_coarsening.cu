#include "helpers.h"
#include "gputimer.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define COARSENING 32

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
    int i1 = (blockIdx.x * COARSENING) * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x; 
    int system_offset = sys_idx * n;

    float myX1 = 0, myY1 = 0, myZ1 = 0;
    float myX2 = 0, myY2 = 0, myZ2 = 0;
    float Fx1 = 0.0f, Fy1 = 0.0f, Fz1 = 0.0f;
    float Fx2 = 0.0f, Fy2 = 0.0f, Fz2 = 0.0f;

    if (i1 < n) {
        myX1 = __ldg(&soa.x[system_offset + i1]);
        myY1 = __ldg(&soa.y[system_offset + i1]);
        myZ1 = __ldg(&soa.z[system_offset + i1]);
    }
    if (i2 < n) {
        myX2 = __ldg(&soa.x[system_offset + i2]);
        myY2 = __ldg(&soa.y[system_offset + i2]);
        myZ2 = __ldg(&soa.z[system_offset + i2]);
    }

    __shared__ float shX[BLOCK_SIZE];
    __shared__ float shY[BLOCK_SIZE];
    __shared__ float shZ[BLOCK_SIZE];

    int num_tiles = n / BLOCK_SIZE; 
    for (int tile = 0; tile < num_tiles; tile++) {
        int j_local = threadIdx.x;
        int j_global = tile * BLOCK_SIZE + j_local;

        shX[j_local] = __ldg(&soa.x[system_offset + j_global]);
        shY[j_local] = __ldg(&soa.y[system_offset + j_global]);
        shZ[j_local] = __ldg(&soa.z[system_offset + j_global]);

        __syncthreads();

        if (i1 < n) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float dx = shX[j] - myX1; float dy = shY[j] - myY1; float dz = shZ[j] - myZ1;
                float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
                float invDist = rsqrtf(distSqr); float invDist3 = invDist * invDist * invDist;
                Fx1 = fmaf(dx, invDist3, Fx1); Fy1 = fmaf(dy, invDist3, Fy1); Fz1 = fmaf(dz, invDist3, Fz1);
            }
        }

        if (i2 < n) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float dx = shX[j] - myX2; float dy = shY[j] - myY2; float dz = shZ[j] - myZ2;
                float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
                float invDist = rsqrtf(distSqr); float invDist3 = invDist * invDist * invDist;
                Fx2 = fmaf(dx, invDist3, Fx2); Fy2 = fmaf(dy, invDist3, Fy2); Fz2 = fmaf(dz, invDist3, Fz2);
            }
        }
        __syncthreads();
    }

    if (i1 < n) {
        soa.vx[system_offset + i1] += dt * Fx1;
        soa.vy[system_offset + i1] += dt * Fy1;
        soa.vz[system_offset + i1] += dt * Fz1;
    }
    if (i2 < n) {
        soa.vx[system_offset + i2] += dt * Fx2;
        soa.vy[system_offset + i2] += dt * Fy2;
        soa.vz[system_offset + i2] += dt * Fz2;
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
    int gridSize = (bodies_per_system + (BLOCK_SIZE * COARSENING) - 1) / (BLOCK_SIZE * COARSENING);
    for (iter = 1; iter <= nIters; iter++) {
        for (sys = 0; sys < num_systems; sys++) {
            bodyForceKernel<<<gridSize, BLOCK_SIZE, 0, streams[sys]>>>(d_data, dt, bodies_per_system, sys);
            int gridInt = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;
            integrateKernel<<<gridInt, BLOCK_SIZE, 0, streams[sys]>>>(d_data, dt, bodies_per_system, sys);
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