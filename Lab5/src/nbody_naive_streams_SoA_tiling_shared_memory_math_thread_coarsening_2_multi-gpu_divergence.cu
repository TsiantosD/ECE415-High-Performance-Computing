#include <omp.h>
#include <sys/time.h>
#include "helpers.h"
#include "gputimer.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define COARSENING 2

#define MAX_GPUS 4

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

GalaxySoA d_data_all[MAX_GPUS];
cudaStream_t *streams_all[MAX_GPUS];

__constant__ float d_dt;

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
__global__ void __launch_bounds__(BLOCK_SIZE, 2) 
bodyForceKernel(GalaxySoA soa, int n_padded, int sys_idx) {

    int i1 = (blockIdx.x * COARSENING) * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x; 
    int system_offset = sys_idx * n_padded;

    float myX1 = __ldg(&soa.x[system_offset + i1]);
    float myY1 = __ldg(&soa.y[system_offset + i1]);
    float myZ1 = __ldg(&soa.z[system_offset + i1]);

    float myX2 = __ldg(&soa.x[system_offset + i2]);
    float myY2 = __ldg(&soa.y[system_offset + i2]);
    float myZ2 = __ldg(&soa.z[system_offset + i2]);

    float Fx1 = 0, Fy1 = 0, Fz1 = 0;
    float Fx2 = 0, Fy2 = 0, Fz2 = 0;

    __shared__ float shX[BLOCK_SIZE], shY[BLOCK_SIZE], shZ[BLOCK_SIZE];

    int num_tiles = n_padded / BLOCK_SIZE; 

    #pragma unroll
    for (int tile = 0; tile < num_tiles; tile++) {
        int j_local = threadIdx.x;
        int j_global = tile * BLOCK_SIZE + j_local;

        shX[j_local] = __ldg(&soa.x[system_offset + j_global]);
        shY[j_local] = __ldg(&soa.y[system_offset + j_global]);
        shZ[j_local] = __ldg(&soa.z[system_offset + j_global]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float dx1 = shX[j] - myX1; float dy1 = shY[j] - myY1; float dz1 = shZ[j] - myZ1;
            float d2_1 = fmaf(dx1, dx1, fmaf(dy1, dy1, fmaf(dz1, dz1, SOFTENING)));
            float inv1 = rsqrtf(d2_1); float inv3_1 = inv1 * inv1 * inv1;
            Fx1 = fmaf(dx1, inv3_1, Fx1); Fy1 = fmaf(dy1, inv3_1, Fy1); Fz1 = fmaf(dz1, inv3_1, Fz1);

            float dx2 = shX[j] - myX2; float dy2 = shY[j] - myY2; float dz2 = shZ[j] - myZ2;
            float d2_2 = fmaf(dx2, dx2, fmaf(dy2, dy2, fmaf(dz2, dz2, SOFTENING)));
            float inv2 = rsqrtf(d2_2); float inv3_2 = inv2 * inv2 * inv2;
            Fx2 = fmaf(dx2, inv3_2, Fx2); Fy2 = fmaf(dy2, inv3_2, Fy2); Fz2 = fmaf(dz2, inv3_2, Fz2);
        }
        __syncthreads();
    }

    soa.vx[system_offset + i1] = fmaf(d_dt, Fx1, soa.vx[system_offset + i1]);
    soa.vy[system_offset + i1] = fmaf(d_dt, Fy1, soa.vy[system_offset + i1]);
    soa.vz[system_offset + i1] = fmaf(d_dt, Fz1, soa.vz[system_offset + i1]);

    soa.vx[system_offset + i2] = fmaf(d_dt, Fx2, soa.vx[system_offset + i2]);
    soa.vy[system_offset + i2] = fmaf(d_dt, Fy2, soa.vy[system_offset + i2]);
    soa.vz[system_offset + i2] = fmaf(d_dt, Fz2, soa.vz[system_offset + i2]);
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
__global__ void integrateKernel(GalaxySoA soa, int n, int sys_idx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = sys_idx * n + i;

    soa.x[offset] += soa.vx[offset] * d_dt;
    soa.y[offset] += soa.vy[offset] * d_dt;
    soa.z[offset] += soa.vz[offset] * d_dt;
}

GalaxySoA d_data;
cudaStream_t *streams;

double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, const float dt, Body *h_data) {
    int factor = BLOCK_SIZE * COARSENING;
    int n_padded = ((bodies_per_system + factor - 1) / factor) * factor;
    int total_bodies_padded = num_systems * n_padded;
    size_t float_size_padded = total_bodies_padded * sizeof(float);
    double time;

    float *tmp_x = (float*)calloc(total_bodies_padded, sizeof(float));
    float *tmp_y = (float*)calloc(total_bodies_padded, sizeof(float));
    float *tmp_z = (float*)calloc(total_bodies_padded, sizeof(float));
    float *tmp_vx = (float*)calloc(total_bodies_padded, sizeof(float));
    float *tmp_vy = (float*)calloc(total_bodies_padded, sizeof(float));
    float *tmp_vz = (float*)calloc(total_bodies_padded, sizeof(float));
    float *tmp_mask = (float*)calloc(total_bodies_padded, sizeof(float));

    for (int s = 0; s < num_systems; s++) {
        for (int b = 0; b < bodies_per_system; b++) {
            int old_idx = s * bodies_per_system + b;
            int new_idx = s * n_padded + b;

            tmp_x[new_idx] = h_data[old_idx].x;
            tmp_y[new_idx] = h_data[old_idx].y;
            tmp_z[new_idx] = h_data[old_idx].z;
            tmp_vx[new_idx] = h_data[old_idx].vx;
            tmp_vy[new_idx] = h_data[old_idx].vy;
            tmp_vz[new_idx] = h_data[old_idx].vz;
            tmp_mask[new_idx] = 1.0f;
        }
    }

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    for (int g = 0; g < num_gpus; g++) { cudaSetDevice(g); cudaFree(0); }

    create_timer();
    start_timer();

    #pragma omp parallel num_threads(num_gpus)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));

        int systems_per_gpu = (num_systems + num_gpus - 1) / num_gpus;
        int start_sys = tid * systems_per_gpu;
        int end_sys = min(start_sys + systems_per_gpu, num_systems);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (my_systems > 0) {
            size_t my_bytes = my_systems * n_padded * sizeof(float);
            int offset = start_sys * n_padded;

            cudaMalloc(&d_data_all[tid].x, my_bytes);
            cudaMalloc(&d_data_all[tid].y, my_bytes);
            cudaMalloc(&d_data_all[tid].z, my_bytes);
            cudaMalloc(&d_data_all[tid].vx, my_bytes);
            cudaMalloc(&d_data_all[tid].vy, my_bytes);
            cudaMalloc(&d_data_all[tid].vz, my_bytes);

            streams_all[tid] = (cudaStream_t*)malloc(my_systems * sizeof(cudaStream_t));
            for (int i = 0; i < my_systems; i++) cudaStreamCreate(&streams_all[tid][i]);

            cudaMemcpy(d_data_all[tid].x, &tmp_x[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].y, &tmp_y[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].z, &tmp_z[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].vx, &tmp_vx[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].vy, &tmp_vy[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].vz, &tmp_vz[offset], my_bytes, cudaMemcpyHostToDevice);

            int gridSize = n_padded / (BLOCK_SIZE * COARSENING);
            int gridInt = n_padded / BLOCK_SIZE;

            for (int iter = 0; iter < nIters; iter++) {
                for (int s = 0; s < my_systems; s++) {
                    bodyForceKernel<<<gridSize, BLOCK_SIZE, 0, streams_all[tid][s]>>>(d_data_all[tid], n_padded, s);
                    integrateKernel<<<gridInt, BLOCK_SIZE, 0, streams_all[tid][s]>>>(d_data_all[tid], n_padded, s);
                }
            }

            cudaDeviceSynchronize();

            // Device to Host
            cudaMemcpy(&tmp_x[offset], d_data_all[tid].x, my_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&tmp_y[offset], d_data_all[tid].y, my_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&tmp_z[offset], d_data_all[tid].z, my_bytes, cudaMemcpyDeviceToHost);
        }
    }

    stop_timer();
    time = (double) get_timer_ms() / 1000.0f;
    destroy_timer();

    for (int s = 0; s < num_systems; s++) {
        for (int b = 0; b < bodies_per_system; b++) {
            int old_idx = s * bodies_per_system + b;
            int new_idx = s * n_padded + b;

            h_data[old_idx].x = tmp_x[new_idx];
            h_data[old_idx].y = tmp_y[new_idx];
            h_data[old_idx].z = tmp_z[new_idx];
        }
    }

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);

        int systems_per_gpu = (num_systems + num_gpus - 1) / num_gpus;
        int start_sys = g * systems_per_gpu;
        int end_sys = min(start_sys + systems_per_gpu, num_systems);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (my_systems > 0) {
            for (int i = 0; i < my_systems; i++) {
                cudaStreamDestroy(streams_all[g][i]);
            }
            free(streams_all[g]);

            cudaFree(d_data_all[g].x);
            cudaFree(d_data_all[g].y);
            cudaFree(d_data_all[g].z);
            cudaFree(d_data_all[g].vx);
            cudaFree(d_data_all[g].vy);
            cudaFree(d_data_all[g].vz);
        }
    }

    free(tmp_x); free(tmp_y); free(tmp_z);
    free(tmp_vx); free(tmp_vy); free(tmp_vz);

    cleanUp();

    return time;
}

void cleanUp() {
    cudaFree(d_data.x);
    cudaFree(d_data.y);
    cudaFree(d_data.z);
    cudaFree(d_data.vx);
    cudaFree(d_data.vy);
    cudaFree(d_data.vz);
    free(streams);
    cudaDeviceReset();
}