#include <omp.h>
#include <sys/time.h>
#include "helpers.h"
#include "gputimer.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define COARSENING 8

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
    int i3 = i2 + blockDim.x;
    int i4 = i3 + blockDim.x;
    int i5 = i4 + blockDim.x;
    int i6 = i5 + blockDim.x;
    int i7 = i6 + blockDim.x;
    int i8 = i7 + blockDim.x;

    int system_offset = sys_idx * n_padded;

    float myX1 = __ldg(&soa.x[system_offset + i1]); float myY1 = __ldg(&soa.y[system_offset + i1]); float myZ1 = __ldg(&soa.z[system_offset + i1]);
    float myX2 = __ldg(&soa.x[system_offset + i2]); float myY2 = __ldg(&soa.y[system_offset + i2]); float myZ2 = __ldg(&soa.z[system_offset + i2]);
    float myX3 = __ldg(&soa.x[system_offset + i3]); float myY3 = __ldg(&soa.y[system_offset + i3]); float myZ3 = __ldg(&soa.z[system_offset + i3]);
    float myX4 = __ldg(&soa.x[system_offset + i4]); float myY4 = __ldg(&soa.y[system_offset + i4]); float myZ4 = __ldg(&soa.z[system_offset + i4]);
    float myX5 = __ldg(&soa.x[system_offset + i5]); float myY5 = __ldg(&soa.y[system_offset + i5]); float myZ5 = __ldg(&soa.z[system_offset + i5]);
    float myX6 = __ldg(&soa.x[system_offset + i6]); float myY6 = __ldg(&soa.y[system_offset + i6]); float myZ6 = __ldg(&soa.z[system_offset + i6]);
    float myX7 = __ldg(&soa.x[system_offset + i7]); float myY7 = __ldg(&soa.y[system_offset + i7]); float myZ7 = __ldg(&soa.z[system_offset + i7]);
    float myX8 = __ldg(&soa.x[system_offset + i8]); float myY8 = __ldg(&soa.y[system_offset + i8]); float myZ8 = __ldg(&soa.z[system_offset + i8]);

    float Fx1 = 0, Fy1 = 0, Fz1 = 0;
    float Fx2 = 0, Fy2 = 0, Fz2 = 0;
    float Fx3 = 0, Fy3 = 0, Fz3 = 0;
    float Fx4 = 0, Fy4 = 0, Fz4 = 0;
    float Fx5 = 0, Fy5 = 0, Fz5 = 0;
    float Fx6 = 0, Fy6 = 0, Fz6 = 0;
    float Fx7 = 0, Fy7 = 0, Fz7 = 0;
    float Fx8 = 0, Fy8 = 0, Fz8 = 0;

    __shared__ float shX[BLOCK_SIZE], shY[BLOCK_SIZE], shZ[BLOCK_SIZE];
    int num_tiles = n_padded / BLOCK_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int j_local = threadIdx.x;

        shX[j_local] = __ldg(&soa.x[system_offset + tile * BLOCK_SIZE + j_local]);
        shY[j_local] = __ldg(&soa.y[system_offset + tile * BLOCK_SIZE + j_local]);
        shZ[j_local] = __ldg(&soa.z[system_offset + tile * BLOCK_SIZE + j_local]);

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            float jX = shX[j]; float jY = shY[j]; float jZ = shZ[j];

            float dx1 = shX[j] - myX1; float dy1 = shY[j] - myY1; float dz1 = shZ[j] - myZ1;
            float d2_1 = fmaf(dx1, dx1, fmaf(dy1, dy1, fmaf(dz1, dz1, SOFTENING)));
            float inv1 = rsqrtf(d2_1); float inv3_1 = inv1 * inv1 * inv1;
            Fx1 = fmaf(dx1, inv3_1, Fx1); Fy1 = fmaf(dy1, inv3_1, Fy1); Fz1 = fmaf(dz1, inv3_1, Fz1);

            float dx2 = shX[j] - myX2; float dy2 = shY[j] - myY2; float dz2 = shZ[j] - myZ2;
            float d2_2 = fmaf(dx2, dx2, fmaf(dy2, dy2, fmaf(dz2, dz2, SOFTENING)));
            float inv2 = rsqrtf(d2_2); float inv3_2 = inv2 * inv2 * inv2;
            Fx2 = fmaf(dx2, inv3_2, Fx2); Fy2 = fmaf(dy2, inv3_2, Fy2); Fz2 = fmaf(dz2, inv3_2, Fz2);

            float dx3 = jX - myX3; float dy3 = jY - myY3; float dz3 = jZ - myZ3;
            float inv3_3 = rsqrtf(fmaf(dx3, dx3, fmaf(dy3, dy3, fmaf(dz3, dz3, SOFTENING))));
            inv3_3 = inv3_3 * inv3_3 * inv3_3;
            Fx3 = fmaf(dx3, inv3_3, Fx3); Fy3 = fmaf(dy3, inv3_3, Fy3); Fz3 = fmaf(dz3, inv3_3, Fz3);

            float dx4 = jX - myX4; float dy4 = jY - myY4; float dz4 = jZ - myZ4;
            float inv3_4 = rsqrtf(fmaf(dx4, dx4, fmaf(dy4, dy4, fmaf(dz4, dz4, SOFTENING))));
            inv3_4 = inv3_4 * inv3_4 * inv3_4;
            Fx4 = fmaf(dx4, inv3_4, Fx4); Fy4 = fmaf(dy4, inv3_4, Fy4); Fz4 = fmaf(dz4, inv3_4, Fz4);

            float dx5 = jX - myX5; float dy5 = jY - myY5; float dz5 = jZ - myZ5;
            float inv3_5 = rsqrtf(fmaf(dx5, dx5, fmaf(dy5, dy5, fmaf(dz5, dz5, SOFTENING))));
            inv3_5 = inv3_5 * inv3_5 * inv3_5;
            Fx5 = fmaf(dx5, inv3_5, Fx5); Fy5 = fmaf(dy5, inv3_5, Fy5); Fz5 = fmaf(dz5, inv3_5, Fz5);

            float dx6 = jX - myX6; float dy6 = jY - myY6; float dz6 = jZ - myZ6;
            float inv3_6 = rsqrtf(fmaf(dx6, dx6, fmaf(dy6, dy6, fmaf(dz6, dz6, SOFTENING))));
            inv3_6 = inv3_6 * inv3_6 * inv3_6;
            Fx6 = fmaf(dx6, inv3_6, Fx6); Fy6 = fmaf(dy6, inv3_6, Fy6); Fz6 = fmaf(dz6, inv3_6, Fz6);

            float dx7 = jX - myX7; float dy7 = jY - myY7; float dz7 = jZ - myZ7;
            float inv3_7 = rsqrtf(fmaf(dx7, dx7, fmaf(dy7, dy7, fmaf(dz7, dz7, SOFTENING))));
            inv3_7 = inv3_7 * inv3_7 * inv3_7;
            Fx7 = fmaf(dx7, inv3_7, Fx7); Fy7 = fmaf(dy7, inv3_7, Fy7); Fz7 = fmaf(dz7, inv3_7, Fz7);

            float dx8 = jX - myX8; float dy8 = jY - myY8; float dz8 = jZ - myZ8;
            float inv3_8 = rsqrtf(fmaf(dx8, dx8, fmaf(dy8, dy8, fmaf(dz8, dz8, SOFTENING))));
            inv3_8 = inv3_8 * inv3_8 * inv3_8;
            Fx8 = fmaf(dx8, inv3_8, Fx8); Fy8 = fmaf(dy8, inv3_8, Fy8); Fz8 = fmaf(dz8, inv3_8, Fz8);
        }
        __syncthreads();
    }

    soa.vx[system_offset + i1] = fmaf(d_dt, Fx1, soa.vx[system_offset + i1]);
    soa.vy[system_offset + i1] = fmaf(d_dt, Fy1, soa.vy[system_offset + i1]);
    soa.vz[system_offset + i1] = fmaf(d_dt, Fz1, soa.vz[system_offset + i1]);

    soa.vx[system_offset + i2] = fmaf(d_dt, Fx2, soa.vx[system_offset + i2]);
    soa.vy[system_offset + i2] = fmaf(d_dt, Fy2, soa.vy[system_offset + i2]);
    soa.vz[system_offset + i2] = fmaf(d_dt, Fz2, soa.vz[system_offset + i2]);

    soa.vx[system_offset + i3] = fmaf(d_dt, Fx3, soa.vx[system_offset + i3]);
    soa.vy[system_offset + i3] = fmaf(d_dt, Fy3, soa.vy[system_offset + i3]);
    soa.vz[system_offset + i3] = fmaf(d_dt, Fz3, soa.vz[system_offset + i3]);

    soa.vx[system_offset + i4] = fmaf(d_dt, Fx4, soa.vx[system_offset + i4]);
    soa.vy[system_offset + i4] = fmaf(d_dt, Fy4, soa.vy[system_offset + i4]);
    soa.vz[system_offset + i4] = fmaf(d_dt, Fz4, soa.vz[system_offset + i4]);

    soa.vx[system_offset + i5] = fmaf(d_dt, Fx5, soa.vx[system_offset + i5]);
    soa.vy[system_offset + i5] = fmaf(d_dt, Fy5, soa.vy[system_offset + i5]);
    soa.vz[system_offset + i5] = fmaf(d_dt, Fz5, soa.vz[system_offset + i5]);

    soa.vx[system_offset + i6] = fmaf(d_dt, Fx6, soa.vx[system_offset + i6]);
    soa.vy[system_offset + i6] = fmaf(d_dt, Fy6, soa.vy[system_offset + i6]);
    soa.vz[system_offset + i6] = fmaf(d_dt, Fz6, soa.vz[system_offset + i6]);

    soa.vx[system_offset + i7] = fmaf(d_dt, Fx7, soa.vx[system_offset + i7]);
    soa.vy[system_offset + i7] = fmaf(d_dt, Fy7, soa.vy[system_offset + i7]);
    soa.vz[system_offset + i7] = fmaf(d_dt, Fz7, soa.vz[system_offset + i7]);

    soa.vx[system_offset + i8] = fmaf(d_dt, Fx8, soa.vx[system_offset + i8]);
    soa.vy[system_offset + i8] = fmaf(d_dt, Fy8, soa.vy[system_offset + i8]);
    soa.vz[system_offset + i8] = fmaf(d_dt, Fz8, soa.vz[system_offset + i8]);
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