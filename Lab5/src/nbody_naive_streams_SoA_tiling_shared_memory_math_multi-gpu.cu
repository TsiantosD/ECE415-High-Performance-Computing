#include <omp.h>

#include "helpers.h"
#include "gputimer.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define COARSENING 2
#define MAX_GPUS 2

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} GalaxySoA;

GalaxySoA d_data_all[MAX_GPUS];
cudaStream_t *streams_all[MAX_GPUS];

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
        myX = __ldg(&soa.x[system_offset + i]);
        myY = __ldg(&soa.y[system_offset + i]);
        myZ = __ldg(&soa.z[system_offset + i]);
    }

    __shared__ float shX[BLOCK_SIZE];
    __shared__ float shY[BLOCK_SIZE];
    __shared__ float shZ[BLOCK_SIZE];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    int num_tiles = n / BLOCK_SIZE; 

    for (int tile = 0; tile < num_tiles; tile++) {

        int j_local = threadIdx.x;
        int j_global = tile * BLOCK_SIZE + j_local;

        shX[j_local] = __ldg(&soa.x[system_offset + j_global]);
        shY[j_local] = __ldg(&soa.y[system_offset + j_global]);
        shZ[j_local] = __ldg(&soa.z[system_offset + j_global]);

        __syncthreads();

        if (i < n) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float dx = shX[j] - myX;
                float dy = shY[j] - myY;
                float dz = shZ[j] - myZ;

                float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));

                float invDist = rsqrtf(distSqr); 
                float invDist3 = invDist * invDist * invDist;

                Fx = fmaf(dx, invDist3, Fx);
                Fy = fmaf(dy, invDist3, Fy);
                Fz = fmaf(dz, invDist3, Fz);
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
    size_t float_size = total_bodies * sizeof(float);
    double time;

    float *tmp_x = (float*)malloc(float_size);
    float *tmp_y = (float*)malloc(float_size);
    float *tmp_z = (float*)malloc(float_size);
    float *tmp_vx = (float*)malloc(float_size);
    float *tmp_vy = (float*)malloc(float_size);
    float *tmp_vz = (float*)malloc(float_size);

    for (int i = 0; i < total_bodies; i++) {
        tmp_x[i] = h_data[i].x; tmp_y[i] = h_data[i].y; tmp_z[i] = h_data[i].z;
        tmp_vx[i] = h_data[i].vx; tmp_vy[i] = h_data[i].vy; tmp_vz[i] = h_data[i].vz;
    }

    int num_gpus;

    cudaGetDeviceCount(&num_gpus);

    if (num_gpus > MAX_GPUS)
        num_gpus = MAX_GPUS;

    create_timer();
    start_timer();

    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);

        int systems_per_gpu = (num_systems + num_gpus - 1) / num_gpus;
        int start_sys = gpu_id * systems_per_gpu;
        int end_sys = min(start_sys + systems_per_gpu, num_systems);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (my_systems > 0) {
            size_t my_float_size = my_systems * bodies_per_system * sizeof(float);
            int body_offset = start_sys * bodies_per_system;

            cudaMalloc(&d_data_all[gpu_id].x, my_float_size);
            cudaMalloc(&d_data_all[gpu_id].y, my_float_size);
            cudaMalloc(&d_data_all[gpu_id].z, my_float_size);
            cudaMalloc(&d_data_all[gpu_id].vx, my_float_size);
            cudaMalloc(&d_data_all[gpu_id].vy, my_float_size);
            cudaMalloc(&d_data_all[gpu_id].vz, my_float_size);

            streams_all[gpu_id] = (cudaStream_t*)malloc(my_systems * sizeof(cudaStream_t));

            for (int i = 0; i < my_systems; i++)
                cudaStreamCreate(&streams_all[gpu_id][i]);

            cudaMemcpy(d_data_all[gpu_id].x, &tmp_x[body_offset], my_float_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[gpu_id].y, &tmp_y[body_offset], my_float_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[gpu_id].z, &tmp_z[body_offset], my_float_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[gpu_id].vx, &tmp_vx[body_offset], my_float_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[gpu_id].vy, &tmp_vy[body_offset], my_float_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[gpu_id].vz, &tmp_vz[body_offset], my_float_size, cudaMemcpyHostToDevice);

            int gridSize = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int iter = 0; iter < nIters; iter++) {
                for (int s = 0; s < my_systems; s++) {
                    bodyForceKernel<<<gridSize, BLOCK_SIZE, 0, streams_all[gpu_id][s]>>>(d_data_all[gpu_id], dt, bodies_per_system, s);
                    integrateKernel<<<gridSize, BLOCK_SIZE, 0, streams_all[gpu_id][s]>>>(d_data_all[gpu_id], dt, bodies_per_system, s);
                }
            }

            cudaDeviceSynchronize();

            cudaMemcpy(&tmp_x[body_offset], d_data_all[gpu_id].x, my_float_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(&tmp_y[body_offset], d_data_all[gpu_id].y, my_float_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(&tmp_z[body_offset], d_data_all[gpu_id].z, my_float_size, cudaMemcpyDeviceToHost);
        }
    }

    for (int i = 0; i < total_bodies; i++) {
        h_data[i].x = tmp_x[i]; h_data[i].y = tmp_y[i]; h_data[i].z = tmp_z[i];
    }

    stop_timer();
    time = (double) get_timer_ms() / 1000.0f;

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);

        int systems_per_gpu = (num_systems + num_gpus - 1) / num_gpus;
        int start_sys = g * systems_per_gpu;
        int end_sys = (start_sys + systems_per_gpu > num_systems) ? num_systems : (start_sys + systems_per_gpu);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (streams_all[g] != NULL) {
            for (int i = 0; i < my_systems; i++) {
                cudaStreamDestroy(streams_all[g][i]);
            }
            free(streams_all[g]);
            streams_all[g] = NULL;
        }

        cudaFree(d_data_all[g].x);
        cudaFree(d_data_all[g].y);
        cudaFree(d_data_all[g].z);
        cudaFree(d_data_all[g].vx);
        cudaFree(d_data_all[g].vy);
        cudaFree(d_data_all[g].vz);

        cudaDeviceReset();
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