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

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
__global__ void __launch_bounds__(BLOCK_SIZE, 2) bodyForceKernel(GalaxySoA soa, float dt, int n, int sys_idx) {
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

    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    for (int tile = 0; tile < num_tiles; tile++) {
        int j_local = threadIdx.x;
        int j_global = tile * BLOCK_SIZE + j_local;

        if (j_global < n) {
            shX[j_local] = __ldg(&soa.x[system_offset + j_global]);
            shY[j_local] = __ldg(&soa.y[system_offset + j_global]);
            shZ[j_local] = __ldg(&soa.z[system_offset + j_global]);
        } else {
            shX[j_local] = 0.0f;
            shY[j_local] = 0.0f;
            shZ[j_local] = 0.0f;
        }

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
    size_t float_size = total_bodies * sizeof(float);
    double time;

    // Απλή malloc (εφόσον τα transfers είναι αμελητέα)
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
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    // Warm-up Contexts (Πολύ σημαντικό για το σωστό timing)
    for (int g = 0; g < num_gpus; g++) { cudaSetDevice(g); cudaFree(0); }

    create_timer();
    start_timer();

    #pragma omp parallel num_threads(num_gpus)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);

        int systems_per_gpu = (num_systems + num_gpus - 1) / num_gpus;
        int start_sys = tid * systems_per_gpu;
        int end_sys = min(start_sys + systems_per_gpu, num_systems);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (my_systems > 0) {
            size_t my_bytes = my_systems * bodies_per_system * sizeof(float);
            int offset = start_sys * bodies_per_system;

            // Allocation στην GPU
            cudaMalloc(&d_data_all[tid].x, my_bytes);
            cudaMalloc(&d_data_all[tid].y, my_bytes);
            cudaMalloc(&d_data_all[tid].z, my_bytes);
            cudaMalloc(&d_data_all[tid].vx, my_bytes);
            cudaMalloc(&d_data_all[tid].vy, my_bytes);
            cudaMalloc(&d_data_all[tid].vz, my_bytes);

            streams_all[tid] = (cudaStream_t*)malloc(my_systems * sizeof(cudaStream_t));
            for (int i = 0; i < my_systems; i++) cudaStreamCreate(&streams_all[tid][i]);

            // Host to Device
            cudaMemcpy(d_data_all[tid].x, &tmp_x[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].y, &tmp_y[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].z, &tmp_z[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].vx, &tmp_vx[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].vy, &tmp_vy[offset], my_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_data_all[tid].vz, &tmp_vz[offset], my_bytes, cudaMemcpyHostToDevice);

            // Υπολογισμός Grid: Κάθε block καλύπτει COARSENING * BLOCK_SIZE σώματα
            int gridSize = (bodies_per_system + (BLOCK_SIZE * COARSENING) - 1) / (BLOCK_SIZE * COARSENING);
            int gridInt = (bodies_per_system + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int iter = 0; iter < nIters; iter++) {
                for (int s = 0; s < my_systems; s++) {
                    bodyForceKernel<<<gridSize, BLOCK_SIZE, 0, streams_all[tid][s]>>>(d_data_all[tid], dt, bodies_per_system, s);
                    integrateKernel<<<gridInt, BLOCK_SIZE, 0, streams_all[tid][s]>>>(d_data_all[tid], dt, bodies_per_system, s);
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

    for (int i = 0; i < total_bodies; i++) {
        h_data[i].x = tmp_x[i]; h_data[i].y = tmp_y[i]; h_data[i].z = tmp_z[i];
    }

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);

        // Πρέπει να υπολογίσουμε ξανά πόσα συστήματα είχε αυτή η GPU
        int systems_per_gpu = (num_systems + num_gpus - 1) / num_gpus;
        int start_sys = g * systems_per_gpu;
        int end_sys = min(start_sys + systems_per_gpu, num_systems);
        int my_systems = (start_sys < num_systems) ? (end_sys - start_sys) : 0;

        if (my_systems > 0) {
            // Καταστροφή των streams της συγκεκριμένης GPU
            for (int i = 0; i < my_systems; i++) {
                cudaStreamDestroy(streams_all[g][i]);
            }
            free(streams_all[g]);

            // Αποδέσμευση της μνήμης της συγκεκριμένης GPU
            cudaFree(d_data_all[g].x);
            cudaFree(d_data_all[g].y);
            cudaFree(d_data_all[g].z);
            cudaFree(d_data_all[g].vx);
            cudaFree(d_data_all[g].vy);
            cudaFree(d_data_all[g].vz);
        }
    }

    // Ελευθέρωση μνήμης Host
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