#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#include <omp.h>

#define SOFTENING 0.01f

typedef struct {
    float x, y, z;
    float vx, vy, vz;
} Body;

void bodyForce(Body * restrict p, float dt, int n)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        const float ix = p[i].x;
        const float iy = p[i].y;
        const float iz = p[i].z;

        #pragma omp simd reduction(+:Fx,Fy,Fz)
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - ix;
            float dy = p[j].y - iy;
            float dz = p[j].z - iz;

            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist  = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

void integrate(Body * restrict p, float dt, int n)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(int argc, const char *argv[])
{
    int num_systems = 32;
    int bodies_per_system = 8192;
    int nIters = 400;
    const float dt = 0.01f;

    FILE *fp;
    Body *data;
    float *buf;
    int total_bodies;
    double totalTime;

    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n",
               num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    total_bodies = num_systems * bodies_per_system;
    data = aligned_alloc(64, total_bodies * sizeof(Body));

    if (fp) {
        fread(data, sizeof(Body), total_bodies, fp);
        fclose(fp);
    } else {
        buf = (float *) data;
        for (int i = 0; i < 6 * total_bodies; i++)
            buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    }

    printf("Running optimized OpenMP CPU simulation...\n");

    omp_set_nested(0);
    omp_set_dynamic(0);

    StartTimer();

    // #pragma omp parallel
    // {
        for (int iter = 0; iter < nIters; iter++) {
            #pragma omp parallel for schedule(static)
            for (int sys = 0; sys < num_systems; sys++) {
                Body *system_ptr = &data[sys * bodies_per_system];

                bodyForce(system_ptr, dt, bodies_per_system);
                integrate(system_ptr, dt, bodies_per_system);
            }
        }
    // }

    totalTime = GetTimer() / 1000.0;

    double interactions =
        (double)bodies_per_system *
        (double)bodies_per_system *
        (double)num_systems *
        (double)nIters;

    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Throughput: %.3f Billion Interactions / second\n", 1e-9 * interactions / totalTime);

    printf("Final position [0][0]: %.4f %.4f %.4f\n", data[0].x, data[0].y, data[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n", data[1].x, data[1].y, data[1].z);

    free(data);
    return 0;
}
