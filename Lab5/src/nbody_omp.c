#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include "helpers.h"
#include "timer.h"

#define OMP_SCHEDULE_TYPE dynamic
#define SOFTENING 0.01f

volatile int counter = 0; 

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

typedef struct {
    float *restrict x, *restrict y, *restrict z;
    float *restrict vx, *restrict vy, *restrict vz;
} GalaxySoA;

void bodyForce(GalaxySoA *p, float dt, int n) {
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;
        
        const float xi = p->x[i];
        const float yi = p->y[i];
        const float zi = p->z[i];
        
        #pragma omp simd reduction(+:Fx,Fy,Fz)
        for (int j = 0; j < n; j++) {
            const float dx = p->x[j] - xi;
            const float dy = p->y[j] - yi;
            const float dz = p->z[j] - zi;
            const float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const float invDist = 1.0f / sqrtf(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p->vx[i] += dt * Fx;
        p->vy[i] += dt * Fy;
        p->vz[i] += dt * Fz;
    }
}

void integrate(GalaxySoA *p, float dt, int n) {
    int i;
    
    for (i = 0; i < n; i++) {
	    p->x[i] += p->vx[i] * dt;
        p->y[i] += p->vy[i] * dt;
        p->z[i] += p->vz[i] * dt;
    }
}

int main(int argc, const char *argv[])
{
    int num_systems = 32;
    int bodies_per_system = 8192;
    const int nIters = 20;
    const float dt = 0.01f;

    FILE *fp;
    Body *data;
    float *buf;
    int total_bodies;
    double totalTime;

    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        if (fread(&num_systems, sizeof(int), 1, fp) != 1 ||
            fread(&bodies_per_system, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Error: failed to read dataset header\n");
            fclose(fp);
            return EXIT_FAILURE;
        }

       printf("Found dataset: %d systems of %d bodies.\n",
               num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    total_bodies = num_systems * bodies_per_system;
    data = malloc(total_bodies * sizeof(Body));

    if (fp) {
        size_t nread = fread(data, sizeof(Body), (size_t)total_bodies, fp);
        if (nread != (size_t)total_bodies) {
            fprintf(stderr,
                    "Error: expected %d bodies, read %zu\n",
                    total_bodies, nread);
            fclose(fp);
            free(data);
            return EXIT_FAILURE;
        }
       fclose(fp);
    } else {
        buf = (float *) data;
        for (int i = 0; i < 6 * total_bodies; i++)
            buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    }

    GalaxySoA *systems = malloc(num_systems * sizeof(GalaxySoA));

    for (int s = 0; s < num_systems; s++) {
        systems[s].x = malloc(bodies_per_system * sizeof(float));
        systems[s].y = malloc(bodies_per_system * sizeof(float));
        systems[s].z = malloc(bodies_per_system * sizeof(float));
        systems[s].vx = malloc(bodies_per_system * sizeof(float));
        systems[s].vy = malloc(bodies_per_system * sizeof(float));
        systems[s].vz = malloc(bodies_per_system * sizeof(float));

        for (int i = 0; i < bodies_per_system; i++) {
            int idx = s * bodies_per_system + i;
            systems[s].x[i] = data[idx].x;
            systems[s].y[i] = data[idx].y;
            systems[s].z[i] = data[idx].z;
            systems[s].vx[i] = data[idx].vx;
            systems[s].vy[i] = data[idx].vy;
            systems[s].vz[i] = data[idx].vz;
        }
    }

    printf("Running optimized OpenMP CPU simulation...\n");

    omp_set_nested(0);
    omp_set_dynamic(0);

    StartTimer();
    
    for (int iter = 0; iter < nIters; iter++) {
        //PRINT_PROGRESS_RATE(iter, nIters);
        #pragma omp parallel for schedule(OMP_SCHEDULE_TYPE) 
        for (int sys = 0; sys < num_systems; sys++) {
            bodyForce(&systems[sys], dt, bodies_per_system);
            integrate(&systems[sys], dt, bodies_per_system);
        }
    }

    totalTime = GetTimer() / 1000.0;

    double interactions_per_system = (double) bodies_per_system * bodies_per_system;
    double total_interactions = interactions_per_system * num_systems * nIters;

    printf("\nTotal Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);

    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           systems[0].x[0], systems[0].y[0], systems[0].z[0]);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           systems[0].x[1], systems[0].y[1], systems[0].z[1]);

    free(data);
    for (int s = 0; s < num_systems; s++) {
        free(systems[s].x);
        free(systems[s].y);
        free(systems[s].z);
        free(systems[s].vx);
        free(systems[s].vy);
        free(systems[s].vz);
    }
    free(systems);
    return 0;
}