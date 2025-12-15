#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "timer.h"

#define SOFTENING 0.01f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

//! setting all to dynamic creates error compared to cpu but is 3x faster

void bodyForce(Body * p, float dt, int n) {
    #pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
        Body *elementPtr = &(p[i]);
	    float Fx = 0.0f;
    	float Fy = 0.0f;
    	float Fz = 0.0f;

        #pragma omp simd reduction(+:Fx,Fy,Fz)
    	for (int j = 0; j < n; j++) {
            Body *elementPtrJ = &(p[j]);
	        const float dx = elementPtrJ->x - elementPtr->x;
            const float dy = elementPtrJ->y - elementPtr->y;
            const float dz = elementPtrJ->z - elementPtr->z;
            const float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const float invDist = 1.0f / sqrtf(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        elementPtr->vx += dt * Fx;
        elementPtr->vy += dt * Fy;
        elementPtr->vz += dt * Fz;
    }
}

void integrate(Body *p, float dt, int n) {    
    #pragma omp for schedule(static)
    for (int i = 0; i < n; i++) {
        Body *elementPtr = &(p[i]);

	    elementPtr->x += elementPtr->vx * dt;
        elementPtr->y += elementPtr->vy * dt;
        elementPtr->z += elementPtr->vz * dt;
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
    data = aligned_alloc(64, total_bodies * sizeof(Body));

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

    printf("Running optimized OpenMP CPU simulation...\n");

    omp_set_nested(1);
    omp_set_dynamic(0);

    StartTimer();

    for (int iter = 0; iter < nIters; iter++) {
        #pragma omp parallel for schedule(static)
        for (int sys = 0; sys < num_systems; sys++) {
            Body *system_ptr = &data[sys * bodies_per_system];
            //OMP_PRINT_NUM_THREADS("Master Loop", sys == 0);

            bodyForce(system_ptr, dt, bodies_per_system);
            integrate(system_ptr, dt, bodies_per_system);
        }
    }

    totalTime = GetTimer() / 1000.0;

    /* Metrics calculation */
    double interactions_per_system = (double) bodies_per_system * bodies_per_system;
    double total_interactions = interactions_per_system * num_systems * nIters;

    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);

    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           data[0].x, data[0].y, data[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           data[1].x, data[1].y, data[1].z);

    free(data);
    return 0;
}