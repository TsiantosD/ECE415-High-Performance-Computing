#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include "helpers.h"
#include "timer.h"

#define OMP_SCHEDULE_TYPE dynamic

typedef struct {
    float *restrict x, *restrict y, *restrict z;
    float *restrict vx, *restrict vy, *restrict vz;
} GalaxySoA;

static void bodyForce(GalaxySoA *p, float dt, int n) {
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

static void integrate(GalaxySoA *p, float dt, int n) {
    int i;
    
    for (i = 0; i < n; i++) {
	    p->x[i] += p->vx[i] * dt;
        p->y[i] += p->vy[i] * dt;
        p->z[i] += p->vz[i] * dt;
    }
}

double run_cpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, 
                          const float dt, Body *data) {
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

    omp_set_nested(0);
    omp_set_dynamic(0);

    StartTimer();
    
    int total_bodies = num_systems * bodies_per_system;
    for (int iter = 0; iter < nIters; iter++) {
        PRINT_PROGRESS_RATE(iter + 1, nIters);
        #pragma omp parallel for schedule(OMP_SCHEDULE_TYPE) 
        for (int sys = 0; sys < num_systems; sys++) {
            bodyForce(&systems[sys], dt, bodies_per_system);
            integrate(&systems[sys], dt, bodies_per_system);
        }

        // Copy back to data for saving frame (optional)
        for (int s = 0; s < num_systems; s++) {
            for (int i = 0; i < bodies_per_system; i++) {
                int idx = s * bodies_per_system + i;
                data[idx].x  = systems[s].x[i];
                data[idx].y  = systems[s].y[i];
                data[idx].z  = systems[s].z[i];
            }
        }
        save_frame(data, total_bodies, iter + 1);
    }

    double total_time = GetTimer() / 1000.0f;

    for (int s = 0; s < num_systems; s++) {
        // Move data back to data array
        for (int i = 0; i < bodies_per_system; i++) {
            int idx = s * bodies_per_system + i;
            data[idx].x  = systems[s].x[i];
            data[idx].y  = systems[s].y[i];
            data[idx].z  = systems[s].z[i];
            data[idx].vx = systems[s].vx[i];
            data[idx].vy = systems[s].vy[i];
            data[idx].vz = systems[s].vz[i];
        }
    }

    for (int s = 0; s < num_systems; s++) {
        free(systems[s].x);
        free(systems[s].y);
        free(systems[s].z);
        free(systems[s].vx);
        free(systems[s].vy);
        free(systems[s].vz);
    }
    free(systems);

    return total_time;
}