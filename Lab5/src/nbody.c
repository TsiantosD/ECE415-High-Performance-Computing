#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "helpers.h"
#include "timer.h"

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
static void bodyForce(Body * p, float dt, int n) {
    int i, j;
    double Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
	    Fx = 0.0f;
    	Fy = 0.0f;
    	Fz = 0.0f;

    	for (j = 0; j < n; j++) {
	        dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}
int counter = 0;
/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
static void integrate(Body * p, float dt, int n) {
    int i;
    
    for (i = 0; i < n; i++) {
        counter++;
	    p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

double run_cpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, 
                        const float dt, Body *data) {
    StartTimer();
    
    Body *system_ptr;
    int iter, sys;
    for (iter = 1; iter <= nIters; iter++) {

        PRINT_PROGRESS_RATE(iter, nIters);
        for (sys = 0; sys < num_systems; sys++) {
	        system_ptr = &data[sys * bodies_per_system];
	        
	        bodyForce(system_ptr, dt, bodies_per_system);
	        integrate(system_ptr, dt, bodies_per_system);
        }
    }

    return GetTimer() / 1000.0;
}