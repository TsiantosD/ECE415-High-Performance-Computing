#include <math.h>
#include <cuda_runtime.h>
#include "helpers.h"
#include "timer.h"

#define BLOCK_SIZE 32

extern "C" double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, 
                          const float dt, Body *data) {

}
