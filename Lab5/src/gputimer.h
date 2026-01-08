#ifndef GPUTIMER_H
#define GPUTIMER_H
#include "helpers.h"
#include <chrono>

static std::chrono::time_point<std::chrono::high_resolution_clock> start_cpu, stop_cpu;

void create_timer() {

}

void destroy_timer() {
    
}

void start_timer() {
    start_cpu = std::chrono::high_resolution_clock::now();
}

void stop_timer() {
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();
    
    stop_cpu = std::chrono::high_resolution_clock::now();
}

float get_timer_ms() {
    std::chrono::duration<float, std::milli> duration = stop_cpu - start_cpu;
    return duration.count();
}

#endif // TIMER_H

