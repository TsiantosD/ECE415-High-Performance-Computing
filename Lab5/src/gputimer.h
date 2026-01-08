#ifndef GPUTIMER_H
#define GPUTIMER_H
#include "helpers.h"

static cudaEvent_t start, stop;

void create_timer() {
    cudaEventCreate(&start);
    CUDA_CHECK_LAST_ERROR();
    cudaEventCreate(&stop);
    CUDA_CHECK_LAST_ERROR();
}

void destroy_timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void start_timer() {
    cudaEventRecord(start);
}

void stop_timer() {
    cudaEventRecord(stop);
    CUDA_CHECK_LAST_ERROR();
    cudaEventSynchronize(stop);
    CUDA_CHECK_LAST_ERROR();
}

float get_timer_ms() {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    CUDA_CHECK_LAST_ERROR();
    return ms;
}

#endif // TIMER_H

