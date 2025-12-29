#ifndef HELPERS_H
#define HELPERS_H
#include <time.h> 
#include <stdio.h>

#define SOFTENING 0.01f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

#ifdef __cplusplus
extern "C" {
#endif

void cleanUp(void);

double run_cpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, 
                            const float dt, Body *data);
double run_gpu_simulation(const int num_systems, const int bodies_per_system, const int nIters, 
                            const float dt, Body *data);

#ifdef __cplusplus
}
#endif

//! These macros are meant to be used instead of writing testing code
//! easier to use and spot/remove when refactoring final version

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define ATOMIC_INCREMENT_COUNTER(counter) \
    _Pragma("omp atomic")                  \
    (counter)++

#define PRINT_COUNTER(counter, msg)             \
    do {                                        \
        printf("%s: %d\n", (msg), (counter));   \
        fflush(stdout);                         \
    } while(0)

#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaDeviceSynchronize();                                             \
        cudaError_t _err = cudaGetLastError();                               \
        if (_err != cudaSuccess) {                                           \
            printf("CUDA Error: %s in %s, line %d\n",                        \
                   cudaGetErrorString(_err), __FILE__, __LINE__);            \
            cleanUp();                                           \
            exit(1);                                             \
        }                                                                    \
    } while (0)

#define OMP_PRINT(msg) \
    do { \
        int _tid = omp_get_thread_num();            \
        printf("Thread #%d: %s\n", _tid, (msg));    \
        fflush(stdout);                             \
    } while (0)


#define OMP_PRINT_NUM_THREADS(msg, enable, threadSelExpr)\
    do {                                                 \
        if (enable) {                                    \
            int __tid = omp_get_thread_num();            \
            if (__tid == threadSelExpr) {                \
                printf("[%d] Using %d threads: %s\n",    \
                __tid, omp_get_num_threads(), (msg));    \
                fflush(stdout);                          \
            }                                            \
        }                                                \
    } while (0)


#define PRINT_PROGRESS(iter, total) \
    printf("\rIter: %d/%d done", (iter), (total)); \
    fflush(stdout);

#define PRINT_PROGRESS_RATE(iter, total)                          \
    do {                                                          \
        static double __last_time = 0.0;                          \
        static int __last_iter = 0;                               \
        double _t = now_sec();                                    \
                                                                  \
        if (__last_time == 0.0) {                                 \
            __last_time = _t;                                     \
            __last_iter = (iter);                                 \
        }                                                         \
                                                                  \
        double __dt = _t - __last_time;                           \
        int __di = (iter) - __last_iter;                          \
        double __rate = (__dt > 0.0) ? __di / __dt : 0.0;         \
        double __pct = 100.0 * (iter) / (total);                  \
                                                                  \
        printf("\r%6.2f%% | %d/%d | %.2f it/s",                   \
               __pct, (iter), (total), __rate);                   \
        fflush(stdout);                                           \
                                                                  \
        __last_time = _t;                                         \
        __last_iter = (iter);                                     \
    } while(0)

#endif
