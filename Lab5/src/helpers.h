#ifndef HELPERS_H
#define HELPERS_H
#include <time.h> 

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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
        int _tid = omp_get_thread_num(); \
        printf("Thread #%d: %s\n", _tid, (msg)); \
    } while (0)


/* Usage: OMP_PRINT_NUM_THREADS("message", 1); */
#define OMP_PRINT_NUM_THREADS(msg, enable)             \
    do {                                               \
        if (enable) {                                  \
            if (omp_get_thread_num() == 0) {           \
                printf("Using %d threads: %s\n",       \
                       omp_get_num_threads(), (msg));  \
            }                                          \
        }                                              \
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