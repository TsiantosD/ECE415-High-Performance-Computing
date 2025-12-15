#ifndef HELPERS_H
#define HELPERS_H
#include <time.h> 

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

#define PRINT_PROGRESS_RATE(iter, total, start_clock) \
    do { \
        double elapsed_sec = (double)(clock() - (start_clock)) / CLOCKS_PER_SEC; \
        double rate = (elapsed_sec > 0) ? (double)(iter) / elapsed_sec : 0.0; \
        printf("\rIter: %d/%d | Rate: %.2f iters/s", (iter), (total), rate); \
        fflush(stdout); \
    } while(0)

#endif