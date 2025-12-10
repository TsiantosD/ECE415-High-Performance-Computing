#ifndef HELPERS_H
#define HELPERS_H

#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaDeviceSynchronize();                                             \
        cudaError_t _err = cudaGetLastError();                               \
        if (_err != cudaSuccess) {                                           \
            printf("CUDA Error: %s in %s, line %d\n",                        \
                   cudaGetErrorString(_err), __FILE__, __LINE__);            \
            cleanUp();                                           \
        }                                                                    \
    } while (0)

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;

PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
bool check_pgm(PGM_IMG img1, PGM_IMG img2);
void free_pgm(PGM_IMG img);

#endif