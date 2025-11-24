/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

#define FILTER_LENGTH    (2 * filter_radius + 1)
#define ABS(val)         ((val)<0.0 ? (-(val)) : (val))
#define accuracy         0.00005 
#define BLOCK_WIDTH       32
#define BLOCK_HEIGHT      32

#define CHECK_ALLOC_HOST(ptr)                        \
    do {                                             \
        if ((ptr) == NULL) {                         \
            printf("Allocation failed: %s\n", #ptr); \
            cleanUp(HOST_ALLOC);                     \
        }                                            \
    } while(0)

#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaDeviceSynchronize();                                             \
        cudaError_t _err = cudaGetLastError();                               \
        if (_err != cudaSuccess) {                                           \
            printf("CUDA Error: %s in %s, line %d\n",                        \
                   cudaGetErrorString(_err), __FILE__, __LINE__);            \
            cleanUp(DEVICE_ERROR);                                           \
        }                                                                    \
    } while (0)

#define CHECK_SCANF(scanf_call)                                          \
    do {                                                                 \
        if ((scanf_call) != 1) {                                         \
            fprintf(stderr, "Error: invalid scanf input at %s:%d\n",     \
                    __FILE__, __LINE__);                                 \
            cleanUp(SCANF);                                              \
        }                                                                \
    } while(0)

typedef enum {
    SCANF = 3,
    HOST_ALLOC = 1,
    DEVICE_ERROR = 2,
    NORMAL = 0
} ErrorCode;

#if USE_DOUBLES == 1
typedef double PixelScalar;
#elif USE_DOUBLES == 0
typedef float PixelScalar;
#else
#error "USE_DOUBLES must be 0 or 1"
#endif

unsigned int filter_radius;

PixelScalar
    *h_Filter = NULL,
    *h_Input = NULL,
    *h_InputPadded = NULL,
    *h_Buffer = NULL,
    *h_OutputCPU = NULL,
    *h_OutputGPU = NULL,
    *d_Filter = NULL,
    *d_Input = NULL,
    *d_Buffer = NULL,
    *d_Output = NULL;

void cleanUp(ErrorCode exitCode) {
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Buffer);
    free(h_Filter);
    free(h_Input);
    free(h_Buffer);
    free(h_OutputCPU);

    cudaDeviceReset();
    exit(exitCode);
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(PixelScalar *h_Dst, PixelScalar *h_Src, PixelScalar *h_Filter, 
                       int imageW, int imageH, int filterR) {

    int x, y, k;
    
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            PixelScalar sum = 0;

            for (k = -filterR; k <= filterR; k++) {
                int d = x + k;

                if (d >= 0 && d < imageW)
                    sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(PixelScalar *h_Dst, PixelScalar *h_Src, PixelScalar *h_Filter,
                          int imageW, int imageH, int filterR) {

    int x, y, k;

    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            PixelScalar sum = 0;

            for (k = -filterR; k <= filterR; k++) {
                int d = y + k;

                if (d >= 0 && d < imageH)
                    sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(PixelScalar *d_Dst, PixelScalar *d_Src, PixelScalar *d_Filter,
                               int imageW, int imageH, int filterR) {
    int k;
    PixelScalar sum = 0;
    int x = blockIdx.x * blockDim.x + threadIdx.x + filterR;
    int y = blockIdx.y * blockDim.y + threadIdx.y + filterR;


    // printf("[x,y]=[%d, %d], idx=%d: %f\n", x, y, y * (imageW + 2 * filterR) + x, d_Src[y * (imageW + 2 * filterR) + x]);
    for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        sum += d_Src[y * (imageW + 2 * filterR) + d] * d_Filter[filterR - k];
    }

    d_Dst[y * (imageW + 2 * filterR) + x] = sum;
}

__global__ void convolutionColumnGPU(PixelScalar *d_Dst, PixelScalar *d_Src, PixelScalar *d_Filter,
                               int imageW, int imageH, int filterR) {
    int k;
    PixelScalar sum = 0;
    int x = blockIdx.x * blockDim.x + threadIdx.x + filterR;
    int y = blockIdx.y * blockDim.y + threadIdx.y + filterR;
    
    for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        sum += d_Src[d * (imageW + 2 * filterR) + x] * d_Filter[filterR - k];
    }

    d_Dst[(y - filterR) * imageW + (x - filterR)] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    int imageW;
    int imageH;
    unsigned int i;
    int correctOutput = 1;
    PixelScalar maxDiff = 0;
	struct timespec  tv1, tv2;
    int imagePad = 0;
    
    printf("Using scalar with sizeof: %lubytes\n", sizeof(PixelScalar));

    printf("Enter filter radius : ");
    CHECK_SCANF(scanf("%d", &filter_radius));
    imagePad = filter_radius;
    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.     
    
    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    CHECK_SCANF(scanf("%d", &imageW));
    imageH = imageW;

    dim3 dimGrid((imageW  + BLOCK_WIDTH  - 1)  / BLOCK_WIDTH, (imageH  + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);
    dim3 dimBlock(imageW > TILE_WIDTH ? TILE_WIDTH : imageW, imageH > TILE_HEIGHT ? TILE_HEIGHT : imageH);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

    h_Filter    = (PixelScalar *)malloc(FILTER_LENGTH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_Filter);
    h_Input     = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_Input);
    h_InputPadded = (PixelScalar *)malloc(((imageW + imagePad * 2) * (imageH + imagePad * 2))* sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_InputPadded);
    h_Buffer    = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_Buffer);
    h_OutputCPU = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_OutputCPU);
    h_OutputGPU = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_OutputGPU);
    cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc((void **) &d_Input, ((imageW + imagePad * 2) * (imageH + imagePad * 2)) * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc((void **) &d_Buffer, ((imageW + imagePad * 2) * (imageH + imagePad * 2)) * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc((void **) &d_Output, imageW * imageH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();

    //* Set cuda memory to 0 for padding just in case
    cudaMemset(d_Input, 0, ((imageW + imagePad * 2) * (imageH + imagePad * 2)) * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMemset(d_Buffer, 0, ((imageW + imagePad * 2) * (imageH + imagePad * 2)) * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    
    //* Initialise random arrays
    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (PixelScalar)(rand() % 16);
    }

    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            PixelScalar value = (PixelScalar)rand() / ((PixelScalar)RAND_MAX / 255) + (PixelScalar)rand() / (PixelScalar)RAND_MAX;
            h_InputPadded[(i + imagePad) * (imageW + imagePad * 2) + (j + imagePad)] = value;
            h_Input[i * imageW + j] = value;
        }
    }

    for (int i = 0; i < imageH + imagePad * 2; i++) {
        for (int j = 0; j < imageW + imagePad * 2; j++) {
            printf("%12.5f ", *(h_InputPadded + i * (imageW + imagePad * 2) + j));
        }
        printf("\n");
    }

    GpuTimer timer = GpuTimer();
    timer.Start();
    
    //* Copy h_Filter and h_Input to GPU
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(PixelScalar), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_Input, h_InputPadded, ((imageW + imagePad * 2) * (imageH + imagePad * 2)) * sizeof(PixelScalar), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    
    printf("GPU computation...\n");
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    //* Transfer data from Device back to Host memory
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(PixelScalar), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    timer.Stop();

    printf("CPU computation...\n");
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    
    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            printf("%12.5f ", *(h_OutputCPU + i * imageW + j));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    printf("\n");

    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            printf("%12.5f ", *(h_OutputGPU + i * imageW + j));
        }
        printf("\n");
    }

    //* Perform comparison between GPU / CPU results
    for (int y = 0; y < imageH; y++) {
        for (int x = 0; x < imageW; x++) {
            int index = y * imageW + x;
            PixelScalar diff = ABS(h_OutputCPU[index] - h_OutputGPU[index]);
            maxDiff = diff > maxDiff ? diff : maxDiff;

            if (diff > accuracy) {
                //TODO: remove this and make it break whenever this is true
                // printf("Accuracy bigger than %f on pixel [%d, %d]\n", accuracy, x, y);
                // printf("  h_OutputCPU[%d]=%f\n", index, h_OutputCPU[index]);
                // printf("  h_OutputGPU[%d]=%f\n", index, h_OutputGPU[index]);
		        correctOutput = 0;
            }
        }
    }

    if (correctOutput)
        printf("Results correct!\n");

    printf("Max difference: %.15lf\n", maxDiff);
    printf("Time in GPU: %f\n", timer.Elapsed()/1000);
    printf("Time in CPU: %lf\n", (double) (tv2.tv_nsec - tv1.tv_nsec) / 1.0E9 + (double) (tv2.tv_sec - tv1.tv_sec));

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cleanUp(NORMAL);

    return NORMAL;
}
