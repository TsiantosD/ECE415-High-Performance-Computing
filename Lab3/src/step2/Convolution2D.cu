/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

#define FILTER_LENGTH    (2 * filter_radius + 1)
#define ABS(val)         ((val)<0.0 ? (-(val)) : (val))
#define accuracy         0.00005 

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

#ifdef USE_DOUBLES
typedef double PixelScalar;
#else
typedef float PixelScalar;
#endif
unsigned int filter_radius;

PixelScalar
    *h_Filter = NULL,
    *h_Input = NULL,
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
    cudaFree(d_Output);
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

                if (d >= 0 && d < imageW) {
                    sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
                }     

                h_Dst[y * imageW + x] = sum; //TODO: Move outside loop
            }
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

                if (d >= 0 && d < imageH) {
                    sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
                }
 
                h_Dst[y * imageW + x] = sum; //TODO: Move outside loop
            }
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
    
    for (k = -filterR; k <= filterR; k++) {
        int d = threadIdx.x + k;

        if (d >= 0 && d < imageW)
            sum += d_Src[threadIdx.y * imageW + d] * d_Filter[filterR - k];
    }

    d_Dst[threadIdx.y * imageW + threadIdx.x] = sum;
}

__global__ void convolutionColumnGPU(PixelScalar *d_Dst, PixelScalar *d_Src, PixelScalar *d_Filter,
                               int imageW, int imageH, int filterR) {
    int k;
    PixelScalar sum = 0;
    
    for (k = -filterR; k <= filterR; k++) {
        int d = threadIdx.y + k;

        if (d >= 0 && d < imageH)
            sum += d_Src[d * imageW + threadIdx.x] * d_Filter[filterR - k];
    }

    d_Dst[threadIdx.y * imageW + threadIdx.x] = sum;
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
    
    printf("Enter filter radius : ");
    CHECK_SCANF(scanf("%d", &filter_radius));
    
    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.     
    
    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    CHECK_SCANF(scanf("%d", &imageW));
    imageH = imageW;

    //TODO: Check if this is it 100%
    dim3 dimGrid(1);
    dim3 dimBlock(imageW, imageH);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

    h_Filter    = (PixelScalar *)malloc(FILTER_LENGTH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_Filter);
    h_Input     = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_Input);
    h_Buffer    = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_Buffer);
    h_OutputCPU = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_OutputCPU);
    h_OutputGPU = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC_HOST(h_OutputGPU);
    cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc((void **) &d_Input, imageW * imageH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc((void **) &d_Buffer, imageW * imageH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMalloc((void **) &d_Output, imageW * imageH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();

    //* Initialise random arrays

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (PixelScalar)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (PixelScalar)rand() / ((PixelScalar)RAND_MAX / 255) + (PixelScalar)rand() / (PixelScalar)RAND_MAX;
    }

    //* Copy h_Filter and h_Input to GPU
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(PixelScalar), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(PixelScalar), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    
    printf("GPU computation...\n");
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);

    //! 'Fall through' Let Host run concurrently with Device

    printf("CPU computation...\n");
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);

    //* Transfer data from Device back to Host memory
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(PixelScalar), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();
    
    //* Perform comparison between GPU / CPU results
    for (int y = 0; y < imageH; y++) {
        for (int x = 0; x < imageW; x++) {
            int index = y * imageW + x;
            PixelScalar diff = ABS(h_OutputCPU[index] - h_OutputGPU[index]);
            maxDiff = diff > maxDiff ? diff : maxDiff;

            if (diff > accuracy) {
		        correctOutput = 0;
                break;
            }
        }

        if (!correctOutput)
            break;
    }

    if (correctOutput)
        printf("Results correct!\n");

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cleanUp(NORMAL);

    return NORMAL;
}
