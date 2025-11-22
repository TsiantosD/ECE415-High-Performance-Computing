/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

typedef enum {
    HOST_ALLOC = 1,
    DEVICE_ERROR = 2,
    NORMAL = 0
} ErrorCode;

typedef float PixelScalar;

unsigned int filter_radius;
ErrorCode exitCode = NORMAL;

#define FILTER_LENGTH    (2 * filter_radius + 1)
#define ABS(val)         ((val)<0.0 ? (-(val)) : (val))
#define accuracy         0.00005 

#define CHECK_ALLOC_HOST(ptr)          \
    do {                               \
        if ((ptr) == NULL) {           \
            exitCode = HOST_ALLOC;     \
            printf("Allocation failed: %s\n", #ptr); \
            goto CLEANUP_HOST;         \
        }                              \
    } while(0)

#define CUDA_CHECK_LAST_ERROR()                                              \
    do {                                                                     \
        cudaDeviceSynchronize();                                             \
        cudaError_t _err = cudaGetLastError();                               \
        if (_err != cudaSuccess) {                                           \
            printf("CUDA Error: %s in %s, line %d\n",                        \
                   cudaGetErrorString(_err), __FILE__, __LINE__);            \
            exitCode = DEVICE_ERROR;                                         \
            goto CLEANUP_DEVICE;                                             \
        }                                                                    \
    } while (0)

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

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    PixelScalar
        *h_Filter = NULL,
        *h_Input = NULL,
        *h_Buffer = NULL,
        *h_OutputCPU = NULL,
        *h_OutputGPU = NULL,
        *d_Filter = NULL,
        *d_Input = NULL,
        *d_Output = NULL;

    int imageW;
    int imageH;
    unsigned int i;
    int correctOutput = 1;
  
    printf("Enter filter radius : ");
    scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.     

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

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
    cudaMalloc((void **) &d_Output, imageW * imageH * sizeof(PixelScalar));
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(PixelScalar), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(PixelScalar), cudaMemcpyHostToDevice);
    CUDA_CHECK_LAST_ERROR();

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (PixelScalar)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (PixelScalar)rand() / ((PixelScalar)RAND_MAX / 255) + (PixelScalar)rand() / (PixelScalar)RAND_MAX;
    }
    
    printf("GPU computation...\n");
    // TODO: Launch kernel

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);

    // After completing CPU computation wait for GPU and check for errors
    CUDA_CHECK_LAST_ERROR();
    cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(PixelScalar), cudaMemcpyDeviceToHost);
    CUDA_CHECK_LAST_ERROR();

    // TODO: Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    // for (int i = 0; i < imageH; i++) {
    //     for (int j = 0; j < imageW; j++) {
    //         printf("%12.5f ", *(h_OutputCPU + i * imageW + j));
    //     }
    //     printf("\n");
    // }
    
    for (int y = 0; y < imageH; y++) {
        for (int x = 0; x < imageW; x++) {
            int index = y * imageW + x;

            if (ABS(h_OutputCPU[index] - h_OutputGPU[index]) > accuracy) {
                printf("Accuracy bigger than %f on pixel [%d, %d]\n", accuracy, x, y);
                printf("  h_OutputCPU[%d]=%f\n", index, h_OutputCPU[index]);
                printf("  h_OutputGPU[%d]=%f\n", index, h_OutputGPU[index]);
                correctOutput = 0;
                break;
            }
        }

        if (!correctOutput)
            break;
    }

    if (correctOutput)
        printf("Results correct!\n");

    // Cleanup sequence
    CLEANUP_DEVICE:
    cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFree(d_Output);
    CLEANUP_HOST:
    free(h_Filter);
    free(h_Input);
    free(h_Buffer);
    free(h_OutputCPU);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();

    return exitCode;
}
