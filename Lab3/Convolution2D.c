/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

typedef enum {
    HOST_ALLOC = 1,
    NORMAL = 0
} ErrorCode;

typedef float PixelScalar;

unsigned int filter_radius;
ErrorCode exitCode = NORMAL;

#define FILTER_LENGTH    (2 * filter_radius + 1)
#define ABS(val)         ((val)<0.0 ? (-(val)) : (val))
#define accuracy         0.00005 

#define CHECK_ALLOC(ptr, label)        \
    do {                               \
        if ((ptr) == NULL) {           \
            exitCode = HOST_ALLOC;     \
            printf("Allocation failed: %s\n", #ptr); \
            goto label;                \
        }                              \
    } while(0)

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

                h_Dst[y * imageW + x] = sum;
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
 
                h_Dst[y * imageW + x] = sum;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    PixelScalar
        *h_Filter,
        *h_Input,
        *h_Buffer,
        *h_OutputCPU;

    int imageW;
    int imageH;
    unsigned int i;
  
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
    CHECK_ALLOC(h_Filter, CLEANUP0);
    h_Input     = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC(h_Input, CLEANUP1);
    h_Buffer    = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC(h_Buffer, CLEANUP2);
    h_OutputCPU = (PixelScalar *)malloc(imageW * imageH * sizeof(PixelScalar));
    CHECK_ALLOC(h_OutputCPU, CLEANUP3);

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
    // TODO: To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.

    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius);
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius);
    
    // TODO: Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    // Cleanup sequence
    CLEANUP4:
    free(h_OutputCPU);
    CLEANUP3:
    free(h_Buffer);
    CLEANUP2:
    free(h_Input);
    CLEANUP1:
    free(h_Filter);
    CLEANUP0:
    // TODO: Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    // cudaDeviceReset();

    return exitCode;
}
