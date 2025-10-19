// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#ifndef SIZE
#warning "SIZE not defined! Using default 4096."
#define SIZE 4096
#endif

#ifndef INPUT_FILE
#warning "INPUT_FILE not defined! Using default input/4096-timescapes.grey."
#define INPUT_FILE "input/4096-timescapes.grey"
#endif

#ifndef OUTPUT_FILE
#warning "OUTPUT_FILE not defined! Using default output/timescapes.grey."
#define OUTPUT_FILE "output/timescapes.grey"
#endif

#ifndef GOLDEN_FILE
#warning "GOLDEN_FILE not defined! Using default golden/timescapes.grey."
#define GOLDEN_FILE "golden/timescapes.grey"
#endif

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];

/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0;
	unsigned long long sum_input = 0, sum_golden = 0;
	int i, j, i_times_SIZE, i_times_SIZE_plus_j, top_row, bottom_row;
	unsigned int pixel_horizontal, pixel_vertical;
	int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	i_times_SIZE = SIZE;
	int inc_i_times_SIZE = SIZE << 1;
	int dec_i_times_SIZE = 0;

	for (i = 1; i < SIZE - 1; i++) {
		// Referencing
		const unsigned char *top_row = &input[dec_i_times_SIZE];
		const unsigned char *bottom_row = &input[inc_i_times_SIZE];
		const unsigned char *mid_row = &input[i_times_SIZE];
		unsigned char *out_row = &output[i_times_SIZE];
		
		for (j = 1; j < SIZE - 1; j++) {
			// Strength reduction
			pixel_horizontal = -top_row[j - 1] + top_row[j + 1];
			pixel_vertical = top_row[j - 1] + (top_row[j] << 1) + top_row[j + 1];
			pixel_horizontal += -(mid_row[j - 1] << 1) + (mid_row[j + 1] << 1)  + -bottom_row[j - 1]+ bottom_row[j + 1];
			pixel_vertical += -bottom_row[j - 1] + -(bottom_row[j] << 1) + -bottom_row[j + 1];
			
			res = sqrt(pixel_horizontal * pixel_horizontal + pixel_vertical * pixel_vertical);
			
			sum_input += (out_row[j] = (res > 255) ? 255 : (unsigned char)res);
		}
		dec_i_times_SIZE += SIZE;
		inc_i_times_SIZE += SIZE;
		i_times_SIZE += SIZE;
	}

	for (int i = 1; i < SIZE - 1; i++) {
		const unsigned char *row = &golden[i * SIZE + 1];
		for (int j = 0; j < SIZE - 2; j++) sum_golden += row[j];
	}

	PSNR = sum_input * sum_input - (sum_input << 1) * sum_golden + sum_golden * sum_golden;
  
	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

