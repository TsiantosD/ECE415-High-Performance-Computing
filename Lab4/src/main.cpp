#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "clahe.h"

double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]){
    PGM_IMG img_in, h_img_out, d_img_out;
    double start, end, h_elapsed, d_elapsed;

    if (argc != 3) {
        printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    printf("Loading image...\n");
    img_in = read_pgm(argv[1]);
    
    printf("Running CPU CLAHE reference...\n");
    start = get_time_sec();
    
    h_img_out = apply_clahe(img_in);
    
    end = get_time_sec();
    h_elapsed = end - start;
    
    printf("Processing time: %.6f seconds\n", h_elapsed);
    printf("Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (h_elapsed * 1e6));

    printf("Running GPU CLAHE reference...\n");
    d_elapsed = d_apply_clahe(img_in, &d_img_out);
    printf("Processing time: %.6f seconds\n", d_elapsed * 1e-3);
    printf("Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (d_elapsed * 1e3));

    write_pgm(d_img_out, argv[2]);
    printf("Result saved to %s\n", argv[2]);

    free_pgm(img_in);
    free_pgm(h_img_out);

    return 0;
}
