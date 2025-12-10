#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"

// Helper: Read PGM
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;

    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    if(fscanf(in_file, "%s", sbuf) != 1) exit(1); /*Skip P5*/
    if(fscanf(in_file, "%d",&result.w) != 1) exit(1);
    if(fscanf(in_file, "%d",&result.h) != 1) exit(1);
    if(fscanf(in_file, "%d",&v_max) != 1) exit(1);
    fgetc(in_file); // Skip the single whitespace/newline after max_val

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    if(fread(result.img, sizeof(unsigned char), result.w*result.h, in_file) != result.w*result.h) exit(1);
    fclose(in_file);

    return result;
}

// Helper: Write PGM
void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;

    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

// Helper: Free PGM Memory
void free_pgm(PGM_IMG img) {
    if(img.img) free(img.img);
}
