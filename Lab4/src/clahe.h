#ifndef CLAHE_H
#define CLAHE_H

#include <stdio.h>
#include <stdlib.h>

// Configuration for CLAHE
// 32x32 is a standard tile size for high-res images
#define TILE_SIZE 32    
// Threshold for contrast limiting (clip limit)    
#define CLIP_LIMIT 2

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;

// I/O functions
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

// Core Processing
PGM_IMG apply_clahe(PGM_IMG img_in);

#endif
