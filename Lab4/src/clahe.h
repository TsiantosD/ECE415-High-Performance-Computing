#ifndef CLAHE_H
#define CLAHE_H
#include "helpers.h"

// Configuration for CLAHE
// 32x32 is a standard tile size for high-res images
#define TILE_SIZE 32    
// Threshold for contrast limiting (clip limit)    
#define CLIP_LIMIT 2

// Core Processing
PGM_IMG apply_clahe(PGM_IMG img_in);
double d_apply_clahe(PGM_IMG img_in, PGM_IMG *img_out);
void cleanUp();

#endif
