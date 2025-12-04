#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"

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
    
    fscanf(in_file, "%s", sbuf); /*Skip P5*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); // Skip the single whitespace/newline after max_val

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
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

// Compute & Clip Histogram for a specific tile
void compute_histogram(unsigned char* data, 
                       int w, int h, int start_x, int start_y, 
                       int tile_w, int tile_h, 
                       int* lut) {
    int hist[256] = {0};
    int x, y, i, avg_inc, val;
    int excess = 0, cdf = 0, total_pixels = tile_w * tile_h; 

    // Build Histogram
    for (y = start_y; y < start_y + tile_h; ++y) {
        for (x = start_x; x < start_x + tile_w; ++x) {
            // Boundary check mostly for the right/bottom edge tiles
            if(x < w && y < h) {
                hist[data[y * w + x]]++;
            }
        }
    }

    // Clip Histogram
    for (i = 0; i < 256; ++i) {
        if (hist[i] > CLIP_LIMIT) {
            excess += (hist[i] - CLIP_LIMIT);
            hist[i] = CLIP_LIMIT;
        }
    }

    // Redistribute Excess (simplisticly)
    avg_inc = excess / 256;
    for (i = 0; i < 256; ++i) {
        hist[i] += avg_inc;
    }
    
    // Compute CDF & LUT
    for (i = 0; i < 256; ++i) {
        cdf += hist[i];
        // Calculate equalized value
        val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
        if (val > 255) 
            val = 255;
        lut[i] = val;
    }
}

// Core CLAHE
PGM_IMG apply_clahe(PGM_IMG img_in) {
    PGM_IMG img_out;
    int w = img_in.w;
    int h = img_in.h;
    int grid_w, grid_h;
    int *all_luts; // Big array to store LUTs for all tiles
    int* current_lut_ptr;
    int ty, tx, x, y, x1, x2, y1, y2, tl, tr, bl, br, val;
    int x_start, y_start, actual_tile_w, actual_tile_h;
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;

    // Allocate output image
    img_out.w = w;
    img_out.h = h;
    img_out.img = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    // Calculate grid dimensions
    grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    
    // Allocate memory for all LUTs: [grid_h][grid_w][256],
    // as an 1D array
    all_luts = (int *)malloc(grid_w * grid_h * 256 * sizeof(int));

    // Precompute all Tile LUTs ---
    for (ty = 0; ty < grid_h; ++ty) {
        for (tx = 0; tx < grid_w; ++tx) {
            x_start = tx * TILE_SIZE;
            y_start = ty * TILE_SIZE;
            
            // Handle boundary tiles that might be smaller than TILE_SIZE
            actual_tile_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
            actual_tile_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
            
            // Pointer to the specific 256-entry LUT for this tile
            current_lut_ptr = &all_luts[(ty * grid_w + tx) * 256];
            
            compute_histogram(img_in.img, w, h, x_start, y_start, 
                              actual_tile_w, actual_tile_h, 
                              current_lut_ptr);
        }
    }

    // Render pixels using Bilinear Interpolation
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            
            // Find relative position in the grid
            // (y / TILE_SIZE) gives the tile index, but we want the center approach
            // So we offset by 0.5 to align interpolation with tile centers
            ty_f = (float)y / TILE_SIZE - 0.5f;
            tx_f = (float)x / TILE_SIZE - 0.5f;
            
            y1 = (int)floor(ty_f);
            x1 = (int)floor(tx_f);
            y2 = y1 + 1;
            x2 = x1 + 1;

            // Weights for interpolation
            y_weight = ty_f - y1;
            x_weight = tx_f - x1;

            // Clamp tile indices to boundaries 
            // If a pixel is near the edge, it might not have 4 neighbors
            if (x1 < 0) 
                x1 = 0;
            if (x2 >= grid_w) 
                x2 = grid_w - 1;
            if (y1 < 0) 
                y1 = 0;
            if (y2 >= grid_h) 
                y2 = grid_h - 1;

            // Original pixel intensity
            val = img_in.img[y * w + x];
            
            // Fetch mapped values from the 4 nearest tile LUTs
            tl = all_luts[(y1 * grid_w + x1) * 256 + val];
            tr = all_luts[(y1 * grid_w + x2) * 256 + val];
            bl = all_luts[(y2 * grid_w + x1) * 256 + val];
            br = all_luts[(y2 * grid_w + x2) * 256 + val];

            // Bilinear interpolation
            top = tl * (1.0f - x_weight) + tr * x_weight;
            bot = bl * (1.0f - x_weight) + br * x_weight;
            final_val = top * (1.0f - y_weight) + bot * y_weight;

            img_out.img[y * w + x] = (unsigned char)(final_val + 0.5f);
        }
    }

    free(all_luts);
    return img_out;
}
