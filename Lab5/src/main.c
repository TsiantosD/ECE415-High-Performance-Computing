#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helpers.h"

#include <sys/stat.h>
#include <sys/types.h>

#define ACCURACY 0.001

void save_frame(Body *data, int total_bodies, int iter) {
    char filename[64];
    mkdir("frames", 0777); 
    sprintf(filename, "frames/frame_%04d.csv", iter);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error opening frame file");
        return;
    }

    fprintf(f, "x,y,z\n");
    for (int i = 0; i < total_bodies; i++) {
        fprintf(f, "%f,%f,%f\n", data[i].x, data[i].y, data[i].z);
    }
    fclose(f);
}

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;       	/* Number of independent galaxies */
    int bodies_per_system = 8192;	/* Number of bodies per galaxy */
    int nIters = 20;                /* Simulation steps */ 
    float dt = 0.01f;               /* Timestep */

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [nIters] [dt] [output_file]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc >= 3) nIters = atoi(argv[2]);
    if (argc >= 4) dt = atof(argv[3]);

    FILE *fp;
    Body *cpu_data, *gpu_data;
    float *buf;
    int total_bodies;
    double totalTime;

    /* Attempt to load dataset */
    fp = fopen(argv[1], "rb");
    if (fp) {
        if (fread(&num_systems, sizeof(int), 1, fp) != 1 ||
            fread(&bodies_per_system, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Error: failed to read dataset header\n");
            fclose(fp);
            return EXIT_FAILURE;
        }

       printf("Found dataset: %d systems of %d bodies.\n",
               num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    /* Allocate memory for ALL systems */
    total_bodies = num_systems * bodies_per_system;
    
    /* Clear old frames if they exist */
    system("rm -rf frames && mkdir -p frames");
    
    cpu_data = (Body *) malloc(total_bodies * sizeof(Body));
    gpu_data = (Body *) malloc(total_bodies * sizeof(Body));

    /* Initialize data */
    if (fp) {
        size_t nread = fread(cpu_data, sizeof(Body), (size_t)total_bodies, fp);
        if (nread != (size_t)total_bodies) {
            fprintf(stderr,
                    "Error: expected %d bodies, read %zu\n",
                    total_bodies, nread);
            fclose(fp);
            free(cpu_data);
            free(gpu_data);
            return EXIT_FAILURE;
        }
        fclose(fp);
    } else { /* Random initialization if file missing */
        buf = (float *) cpu_data;
        for (int i = 0; i < 6 * total_bodies; i++)
            buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    }

    memcpy(gpu_data, cpu_data, total_bodies* sizeof(Body));

    /* Metrics calculation */
    double interactions_per_system = (double) bodies_per_system * bodies_per_system;
    double total_interactions = interactions_per_system * num_systems * nIters;

#if CHECK_OUTPUT==1 || ONLY_CPU==1
#if SEQ_CPU == 1
    printf("Running sequential CPU simulation for %d systems...\n",
           num_systems);
#else
    printf("Running parallel CPU simulation for %d systems...\n",
           num_systems);
#endif
    totalTime = run_cpu_simulation(num_systems, bodies_per_system, nIters, dt, cpu_data);
    printf("Total CPU Time: %.3f seconds\n", totalTime);
    printf("Average CPU Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);
#endif

#if ONLY_CPU==0
    printf("\n");
    printf("Running GPU simulation for %d systems...\n",
           num_systems);
    totalTime = run_gpu_simulation(num_systems, bodies_per_system, nIters, dt, gpu_data);
    printf("Total GPU Time: %.3f seconds\n", totalTime);
    printf("Average GPU Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);
#endif

#if CHECK_OUTPUT==1 && ONLY_CPU==0
    int correct = 1;
    for (int i = 0; i < total_bodies; i++) {
    double x_diff = fabs(cpu_data[i].x - gpu_data[i].x);
    double y_diff = fabs(cpu_data[i].y - gpu_data[i].y);
    double z_diff = fabs(cpu_data[i].z - gpu_data[i].z);
        if (x_diff > ACCURACY || y_diff > ACCURACY || z_diff > ACCURACY) {
            correct = 0;
            break;
        }
    }

    if (!correct)
        printf("\nGPU data is not correct!\n");
#endif

#if ONLY_CPU==1
    /* Ensure an output filename was provided (now potentially at argv[4]) */
    const char *out_filename = NULL;
    if (argc >= 5) out_filename = argv[4];
    else if (argc == 2) { /* Fallback for original usage */ }

    if (out_filename) {
        fp = fopen(out_filename, "wb");
        if (!fp) {
            fprintf(stderr, "\nError: failed to open output file %s\n", out_filename);
        } else {
            /* Write the header (Systems + Bodies count) to match input format */
            if (fwrite(&num_systems, sizeof(int), 1, fp) != 1 ||
                fwrite(&bodies_per_system, sizeof(int), 1, fp) != 1) {
                fprintf(stderr, "\nError: failed to write dataset header to %s\n", out_filename);
            } else {
                /* Write the actual body data */
                size_t nwrite = fwrite(cpu_data, sizeof(Body), (size_t)total_bodies, fp);
                if (nwrite != (size_t)total_bodies) {
                    fprintf(stderr,
                            "\nError: expected %d bodies, wrote %zu\n",
                            total_bodies, nwrite);
                } else {
                    printf("\nSuccessfully wrote CPU simulation data to %s\n", out_filename);
                }
            }
            fclose(fp);
        }
    } else {
        printf("\nWarning: WRITE_CPU_OUTPUT enabled but no output file argument provided.\n");
    }
#endif

    free(cpu_data);
    free(gpu_data);
    return 0;
}
