#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"

#define ACCURACY 0.001

void check_data(Body *golden_data, Body *data_under_test, int total_bodies) {
    int correct = 1;
    for (int i = 0; i < total_bodies; i++) {
    double x_diff = fabs(golden_data[i].x - data_under_test[i].x);
    double y_diff = fabs(golden_data[i].y - data_under_test[i].y);
    double z_diff = fabs(golden_data[i].z - data_under_test[i].z);
        if (x_diff > ACCURACY || y_diff > ACCURACY || z_diff > ACCURACY) {
            correct = 0;
            break;
        }
    }

    if (!correct)
        fprintf(stderr, "Data differs!\n");
}

int load_header(const char *filename, int *num_systems, int *bodies_per_system, int is_golden) {
    int ut_num_systems, ut_bodies_per_systems;
    FILE *fp = fopen(filename, "rb");
    if (fp) {
        if (fread(&ut_num_systems, sizeof(int), 1, fp) != 1 ||
            fread(&ut_bodies_per_systems, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "Error: failed to read golden header\n");
            fclose(fp);
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Error: failed to read golden header\n");
        return EXIT_FAILURE;
    }

    if (is_golden) {
        *num_systems = ut_num_systems;
        *bodies_per_system = ut_bodies_per_systems;
    } else if (ut_num_systems != *num_systems || ut_bodies_per_systems != *bodies_per_system) {
        fprintf(stderr, "Error: under test header doesn't match golden header\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    fclose(fp);
    return EXIT_SUCCESS;
}

int load_data(const char *filename, Body *data, int total_bodies) {
    FILE *fp = fopen(filename, "rb");
    size_t nread = fread(data, sizeof(Body), (size_t)total_bodies, fp);
    if (nread != (size_t)total_bodies) {
        fprintf(stderr,
                "Error: expected %d bodies, read %zu\n",
                total_bodies, nread);
        fclose(fp);
        return EXIT_FAILURE;
    }
    fclose(fp);
    return EXIT_SUCCESS;
}

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems;       	/* Number of independent galaxies */
    int bodies_per_system;	/* Number of bodies per galaxy */
    int total_bodies;

    Body *golden_data, *data_under_test;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <golden_data_file> <data_under_test_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (load_header(argv[1], &num_systems, &bodies_per_system, 1) == EXIT_FAILURE)
        return EXIT_FAILURE;

    /* Allocate memory for ALL systems */
    total_bodies = num_systems * bodies_per_system;
    golden_data = (Body *) malloc(total_bodies * sizeof(Body));
    data_under_test = (Body *) malloc(total_bodies * sizeof(Body));

    /* Initialize data */
    if (load_header(argv[2], &num_systems, &bodies_per_system, 0) == EXIT_FAILURE) {
        free(golden_data);
        free(data_under_test);
        return EXIT_FAILURE;
    }
    
    if (load_data(argv[1], golden_data, total_bodies) == EXIT_FAILURE) {
        free(golden_data);
        free(data_under_test);
        return EXIT_FAILURE;
    }

    if (load_data(argv[2], data_under_test, total_bodies) == EXIT_FAILURE) {
        free(golden_data);
        free(data_under_test);
        return EXIT_FAILURE;
    }

    check_data(golden_data, data_under_test, total_bodies);

    free(golden_data);
    free(data_under_test);
    return 0;
}
