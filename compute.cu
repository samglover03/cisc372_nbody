#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void compute_kernel(double* hPos, double* hVel, double* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUMENTITIES) {
        return;
    }

    // Make an acceleration matrix which is NUMENTITIES squared in size
    double* values = (double*)malloc(sizeof(double) * NUMENTITIES * NUMENTITIES);
    double** accels = (double**)malloc(sizeof(double*) * NUMENTITIES);
    for (int j = 0; j < NUMENTITIES; j++) {
        accels[j] = &values[j * NUMENTITIES];
    }

    // First compute the pairwise accelerations. Effect is on the first argument.
    for (int j = 0; j < NUMENTITIES; j++) {
        if (i == j) {
            for (int k = 0; k < 3; k++) {
                accels[i][j * 3 + k] = 0;
            }
        } else {
            double distance[3];
            for (int k = 0; k < 3; k++) {
                distance[k] = hPos[i * 3 + k] - hPos[j * 3 + k];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            for (int k = 0; k < 3; k++) {
                accels[i][j * 3 + k] = accelmag * distance[k] / magnitude;
            }
        }
    }

    // Sum up the rows of our matrix to get effect on each entity, then update velocity and position
    double accel_sum[3] = {0};
    for (int j = 0; j < NUMENTITIES; j++) {
        for (int k = 0; k < 3; k++) {
            accel_sum[k] += accels[i][j * 3 + k];
        }
    }
    // Compute the new velocity based on the acceleration and time interval
    // Compute the new position based on the velocity and time interval
    for (int k = 0; k < 3; k++) {
        hVel[i * 3 + k] += accel_sum[k] * INTERVAL;
        hPos[i * 3 + k] += hVel[i * 3 + k] * INTERVAL;
    }

    free(accels);
    free(values);
}

void compute() {
    int i, j, k;
    double distance_x, distance_y, distance_z, magnitude_sq, magnitude, accelmag;
    vector3 accel_sum = {0, 0, 0};
    vector3 *d_accels, *h_values;
    vector3 **d_accels_ptrs, **h_accels_ptrs;

    // Allocate memory for host and device matrices
    h_values = (vector3 *) malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    h_accels_ptrs = (vector3 **) malloc(sizeof(vector3 *) * NUMENTITIES);
    cudaMalloc(&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc(&d_accels_ptrs, sizeof(vector3 *) * NUMENTITIES);

    // Set up pointers for 2D array on host and device
    for (i = 0; i < NUMENTITIES; i++) {
        h_accels_ptrs[i] = &h_values[i * NUMENTITIES];
    }
    cudaMemcpy(d_accels_ptrs, h_accels_ptrs, sizeof(vector3 *) * NUMENTITIES, cudaMemcpyHostToDevice);

    // Compute pairwise accelerations on device
    dim3 grid(NUMENTITIES, 1, 1);
    dim3 block(NUMENTITIES, 1, 1);
    pairwise_accel<<<grid, block>>>(d_accels, d_accels_ptrs, hPos, mass, GRAV_CONSTANT, NUMENTITIES);

    // Copy results from device to host
    cudaMemcpy(h_values, d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);

    // Compute total acceleration for each entity and update position and velocity
    for (i = 0; i < NUMENTITIES; i++) {
        accel_sum.x = 0;
        accel_sum.y = 0;
        accel_sum.z = 0;
        for (j = 0; j < NUMENTITIES; j++) {
            distance_x = hPos[i].x - hPos[j].x;
            distance_y = hPos[i].y - hPos[j].y;
            distance_z = hPos[i].z - hPos[j].z;
            magnitude_sq = distance_x * distance_x + distance_y * distance_y + distance_z * distance_z;
            if (i != j) {
                magnitude = sqrt(magnitude_sq);
                accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
                accel_sum.x += accelmag * distance_x / magnitude;
                accel_sum.y += accelmag * distance_y / magnitude;
                accel_sum.z += accelmag * distance_z / magnitude;
            }
        }
        // Update velocity and position on host
        hVel[i].x += accel_sum.x * INTERVAL;
        hVel[i].y += accel_sum.y * INTERVAL;
        hVel[i].z += accel_sum.z * INTERVAL;
        hPos[i].x += hVel[i].x * INTERVAL;
        hPos[i].y += hVel[i].y * INTERVAL;
        hPos[i].z += hVel[i].z * INTERVAL;
    }

    // Free memory on host and device
    free(h_values);
    free(h_accels_ptrs);
    cudaFree(d_accels);
    cudaFree(d_accels_ptrs);
}