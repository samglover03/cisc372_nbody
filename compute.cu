#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define BLOCK_SIZE 256

__global__ void compute_kernel(vector3* dPos, vector3* dVel, double* dMass, int numEntities, double interval) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numEntities) {
        vector3 accel_sum = {0, 0, 0};
        for (int j = 0; j < numEntities; j++) {
            if (i != j) {
                vector3 distance;
                for (int k = 0; k < 3; k++) {
                    distance[k] = dPos[j][k] - dPos[i][k];
                }
                double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
                double magnitude = sqrt(magnitude_sq);
                double accelmag = -1 * GRAV_CONSTANT * dMass[j] / magnitude_sq;
                accel_sum[0] += accelmag * distance[0] / magnitude;
                accel_sum[1] += accelmag * distance[1] / magnitude;
                accel_sum[2] += accelmag * distance[2] / magnitude;
            }
        }
        dVel[i][0] += accel_sum[0] * interval;
        dVel[i][1] += accel_sum[1] * interval;
        dVel[i][2] += accel_sum[2] * interval;
        dPos[i][0] += dVel[i][0] * interval;
        dPos[i][1] += dVel[i][1] * interval;
        dPos[i][2] += dVel[i][2] * interval;
    }
}

void compute(){
    //make an acceleration matrix which is NUMENTITIES squared in size;
    int i, j, k;
    vector3* values = (vector3*) malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    vector3 **d_accels, **h_accels = (vector3**) malloc(sizeof(vector3*) * NUMENTITIES);
    cudaMalloc((void**) &d_accels, sizeof(vector3*) * NUMENTITIES);
    cudaMalloc((void**) &h_accels[0], sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    for (i = 0; i < NUMENTITIES; i++) {
        h_accels[i] = &h_accels[0][i * NUMENTITIES];
        cudaMemcpy(d_accels + i, h_accels + i, sizeof(vector3*), cudaMemcpyHostToDevice);
    }
    //first compute the pairwise accelerations.  Effect is on the first argument.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((NUMENTITIES + dimBlock.x - 1) / dimBlock.x, (NUMENTITIES + dimBlock.y - 1) / dimBlock.y);
    pairwise_accel<<<dimGrid, dimBlock>>>(d_accels, hPos, mass);
    cudaDeviceSynchronize();
    //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
    for (i = 0; i < NUMENTITIES; i++) {
        vector3 accel_sum = {0, 0, 0};
        for (j = 0; j < NUMENTITIES; j++) {
            accel_sum.x += h_accels[i][j].x;
            accel_sum.y += h_accels[i][j].y;
            accel_sum.z += h_accels[i][j].z;
        }
        //compute the new velocity based on the acceleration and time interval
        //compute the new position based on the velocity and time interval
        hVel[i].x += accel_sum.x * INTERVAL;
        hVel[i].y += accel_sum.y * INTERVAL;
        hVel[i].z += accel_sum.z * INTERVAL;
        hPos[i].x += hVel[i].x * INTERVAL;
        hPos[i].y += hVel[i].y * INTERVAL;
        hPos[i].z += hVel[i].z * INTERVAL;
    }
    cudaFree(d_accels);
    cudaFree(h_accels[0]);
    free(h_accels);
    free(values);
}