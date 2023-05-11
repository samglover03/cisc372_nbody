#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void pairwise_acc(vector3* hPos, vector3* hVel, double* mass, vector3* accels){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES){
        if (i == j){
            FILL_VECTOR(accels[i*NUMENTITIES+j], 0, 0, 0);
        } else {
            vector3 distance;
            for (int k = 0; k < 3; k++){
                distance[k] = hPos[i][k] - hPos[j][k];
                double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
                double magnitude = sqrt(magnitude_sq);
                double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
                FILL_VECTOR(accels[i*NUMENTITIES+j], accelmag * distance[0]/magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
            }
        }
    }
}

__global__ void sum_acc(vector3* accels, vector3* accel_sums){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES){
        vector3 sum = {0,0,0};
        for (int j = 0; j < NUMENTITIES; j++){
            for (int k = 0; k < 3; k++){
                sum[k] += accels[i*NUMENTITIES+j][k];
            }
        }
        accel_sums[i] = sum;
    }
}

void compute(){
    vector3* d_hPos; 
    vector3* d_hVel;
    double* d_mass;

    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);

    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    vector3* d_accels;
    vector3* d_accel_sums;
    cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc((void **)&d_accel_sums, sizeof(vector3) * NUMENTITIES);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, (NUMENTITIES + threadsPerBlock
}