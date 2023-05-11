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
    dim3 numBlocks((NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, (NUMENTITIES + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pairwise_acc<<<numBlocks, threadsPerBlock>>>(d_hPos, d_hVel, d_mass, d_accels);
    sum_acc<<<(NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock>>>(d_accels, d_accel_sums);

    cudaMemcpy(accel_sums, d_accel_sums, sizeof(vector3) * NUMENTITIES, cudaMemcpy);

        // Copy accels to device
    cudaMemcpy(d_accels, accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyHostToDevice);

    // Compute pairwise accelerations
    pairwise_acc<<<numBlocks, threadsPerBlock>>>(d_hPos, d_hVel, d_mass, d_accels);

    // Sum accelerations for each entity
    sum_acc<<<(NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock>>>(d_accels, d_accel_sums);

    // Copy accel_sums back to host
    cudaMemcpy(accel_sums, d_accel_sums, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    // Update velocities and positions
    for (int i = 0; i < NUMENTITIES; i++){
        for (int j = 0; j < 3; j++){
            hVel[i][j] += accel_sums[i][j] * DT;
            hPos[i][j] += hVel[i][j] * DT;
        }
    }

    // Free device memory
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_accels);
    cudaFree(d_accel_sums);
}

int main(){
    // Allocate host memory
    hPos = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    hVel = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    mass = (double*)malloc(sizeof(double) * NUMENTITIES);
    accel_sums = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);

    // Initialize entities
    for (int i = 0; i < NUMENTITIES; i++){
        FILL_VECTOR(hPos[i], (double)rand()/(double)RAND_MAX - 0.5, (double)rand()/(double)RAND_MAX - 0.5, (double)rand()/(double)RAND_MAX - 0.5);
        FILL_VECTOR(hVel[i], 0, 0, 0);
        mass[i] = (double)rand()/(double)RAND_MAX * MASS_RANGE + MIN_MASS;
    }

    // Run simulation
    for (int i = 0; i < NUMSTEPS; i++){
        compute();
    }

    // Free host memory
    free(hPos);
    free(hVel);
    free(mass);
    free(accel_sums);

    return 0;
}