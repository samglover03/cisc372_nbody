#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void compute_kernel(double* hPos, double* hVel, double* mass, vector3* accels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == j){
        FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
        return;
    }
    vector3 distance;
	for (int k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
	double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
	double magnitude=sqrt(magnitude_sq);
	double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	FILL_VECTOR(accels[i * NUMENTITIES + j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
}

__global__ void sum(vector3 *accels, vector3 *accel_sum, vector3 *dPos, vector3 *dVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUMENTITIES) {
        FILL_VECTOR(accel_sum[i], 0, 0, 0);
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
                accel_sum[i][k] += accels[(i * NUMENTITIES) + j][k];
            }
        }
        // Compute the new velocity based on the acceleration and time interval
        // Compute the new position based on the velocity and time interval
        for (int k = 0; k < 3; k++) {
            dVel[i][k] += accel_sum[i][k] * INTERVAL;
            dPos[i][k] += dVel[i][k] * INTERVAL; 
        }
    }
}

void compute() {
    double *dmass;
    vector3 *dhPos, *dhVel, *dacc, *dsum;

	int block = ceilf(NUMENTITIES / 16.0f);
	int thread = ceilf(NUMENTITIES / (float) block);

	dim3 gridDim(block, block, 1);
	dim3 blockDim(thread, thread, 1);

	cudaMalloc((void**) &dmass, sizeof(double) * NUMENTITIES);
	cudaMalloc((void**) &dhPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dhVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dacc, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dsum, sizeof(vector3) * NUMENTITIES);
	
    cudaMemcpy(dmass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dhPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dhVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	
	compute_kernel<<<gridDim, blockDim>>>(dhPos, dhVel, dmass, dacc);
	cudaDeviceSynchronize();

	sum<<<gridDim.x, blockDim.x>>>(dacc, dsum, dhPos, dhVel);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, dhPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dhVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(dhPos);
	cudaFree(dhVel);
	cudaFree(dmass);
	cudaFree(dacc);
}