#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"
#include <cuda_runtime.h>
#include <cmath>

//GPU memory pointers
double* devSpins = NULL;
double* devMat = NULL;
double* forceElems = NULL;
int* devSize = NULL;
int* devSpinIndex = NULL;

//CPU variables
double* hForceElems;
double temp;
int size;

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock) {
	while (atomicCAS((int *) lock, 0, 1) != 0)
		;
}

__device__ void release_semaphore(volatile int *lock) {
	*lock = 0;
	__threadfence();
}

void checkError(cudaError_t err, string arg = "") {
	if (err != cudaSuccess) {
		cout << "Error: " << cudaGetErrorString(err) << endl;
		if (arg != "")
			cout << "Additional data: " << arg << endl;
		std::exit(-1);
	}
}

void CudaOperations::cudaInit(Matrix matrix) {
	// Set pointers to null
	devSpins = NULL;
	devMat = NULL;
	forceElems = NULL;
	devSize = NULL;
	devSpinIndex = NULL;

	size = matrix.getSize();

	// Allocate memory for pointers at GPU
	checkError(cudaMalloc((void**) &forceElems, sizeof(double) * modelSize),
			"malloc");
	cudaMalloc((void**) &oForce, sizeof(double));
	cudaMalloc((void**) &devMat, sizeof(double) * modelSize * modelSize);
	cudaMalloc((void**) &devSpins, sizeof(double) * modelSize);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &devSpinIndex, sizeof(int));

	//Allocate memory for CPU pointers
	hForceElems = (double*) malloc(sizeof(double) * size);

	// Copy model data to GPU memory
	cudaMemcpy(devMat, matrix.getArray(), sizeof(double) * size * size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);
}

void CudaOperations::cudaLoadSpinset(Spinset spinset) {
	checkError(
			cudaMemcpy(devSpins, spinset.getArray(), sizeof(double) * size,
					cudaMemcpyHostToDevice));
	temp = spinset.temp;
}

void CudaOperations::cudaClear() {
	//Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(forceElems);
	cudaFree (oForces);
	cudaFree(devSize);
}

double extractPreferredVal() {
	checkError(
			cudaMemcpy(hForceElems, forceElems, sizeof(double) * size,
					cudaMemcpyDeviceToHost), "Memcpy force elements");
	double force = 0;
	for (int i = 0; i < size; i++)
		force += hForceElems[i];
	if (temp <= 0) {
		if (getForce(index, matrix) > 0)
			return -1;
		else if (getForce(index, matrix) < 0)
			return 1;
		else
			return 0;
	} else
		return tanh((-1 * getForce(index, matrix)) / temp);
}

__global__ void cuGetForce() {
	int i = threadIdx.x;
	if (i == *devSpinIndex || i >= devSize)
		return;
	if (i < *devSpinIndex)
		forceElems[i] = devSpins[i] * devMat[i * *devSize + *devSpinIndex];
	else
		forceElems[i] = devSpins[i] * devMat[*devSpinIndex * *devSize + i];
}

double CudaOperations::cudaIterate() {
	double diff = 0;
	int spinIndex = 0;
	for (spinIndex = 0; spinIndex < size; ++spinIndex) {
		cudaMemcpy(devSpinIndex, &spinIndex, sizeof(int),
				cudaMemcpyHostToDevice);

	}
}
