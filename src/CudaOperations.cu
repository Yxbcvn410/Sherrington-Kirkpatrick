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
double* energyMat1 = NULL;
double* energyMat2 = NULL;
double* energy = NULL;
double* dForce = NULL;

//CPU variables
double temp;
int size;

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
	energyMat1 = NULL;
	energyMat2 = NULL;
	dForce = NULL;

	size = matrix.getSize();

	// Allocate memory for pointers at GPU
	checkError(cudaMalloc((void**) &forceElems, sizeof(double) * size),
			"malloc");
	cudaMalloc((void**) &devMat, sizeof(double) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(double) * size);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &devSpinIndex, sizeof(int));
	cudaMalloc((void**) &energyMat1, sizeof(double) * size * size);
	cudaMalloc((void**) &energyMat2, sizeof(double) * size);
	cudaMalloc((void**) &energy, sizeof(double));
	cudaMalloc((void**) &dForce, sizeof(double));

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
	cudaFree(devSpinIndex);
	cudaFree(devSize);
}

double extractPreferredVal() {
	double force = 0;
	checkError(
			cudaMemcpy(&force, dForce, sizeof(double),
					cudaMemcpyDeviceToHost), "Memcpy force");
	if (temp <= 0.0000001) {
		if (force > 0)
			return -1;
		else if (force < 0)
			return 1;
		else
			return 0;
	} else
		return tanh((-1 * force) / temp);
}

__global__ void cuGetForce(double* devMat, double* devSpins, int* devSize,
		int* devSpinIndex, double* forceElems) {
	int i = threadIdx.x;
	if (i >= *devSize)
		return;
	if (i < *devSpinIndex)
		forceElems[i] = devSpins[i] * devMat[i * *devSize + *devSpinIndex];
	else
		forceElems[i] = devSpins[i] * devMat[*devSpinIndex * *devSize + i];
}

__global__ void cuSumForce(double* forceElems, int* devSize, double* force){
	*force = 0;
	for (int i = 0; i < *devSize; ++i) {
		*force += forceElems[i];
	}
}

double setDevSpin(int index, double* value) {
	double old;
	cudaMemcpy(&old, &devSpins[index], sizeof(double), cudaMemcpyDeviceToHost);
	checkError(
			cudaMemcpy(&devSpins[index], value, sizeof(double),
					cudaMemcpyHostToDevice), "memcpyDS");
	return abs(old - *value);
}

double CudaOperations::cudaIterate() {
	double diff = 0;
	int spinIndex = 0;
	double prefSpin;
	for (spinIndex = 0; spinIndex < size; ++spinIndex) {
		cudaMemcpy(devSpinIndex, &spinIndex, sizeof(int),
				cudaMemcpyHostToDevice);
		cuGetForce<<<1, size>>>(devMat, devSpins, devSize, devSpinIndex,
				forceElems);
		cuSumForce<<<1,1>>>(forceElems, devSize, dForce);
		cudaDeviceSynchronize();
		prefSpin = extractPreferredVal();
		double nDiff = setDevSpin(spinIndex, &prefSpin);
		if (nDiff > diff)
			diff = nDiff;
	}
	return diff;
}

void CudaOperations::cudaStabilize() {
	while (CudaOperations::cudaIterate() > 0.000001) {
	}
}

void CudaOperations::cudaPull(double pStep) {
	while (temp > 0) {
		CudaOperations::cudaStabilize();
		temp -= pStep;
	}
	temp = 0;
}

__global__ void calcEnergy1(double* devMat, double* devSpins, int* devSize,
		double* energyMat1) {
	int i = threadIdx.x, j = threadIdx.y;
	energyMat1[i * *devSize + j] = devSpins[i] * devSpins[j]
			* devMat[i * *devSize + j];
}

__global__ void calcEnergy2(double* energyMat1, double* energyMat2,
		int* devSize) {
	int inn = threadIdx.x;
	energyMat2[inn] = 0;
	for (int i = 0; i < *devSize; ++i) {
		energyMat2[inn] += energyMat1[inn * *devSize + i];
	}
}

__global__ void calcEnergy3(double* energyMat2, double* energy, int* devSize) {
	*energy = 0;
	for (int i = 0; i < *devSize; ++i) {
		*energy += energyMat2[i];
	}
}

double CudaOperations::extractEnergy() {
	calcEnergy1<<<1, dim3(size, size)>>>(devMat, devSpins, devSize, energyMat1);
	calcEnergy2<<<1, size>>>(energyMat1, energyMat2, devSize);
	calcEnergy3<<<1,1>>>(energyMat2, energy, devSize);
	cudaDeviceSynchronize();
	double out;
	checkError(cudaMemcpy(&out, energy, sizeof(double), cudaMemcpyDeviceToHost),
			"energy memcpy");
	return out;
}
