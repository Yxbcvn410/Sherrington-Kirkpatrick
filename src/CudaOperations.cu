#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"
#include <cuda_runtime.h>
#include <sstream>
#include <math.h>

//GPU memory pointers
double* devSpins = NULL; //Spinset
double* devMat = NULL; //Matrix
int* devSize = NULL; //Size
double* devTemp = NULL; //Temperature
double* meanFieldElems = NULL; //Temporary storage for force computation
double* delta = NULL;
double* energyMat = NULL; //Temporary storage for energy computation
double* energy = NULL;

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
	meanFieldElems = NULL;
	devSize = NULL;
	energyMat = NULL;
	delta = NULL;
	devTemp = NULL;

	size = matrix.getSize();

	// Allocate memory for pointers at GPU
	checkError(cudaMalloc((void**) &meanFieldElems, sizeof(double) * size),
			"malloc");
	cudaMalloc((void**) &devMat, sizeof(double) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(double) * size);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &energyMat, sizeof(double) * size * size);
	cudaMalloc((void**) &energy, sizeof(double));
	cudaMalloc((void**) &devTemp, sizeof(double));
	cudaMalloc((void**) &delta, sizeof(double));

	// Copy model data to GPU memory
	cudaMemcpy(devMat, matrix.getArray(), sizeof(double) * size * size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);
}

void CudaOperations::cudaLoadSpinset(Spinset spinset) {
	checkError(
			cudaMemcpy(devSpins, spinset.getArray(), sizeof(double) * size,
					cudaMemcpyHostToDevice));
	cudaMemcpy(devTemp, &(spinset.temp), sizeof(double),
			cudaMemcpyHostToDevice);
	temp = spinset.temp;
}

void CudaOperations::cudaClear() {
	//Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(meanFieldElems);
	cudaFree(devSize);
	cudaFree(devTemp);
	cudaFree(energy);
	cudaFree(energyMat);
	cudaFree(delta);
}

__global__ void quickSumEnergy(double* devMat, double* devSpins, int* size, double* output, double* energyTempor) {
	// Invoke with dim3(size, size)
	int i = threadIdx.x, j = threadIdx.y, thrId = i * *size + j;
		energyTempor[thrId] = devSpins[i] * devSpins[j]
				* devMat[thrId];

	int offset = 1;
	while (offset < *size * *size) {
		if ((thrId * 2 + 1) * offset < *size * *size)
			energyTempor[thrId * 2 * offset] += energyTempor[(thrId * 2 + 1) * offset];
		offset *= 2;
		__syncthreads();
	}
	if (thrId == 0)
		*output = energyTempor[0];
}

double CudaOperations::extractEnergy() {
	quickSumEnergy<<<1, dim3(size, size)>>>(devMat, devSpins, devSize, energy, energyMat);
	cudaDeviceSynchronize();
	double out;
	checkError(cudaMemcpy(&out, energy, sizeof(double), cudaMemcpyDeviceToHost),
			"memcpy energy to host");
	return out;
}

Spinset CudaOperations::extractSpinset() {
	double* hSpins = (double*)malloc(sizeof(double)*size);
	checkError(cudaMemcpy(hSpins, devSpins, sizeof(double)*size, cudaMemcpyDeviceToHost), "memcpy spins to host");
	Spinset outSpins(size);
	for(int i = 0; i < size; i++)
		outSpins.SetSpin(i, hSpins[i]);
	return outSpins;
}

__global__ void cudaStabilize(double* mat, double* spins, int* size,
		double* temp, double* forceElements, double* diff, int* itC) {
	// Invoke with div3(size)
	int thrId = threadIdx.x;
	if (thrId >= *size) {
		return;
	}
	if (thrId == 0)
		*itC = 0;

	while (true) {
		__syncthreads();
		//Iterate on all spins
		if (thrId == 0) {
			*diff = 0;
			*itC = *itC + 1;
		}

		for (int spinId = 0; spinId < *size; ++spinId) {
			__syncthreads();
			if (thrId > spinId)
				forceElements[thrId] = mat[spinId * *size + thrId]
						* spins[thrId];
			else
				forceElements[thrId] = mat[thrId * *size + spinId]
						* spins[thrId];
			__syncthreads();

			// Parallelized mean-field computation
			double force = 0;
			int offset = 1;
			while (offset < *size) {
				if ((thrId * 2 + 1) * offset < *size)
					forceElements[thrId * 2 * offset] += forceElements[(thrId
							* 2 + 1) * offset];
				offset *= 2;
				__syncthreads();
			}
			if (thrId == 0)
				force = forceElements[0];

			// Mean-field calculation complete - write new spin and delta
			if (thrId == 0) {
				double old = spins[spinId];
				if (*temp > 0) {
					spins[spinId] = -1 * tanh(force / *temp);
				} else if (force > 0)
					spins[spinId] = -1;
				else if (force < 0)
					spins[spinId] = 1;
				else
					spins[spinId] = 0;

				// Refresh delta
				if (*diff < fabs(old - spins[spinId]))
					*diff = fabs(old - spins[spinId]);
			}
			__syncthreads();
		}

		__syncthreads();
		if (*diff < 0.000001)
			return; // diff link is same for all threads; Terminate all if diff is appropriate
	}
}

void CudaOperations::cudaPull(double pStep) {
	cudaMemcpy(devTemp, &temp, sizeof(double), cudaMemcpyHostToDevice);
	int* itC;
	cudaMalloc((void**) &itC, sizeof(int));
	cudaStabilize<<<1, size>>>(devMat, devSpins, devSize, devTemp, meanFieldElems,
			delta, itC);
	cudaDeviceSynchronize();
	do {
		temp -= pStep;
		checkError(
				cudaMemcpy(devTemp, &temp, sizeof(double),
						cudaMemcpyHostToDevice), "memcpy temperature");
		cudaStabilize<<<1, size>>>(devMat, devSpins, devSize, devTemp,
				meanFieldElems, delta, itC);
		cudaDeviceSynchronize();
	} while (temp > 0);

}
