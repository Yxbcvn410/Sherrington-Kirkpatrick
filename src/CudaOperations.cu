#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"
#include <cuda_runtime.h>
#include <sstream>
#include <math.h>

//GPU memory pointers
double* devSpins = NULL; //Spinset
double* devMat = NULL; //Matrix
double* forceElems = NULL; //Temporary storage for counting force
int* devSize = NULL; //Size
double* devTemp = NULL; //Temperature
double* diff = NULL;
double* energyMat = NULL;
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
	forceElems = NULL;
	devSize = NULL;
	energyMat = NULL;
	diff = NULL;
	devTemp = NULL;

	size = matrix.getSize();

	// Allocate memory for pointers at GPU
	checkError(cudaMalloc((void**) &forceElems, sizeof(double) * size),
			"malloc");
	cudaMalloc((void**) &devMat, sizeof(double) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(double) * size);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &energyMat, sizeof(double) * size * size);
	cudaMalloc((void**) &energy, sizeof(double));
	cudaMalloc((void**) &devTemp, sizeof(double));
	cudaMalloc((void**) &diff, sizeof(double));

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
	cudaFree(forceElems);
	cudaFree(devSize);
	cudaFree(devTemp);
	cudaFree(energy);
	cudaFree(energyMat);
	cudaFree(diff);
}

__global__ void calcEnergy1(double* devMat, double* devSpins, int* devSize,
		double* energyMat) {
	int i = threadIdx.x, j = threadIdx.y;
	energyMat[i * *devSize + j] = devSpins[i] * devSpins[j]
			* devMat[i * *devSize + j];
}

__global__ void quickSum(double* mat, int* size, double* output) {
	// Invoke with dim3(size, size)
	int thrId = threadIdx.x * blockDim.x + threadIdx.y;

	int offset = 1;
	while (offset < *size * *size) {
		if ((thrId * 2 + 1) * offset < *size * *size)
			mat[thrId * 2 * offset] += mat[(thrId * 2 + 1) * offset];
		offset *= 2;
		__syncthreads();
	}
	if (thrId == 0)
		*output = mat[0];
}

double CudaOperations::extractEnergy() {
	calcEnergy1<<<1, dim3(size, size)>>>(devMat, devSpins, devSize, energyMat);
	quickSum<<<1, dim3(size, size)>>>(energyMat, devSize, energy);
	cudaDeviceSynchronize();
	double out;
	checkError(cudaMemcpy(&out, energy, sizeof(double), cudaMemcpyDeviceToHost),
			"energy memcpy");
	return out;
}

__global__ void cudaStabilize(double* mat, double* spins, int* size,
		double* temp, double* forceElements, double* diff, int* itC) {
	//Launch in size threads
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
	cudaStabilize<<<1, size>>>(devMat, devSpins, devSize, devTemp, forceElems,
			diff, itC);
	cudaDeviceSynchronize();
	do {
		temp -= pStep;
		checkError(
				cudaMemcpy(devTemp, &temp, sizeof(double),
						cudaMemcpyHostToDevice), "memcpy temperature");
		cudaStabilize<<<1, size>>>(devMat, devSpins, devSize, devTemp,
				forceElems, diff, itC);
		cudaDeviceSynchronize();
	} while (temp > 0);

}
