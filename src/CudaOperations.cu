#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"
#include <cuda_runtime.h>
#include <math.h>

//GPU memory pointers
double* devSpins = NULL; //Spinset
double* devMat = NULL; //Matrix
double* forceElems = NULL; //Temporary storage for counting force
int* devSize = NULL; //Size
double* devTemp = NULL; //Temperature
double* diff = NULL;
double* energyMat1 = NULL;
double* energyMat2 = NULL;
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
	energyMat1 = NULL;
	energyMat2 = NULL;
	diff = NULL;
	devTemp = NULL;

	size = matrix.getSize();

	// Allocate memory for pointers at GPU
	checkError(cudaMalloc((void**) &forceElems, sizeof(double) * size),
			"malloc");
	cudaMalloc((void**) &devMat, sizeof(double) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(double) * size);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &energyMat1, sizeof(double) * size * size);
	cudaMalloc((void**) &energyMat2, sizeof(double) * size);
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
	cout << "spinset loaded" << endl << spinset.getSpins() << endl << temp
			<< endl;
}

void CudaOperations::cudaClear() {
	//Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(forceElems);
	cudaFree(devSize);
	cudaFree(devTemp);
	cudaFree(energy);
	cudaFree(energyMat1);
	cudaFree(energyMat2);
	cudaFree(diff);
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
	calcEnergy3<<<1, 1>>>(energyMat2, energy, devSize);
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
		*itC = -1;
		return;
	}
	if (thrId == 1)
	*itC = 0;

	while (true) {
		//Iterate on all spins
		if (thrId == 0) {
			*diff = 0;
			*itC++;
		}
		for (int spinId = 0; spinId < *size; ++spinId) {
			forceElements[thrId] = mat[spinId * *size + thrId];
			__syncthreads();

			//Here you will be able to see the hellish code for calculating the sum of an array in log(N) time
			if (thrId == 0) {
				// Calculate force...
				double force = 0;
				for (int i = 0; i < *size; ++i) {
					force += forceElements[i];
				}

				// Calculate new spin...
				double old = spins[spinId];
				if (*temp > 0)
					spins[spinId] = -1 * tanh(force / *temp);
				else if (force > 0)
					spins[spinId] = -1;
				else if (force < 0)
					spins[spinId] = 1;
				else
					spins[spinId] = 0;

				//And refresh diff
				if (*diff < abs(old - spins[spinId]))
					*diff = abs(old - spins[spinId]);
			}
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
	int ill;
	cudaMemcpy(&ill, itC, sizeof(int), cudaMemcpyDeviceToHost);
	cout << ill;
	do {
		temp -= pStep;
		checkError(
				cudaMemcpy(devTemp, &temp, sizeof(double),
						cudaMemcpyHostToDevice), "memcpy temperature");
		cudaStabilize<<<1, size>>>(devMat, devSpins, devSize, devTemp,
				forceElems, diff, itC);
		cudaDeviceSynchronize();
		int ill;
		cudaMemcpy(&ill, itC, sizeof(int), cudaMemcpyDeviceToHost);
		cout << ill;
	} while (temp > 0);
	double* spinset = new double[size];
	cudaMemcpy(spinset, devSpins, sizeof(double) * size,
			cudaMemcpyDeviceToHost);
	cout << "stable:" << endl;
	for (int i = 0; i < size; i++)
		cout << spinset[i] << " ";
	cout << endl;
}
