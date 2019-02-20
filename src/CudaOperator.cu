#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperator.h"
#include <cuda_runtime.h>
#include <sstream>
#include <math.h>

void checkError(cudaError_t err, string arg = "") {
	if (err != cudaSuccess) {
		cout << "Error: " << cudaGetErrorString(err) << endl;
		if (arg != "")
			cout << "Additional data: " << arg << endl;
		std::exit(-1);
	}
}

CudaOperator::CudaOperator(Matrix matrix) {
	// Set pointers to null
	devSpins = NULL;
	devMat = NULL;
	meanFieldElems = NULL;
	devSize = NULL;
	energyElems = NULL;
	delta = NULL;
	devTemp = NULL;

	size = matrix.getSize();
	blockSize = 512;

	// Allocate memory for pointers at GPU
	checkError(cudaMalloc((void**) &meanFieldElems, sizeof(double) * size),
			"malloc");
	cudaMalloc((void**) &devMat, sizeof(double) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(double) * size);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &energyElems, sizeof(double) * size * size);
	cudaMalloc((void**) &energy, sizeof(double));
	cudaMalloc((void**) &devTemp, sizeof(double));
	cudaMalloc((void**) &delta, sizeof(double));

	// Copy model data to GPU memory
	checkError(
			cudaMemcpy(devMat, matrix.getArray(), sizeof(double) * size * size,
					cudaMemcpyHostToDevice), "memcpy mat to host");
	cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceProp deviceProp;
	checkError(cudaGetDeviceProperties(&deviceProp, 0), "getProp");
	blockSize = deviceProp.maxThreadsPerBlock;
}

void CudaOperator::cudaLoadSpinset(Spinset spinset) {
	checkError(
			cudaMemcpy(devSpins, spinset.getArray(), sizeof(double) * size,
					cudaMemcpyHostToDevice), "memcpy spinset to device");
	cudaMemcpy(devTemp, &(spinset.temp), sizeof(double),
			cudaMemcpyHostToDevice);
	temp = spinset.temp;
}

void CudaOperator::cudaClear() {
	//Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(meanFieldElems);
	cudaFree(devSize);
	cudaFree(devTemp);
	cudaFree(energy);
	cudaFree(energyElems);
	cudaFree(delta);
}

__global__ void quickSumEnergy(double* devMat, double* devSpins, int* size,
		double* output, int thCount, double* energyTempor) {
	int thrId = threadIdx.x;
	int i;
	int j;

	int wIndex = thrId;
	while (wIndex < *size * *size) {
		i = wIndex % *size;
		j = (int) (wIndex / *size);
		energyTempor[wIndex] = devSpins[i] * devSpins[j] * devMat[wIndex];
		wIndex = wIndex + thCount;
	}
	__syncthreads();

	int offset = 1;
	while (offset < *size * *size) {
		wIndex = thrId;
		while ((wIndex * 2 + 1) * offset < *size * *size) {
			energyTempor[wIndex * 2 * offset] += energyTempor[(wIndex * 2 + 1)
					* offset];
			wIndex = wIndex + thCount;
		}
		offset *= 2;
		__syncthreads();
	}

	if (thrId == 0)
		*output = energyTempor[0];
}

double CudaOperator::extractEnergy() {
	quickSumEnergy<<<1, blockSize>>>(devMat, devSpins, devSize, energy,
			blockSize, energyElems);
	cudaDeviceSynchronize();
	double out;
	checkError(cudaMemcpy(&out, energy, sizeof(double), cudaMemcpyDeviceToHost),
			"memcpy energy to host");
	return out;
}

Spinset CudaOperator::extractSpinset() {
	double* hSpins = (double*) malloc(sizeof(double) * size);
	checkError(
			cudaMemcpy(hSpins, devSpins, sizeof(double) * size,
					cudaMemcpyDeviceToHost), "memcpy spins to host");
	Spinset outSpins(size);
	for (int i = 0; i < size; i++)
		outSpins.SetSpin(i, hSpins[i]);
	return outSpins;
}

__global__ void cudaStabilize(double* mat, double* spins, int* size,
		double* temp, int thCount, double* meanFieldElements, double* diff,
		int* itC) {
	// Invoke with div3(size)
	int thrId = threadIdx.x;
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
			int wIndex = thrId;
			while (wIndex < *size) {
				if (wIndex > spinId)
					meanFieldElements[wIndex] = mat[spinId * *size + wIndex]
							* spins[wIndex];
				else
					meanFieldElements[wIndex] = mat[wIndex * *size + spinId]
							* spins[wIndex];
				wIndex = wIndex + thCount;
			}
			__syncthreads();

			// Parallelized mean-field computation
			double force = 0;
			int offset = 1;
			while (offset < *size) {
				wIndex = thrId;
				while ((wIndex * 2 + 1) * offset < *size) {
					meanFieldElements[wIndex * 2 * offset] +=
							meanFieldElements[(wIndex * 2 + 1) * offset];
					wIndex = wIndex + thCount;
				}
				offset *= 2;
				__syncthreads();
			}
			if (thrId == 0)
				force = meanFieldElements[0];

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

void CudaOperator::cudaPull(double pStep) {
	cudaMemcpy(devTemp, &temp, sizeof(double), cudaMemcpyHostToDevice);
	int* itC;
	cudaMalloc((void**) &itC, sizeof(int));
	cudaStabilize<<<1, blockSize>>>(devMat, devSpins, devSize, devTemp,
			blockSize, meanFieldElems, delta, itC);
	cudaDeviceSynchronize();
	do {
		temp -= pStep;
		checkError(
				cudaMemcpy(devTemp, &temp, sizeof(double),
						cudaMemcpyHostToDevice), "memcpy temperature");
		cudaStabilize<<<1, blockSize>>>(devMat, devSpins, devSize, devTemp,
				blockSize, meanFieldElems, delta, itC);
		cudaDeviceSynchronize();
	} while (temp > 0);

}
