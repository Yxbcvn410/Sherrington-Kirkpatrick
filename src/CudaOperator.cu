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

CudaOperator::CudaOperator(Matrix matrix, int blockCnt) {
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
	blockCount = blockCnt;

	cudaDeviceProp deviceProp;
	checkError(cudaGetDeviceProperties(&deviceProp, 0), "getProp");
	blockSize = deviceProp.maxThreadsPerBlock;

	// Allocate memory for pointers at GPU
	checkError(
			cudaMalloc((void**) &meanFieldElems,
					sizeof(double) * size * blockCount), "malloc");
	cudaMalloc((void**) &devMat, sizeof(double) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(double) * size * blockCount);
	cudaMalloc((void**) &devSize, sizeof(int));
	cudaMalloc((void**) &energyElems, sizeof(double) * size * size);
	cudaMalloc((void**) &devTemp, sizeof(double) * blockCount);
	cudaMalloc((void**) &delta, sizeof(double));

	// Copy model data to GPU memory
	checkError(
			cudaMemcpy(devMat, matrix.getArray(), sizeof(double) * size * size,
					cudaMemcpyHostToDevice), "memcpy mat to host");
	cudaMemcpy(devSize, &size, sizeof(int), cudaMemcpyHostToDevice);
}

void CudaOperator::cudaLoadSpinset(Spinset spinset, int index) {
	checkError(
			cudaMemcpy(&devSpins[index * size], spinset.getArray(),
					sizeof(double) * size, cudaMemcpyHostToDevice),
			"memcpy spinset to device");
	cudaMemcpy(&devTemp[index], &(spinset.temp), sizeof(double),
			cudaMemcpyHostToDevice);
}

void CudaOperator::cudaClear() {
	//Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(meanFieldElems);
	cudaFree(devSize);
	cudaFree(devTemp);
	cudaFree(energyElems);
	cudaFree(delta);
}

__global__ void quickSumEnergy(double* devMat, double* devSpins, int index,
		int* size, int thCount, double* energyTempor) {
	int thrId = threadIdx.x;
	int i;
	int j;

	int wIndex = thrId;
	while (wIndex < *size * *size) {
		i = wIndex % *size;
		j = (int) (wIndex / *size);
		energyTempor[wIndex] = devSpins[i + index * *size]
				* devSpins[j + index * *size] * devMat[wIndex];
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
}

double CudaOperator::extractEnergy(int index) {
	quickSumEnergy<<<1, blockSize>>>(devMat, devSpins, index, devSize,
			blockSize, energyElems);
	cudaDeviceSynchronize();
	double out;
	checkError(
			cudaMemcpy(&out, energyElems, sizeof(double),
					cudaMemcpyDeviceToHost), "memcpy energy to host");
	return out;
}

Spinset CudaOperator::extractSpinset(int index) {
	double* hSpins = (double*) malloc(sizeof(double) * size);
	checkError(
			cudaMemcpy(hSpins, &devSpins[index * blockCount],
					sizeof(double) * size, cudaMemcpyDeviceToHost),
			"memcpy spins to host");
	Spinset outSpins(size);
	for (int i = 0; i < size; i++)
		outSpins.SetSpin(i, hSpins[i]);
	return outSpins;
}

__global__ void cudaKernelPull(double* mat, double* spins, int* size,
		double* temp, double tempStep, int thCount, double* meanFieldElements,
		double* diff) {
	int blockId = blockIdx.x;
	int thrId = threadIdx.x;

	bool flag;
	while (temp[blockId] > 0) {
		//Lessen temperature
		temp[blockId] = temp[blockId] - tempStep;
		//Stabilize
		flag = true;
		while (flag) {
			__syncthreads();
			//Iterate on all spins
			if (thrId == 0)
				*diff = 0;

			for (int spinId = 0; spinId < *size; ++spinId) {
				__syncthreads();
				int wIndex = thrId;
				while (wIndex < *size) {
					if (wIndex > spinId)
						meanFieldElements[wIndex + blockId * *size] = mat[spinId
								* *size + wIndex]
								* spins[wIndex + blockId * *size];
					else
						meanFieldElements[wIndex + blockId * *size] = mat[wIndex
								* *size + spinId]
								* spins[wIndex + blockId * *size];
					wIndex = wIndex + thCount;
				}
				__syncthreads();

				// Parallelized mean-field computation
				double force = 0;
				int offset = 1;
				while (offset < *size) {
					wIndex = thrId;
					while ((wIndex * 2 + 1) * offset < *size) {
						meanFieldElements[wIndex * 2 * offset + blockId * *size] +=
								meanFieldElements[(wIndex * 2 + 1) * offset
										+ blockId * *size];
						wIndex = wIndex + thCount;
					}
					offset *= 2;
					__syncthreads();
				}
				if (thrId == 0)
					force = meanFieldElements[blockId * *size];

				// Mean-field calculation complete - write new spin and delta
				if (thrId == 0) {
					double old = spins[spinId + blockId * *size];
					if (temp[blockId] > 0) {
						spins[spinId + blockId * *size] = -1
								* tanh(force / temp[blockId]);
					} else if (force > 0)
						spins[spinId + blockId * *size] = -1;
					else if (force < 0)
						spins[spinId + blockId * *size] = 1;
					else
						spins[spinId + blockId * *size] = 0;

					// Refresh delta
					if (*diff < fabs(old - spins[spinId + blockId * *size]))
						*diff = fabs(old - spins[spinId + blockId * *size]);
				}
				__syncthreads();
			}

			__syncthreads();
			if (*diff < 0.000001)
				flag = false; // diff link is same for all threads; Abort stabilization if diff is appropriate
		}
	}
}

void CudaOperator::cudaPull(double pStep) {
	double* pullSt = NULL;
	checkError(cudaMalloc((void**) &pullSt, sizeof(double)), "tMalloc");
	cudaMemcpy(pullSt, &pStep, sizeof(double), cudaMemcpyHostToDevice);
	cudaKernelPull<<<blockCount, blockSize>>>(devMat, devSpins, devSize,
			devTemp, pStep, blockSize, meanFieldElems, delta);
	cudaDeviceSynchronize();
}
