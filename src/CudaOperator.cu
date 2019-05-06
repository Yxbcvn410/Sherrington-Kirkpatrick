#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperator.h"
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

CudaOperator::CudaOperator(Matrix matrix, int blockCnt, float _minDiff) {
	minDiff = _minDiff;
	// Set pointers to null
	devSpins = NULL;
	devMat = NULL;
	devUnemptyMat = NULL;
	meanFieldElems = NULL;
	energyElems = NULL;
	continueIteration = NULL;
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
					sizeof(float) * size * size * blockCount), "malloc");
	cudaMalloc((void**) &devMat, sizeof(float) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(float) * size * blockCount);
	cudaMalloc((void**) &devUnemptyMat, sizeof(int) * size * (size + 1));
	cudaMalloc((void**) &energyElems, sizeof(double) * size * size);
	cudaMalloc((void**) &devTemp, sizeof(float) * blockCount);
	cudaMalloc((void**) &continueIteration, sizeof(bool) * blockCnt);

	// Copy model data to GPU memory
	checkError(
			cudaMemcpy(devMat, matrix.getArray(), sizeof(float) * size * size,
					cudaMemcpyHostToDevice), "memcpy mat to host");
	cudaMemcpy(devUnemptyMat, matrix.getUnemptyMat(),
			sizeof(int) * size * (size + 1), cudaMemcpyHostToDevice);
}

void CudaOperator::cudaLoadSpinset(Spinset spinset, int index) {
	checkError(
			cudaMemcpy(&devSpins[index * size], spinset.getArray(),
					sizeof(float) * size, cudaMemcpyHostToDevice),
			"memcpy spinset to device");
	cudaMemcpy(&devTemp[index], &(spinset.temp), sizeof(float),
			cudaMemcpyHostToDevice);
}

void CudaOperator::cudaClear() {
	//Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(meanFieldElems);
	cudaFree(devTemp);
	cudaFree(devUnemptyMat);
	cudaFree(energyElems);
	cudaFree(continueIteration);
}

__global__ void allocHamiltonian(float* devMat, float* devSpins, int index,
		int size, double* energyTempor) {
	int i;
	int j;

	int wIndex = threadIdx.x;
	while (wIndex < size * size) {
		i = wIndex % size;
		j = (int) (wIndex / size);
		energyTempor[wIndex] = (double) (devSpins[i + index * size]
				* devSpins[j + index * size] * devMat[wIndex]);
		wIndex = wIndex + blockDim.x;
	}
}

__global__ void quickSum(double* energyTempor, int size) {
	long long offset = 1;
	int wIndex;
	while (offset < size * size) {
		wIndex = threadIdx.x;
		while ((wIndex * 2 + 1) * offset < size * size) {
			energyTempor[wIndex * 2 * offset] += energyTempor[(wIndex * 2 + 1)
					* offset];
			wIndex = wIndex + blockDim.x;
		}
		offset *= 2;
		__syncthreads();
	}
}

double CudaOperator::extractHamiltonian(int index) {
	allocHamiltonian<<<1, blockSize>>>(devMat, devSpins, index, size,
			energyElems);
	quickSum<<<1, blockSize>>>(energyElems, size);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	checkError(err, "Kernel at extractEnergy");
	double out;
	checkError(
			cudaMemcpy(&out, energyElems, sizeof(double),
					cudaMemcpyDeviceToHost), "memcpy energy to host");
	return out;
}

Spinset CudaOperator::extractSpinset(int index) {
	float* hSpins = (float*) malloc(sizeof(float) * size);
	checkError(
			cudaMemcpy(hSpins, &devSpins[index * size], sizeof(float) * size,
					cudaMemcpyDeviceToHost), "memcpy spins to host");
	Spinset outSpins(size);
	for (int i = 0; i < size; i++)
		outSpins.SetSpin(i, hSpins[i]);
	return outSpins;
}

__global__ void cudaKernelPull(float* mat, float* spins, int size, float* temp,
		float tempStep, float* meanFieldElements, bool* continueIteration,
		float minDiff, int* unemptyCells) {
	int blockId = blockIdx.x;
	int thrId = threadIdx.x;

	while (temp[blockId] > 0) {
		//Stabilize
		do {
			// No continue by default
			if (thrId == 0)
				continueIteration[blockId] = false;

			__syncthreads();

			//START
			for (int spinId = 0; spinId < size; spinId++) {
				int wIndex = 0;
				while (wIndex < size * size) {
					meanFieldElements[wIndex + blockId * size * size] =
							spins[wIndex + blockId * size] * mat[wIndex];
					wIndex = wIndex + 1;
				}
				__syncthreads();
				if (thrId == 0) {
					for (int i = 0; i < size; ++i) {
						meanFieldElements[spinId * size + blockId * size * size] +=
								meanFieldElements[i + spinId * size
										+ blockId * size * size];
					}
				}

				float old = spins[spinId + blockId * size];
				spins[spinId + blockId * size] = -1
						* tanh(
								meanFieldElements[spinId * size
										+ blockId * size * size]
										/ temp[blockId]);

				// Refresh delta
				if (minDiff < fabs(old - spins[spinId + blockId * size]))
					continueIteration[blockId] = true;

			}
			__syncthreads();
		} while (continueIteration[blockId]);
		//Lessen temperature
		if (thrId == 0)
			temp[blockId] = temp[blockId] - tempStep;
	}
}

__global__ void cudaKernelToFinal(float* mat, float* spins, int size,
		float* meanFieldElements, bool* continueIteration) {
	int blockId = blockIdx.x;
	int thrId = threadIdx.x;

	continueIteration[blockId] = true;
	//Stabilize
	while (continueIteration[blockId]) {
		if (thrId == 0)
			continueIteration[blockId] = false;

		for (int spinId = 0; spinId < size; ++spinId) {
			__syncthreads();

			// Transitional value assignment
			int wIndex = thrId;
			while (wIndex < size) {
				meanFieldElements[wIndex + blockId * size * size] = spins[wIndex
						+ blockId * size]
						* ((wIndex > spinId) ?
								mat[spinId * size + wIndex] :
								mat[wIndex * size + spinId]);
				wIndex = wIndex + blockDim.x;
			}
			__syncthreads();

			// Parallelized mean-field computation
			float meanField = 0;
			long long offset = 1;
			while (offset < size) {
				wIndex = thrId;
				while ((wIndex * 2 + 1) * offset < size) {
					meanFieldElements[wIndex * 2 * offset
							+ blockId * size * size] +=
							meanFieldElements[(wIndex * 2 + 1) * offset
									+ blockId * size * size];
					wIndex = wIndex + blockDim.x;
				}
				offset *= 2;
				__syncthreads();
			}
			if (thrId == 0)
				meanField = meanFieldElements[blockId * size * size];

			// Mean-field calculation complete - write new spin and delta
			if (thrId == 0) {
				float old = spins[spinId + blockId * size];
				if (meanField > 0)
					spins[spinId + blockId * size] = -1;
				else
					spins[spinId + blockId * size] = 1;

				// Refresh delta
				if (0.1 < fabs(old - spins[spinId + blockId * size]))
					continueIteration[blockId] = true;
			}
			__syncthreads();
		}

		__syncthreads();
	}

}

void CudaOperator::cudaPull(float pStep) {
	cudaKernelPull<<<blockCount, blockSize>>>(devMat, devSpins, size, devTemp,
			pStep, meanFieldElems, continueIteration, minDiff, devUnemptyMat);
	cudaKernelToFinal<<<blockCount, blockSize>>>(devMat, devSpins, size, meanFieldElems, continueIteration);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	checkError(err, "Kernel at cudaPull");
}
