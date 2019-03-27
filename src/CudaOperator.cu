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
					sizeof(float) * size * blockCount), "malloc");
	cudaMalloc((void**) &devMat, sizeof(float) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(float) * size * blockCount);
	cudaMalloc((void**) &energyElems, sizeof(double) * size * size);
	cudaMalloc((void**) &devTemp, sizeof(float) * blockCount);
	cudaMalloc((void**) &delta, sizeof(float) * blockCnt);

	// Copy model data to GPU memory
	checkError(
			cudaMemcpy(devMat, matrix.getArray(), sizeof(float) * size * size,
					cudaMemcpyHostToDevice), "memcpy mat to host");
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
	cudaFree(energyElems);
	cudaFree(delta);
}

__global__ void allocHamiltonian(float* devMat, float* devSpins, int index,
		int size, double* energyTempor) {
	int i;
	int j;

	int wIndex = threadIdx.x;
	while (wIndex < size * size) {
		i = wIndex % size;
		j = (int) (wIndex / size);
		energyTempor[wIndex] = (double)(devSpins[i + index * size]
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

__global__ void cudaKernelPull(float* mat, float* spins, int size,
		float* temp, float tempStep, float* meanFieldElements,
		float* diff) {
	int blockId = blockIdx.x;
	int thrId = threadIdx.x;

	bool flag;
	bool firstrun = true;
	while (temp[blockId] >= 0 || firstrun) {
		firstrun = false;
		//Lessen temperature
		if (thrId == 0)
			temp[blockId] = temp[blockId] - tempStep;
		//Stabilize
		flag = true;
		while (flag) {
			__syncthreads();
			//Iterate on all spins
			if (thrId == 0)
				diff[blockId] = 0;

			for (int spinId = 0; spinId < size; ++spinId) {
				__syncthreads();
				int wIndex = thrId;
				while (wIndex < size) {
					if (wIndex > spinId)
						meanFieldElements[wIndex + blockId * size] = mat[spinId
								* size + wIndex]
								* spins[wIndex + blockId * size];
					else
						meanFieldElements[wIndex + blockId * size] = mat[wIndex
								* size + spinId]
								* spins[wIndex + blockId * size];
					wIndex = wIndex + blockDim.x;
				}
				__syncthreads();

				// Parallelized mean-field computation
				float meanField = 0;
				long long offset = 1;
				while (offset < size) {
					wIndex = thrId;
					while ((wIndex * 2 + 1) * offset < size) {
						meanFieldElements[wIndex * 2 * offset + blockId * size] +=
								meanFieldElements[(wIndex * 2 + 1) * offset
										+ blockId * size];
						wIndex = wIndex + blockDim.x;
					}
					offset *= 2;
					__syncthreads();
				}
				if (thrId == 0)
					meanField = meanFieldElements[blockId * size];

				// Mean-field calculation complete - write new spin and delta
				if (thrId == 0) {
					float old = spins[spinId + blockId * size];
					if (temp[blockId] > 0) {
						spins[spinId + blockId * size] = -1
								* tanh(meanField / temp[blockId]);
					} else if (meanField > 0)
						spins[spinId + blockId * size] = -1;
					else
						spins[spinId + blockId * size] = 1;

					// Refresh delta
					if (diff[blockId]
							< fabs(old - spins[spinId + blockId * size]))
						diff[blockId] = fabs(
								old - spins[spinId + blockId * size]);
				}
				__syncthreads();
			}

			__syncthreads();
			if (diff[blockId] < 0.000001)
				flag = false; // diff link is same for all threads; Abort stabilization if diff is appropriate
		}
	}
}

void CudaOperator::cudaPull(float pStep) {
	cudaKernelPull<<<blockCount, blockSize>>>(devMat, devSpins, size, devTemp,
			pStep, meanFieldElems, delta);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	checkError(err, "Kernel at cudaPull");
}
