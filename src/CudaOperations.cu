#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"
#include <cuda_runtime.h>

double CudaOperations::getEnergy(Matrix matrix, double* spinset) {
}

__global__ void calcF(double *matrix, double *spinset, int* spinIndex,
		int* size, double *calc) {
	int i = threadIdx.x;
	if (i >= size[0])
		return;
	if (i > spinIndex[0])
		calc[i] = spinset[i] * matrix[(spinIndex[0]) * (size[0]) + i];
	else
		calc[i] = spinset[i] * matrix[i * (size[0]) + spinIndex[0]];
}

void checkError(cudaError_t err, string arg = "") {
	if (err != cudaSuccess) {
		cout << "Error: " << cudaGetErrorString(err) << endl;
		if (arg != "")
			cout << "Additional data: " << arg << endl;
		std::exit(-1);
	}
}

double CudaOperations::getForce(Matrix matrix, double* spinset, int spinIndex) {
	//Define
	double* outp = NULL;
	double* dMat = NULL;
	double* dSpins = NULL;
	int* dSInd = NULL;
	int* size = NULL;

	//Allocate memory in GPU
	checkError(cudaMalloc((void **) &outp, sizeof(double) * matrix.getSize()),
			"malloc");
	cudaMalloc((void**) &dMat, sizeof(double) * matrix.getSize() * matrix.getSize());
	cudaMalloc((void**) &dSpins,
			sizeof(double) * matrix.getSize());
	cudaMalloc((void**) &dSInd, sizeof(int));
	cudaMalloc((void**) &size, sizeof(int));

	//Copy data
	checkError(
			cudaMemcpy(dMat, matrix.getArray(),
					sizeof(double) * matrix.getSize() * matrix.getSize(), cudaMemcpyHostToDevice),
			"memcpy");
	cudaMemcpy(dSpins, spinset,
			sizeof(double) * matrix.getSize(),
			cudaMemcpyHostToDevice);
	cudaMemcpy(dSInd, &(spinIndex), sizeof(int), cudaMemcpyHostToDevice);
	int* ss = (int*) malloc(sizeof(int));
	*ss = matrix.getSize();
	cudaMemcpy(size, ss, sizeof(int), cudaMemcpyHostToDevice);

	//Start
	calcF<<<1, *ss>>>(dMat, dSpins, dSInd, size, outp);

	cudaDeviceSynchronize();
	double result = 0;
	double* ress = new double[*ss];
	checkError(
			cudaMemcpy(ress, outp, sizeof(double) * (*ss),
					cudaMemcpyDeviceToHost), "memcpy back");
	for (int i = 0; i < matrix.getSize(); ++i) {
		result += ress[i];
	}
	//Free memory of GPU
	cudaFree(outp);
	cudaFree(dMat);
	cudaFree(dSpins);
	cudaFree(dSInd);
	cudaFree(size);

	//Free CPU memory too
	free(ress);

	return result;
}
