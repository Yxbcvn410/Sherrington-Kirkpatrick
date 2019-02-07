#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"

double CudaOperations::getEnergy(Matrix matrix, double* spinset) {
}

__global__ void calcF(double *matrix, double *spinset, int spinIndex, int size,
		double *calc) {
	int i = threadIdx.x;
	if (i >= size || i == spinIndex)
		return;
	if (i > spinIndex)
		calc[spinIndex] = spinset[spinIndex] * matrix[spinIndex * size + i];
	else
		calc[spinIndex] = spinset[spinIndex] * matrix[i * size + spinIndex];
}

double CudaOperations::getForce(Matrix matrix, double* spinset, int spinIndex) {
	//Define
	double* outp = NULL;
	double* dMat = NULL;
	double* dSpins = NULL;
	int dSInd = NULL;
	int size = NULL;

	//Allocate memory in GPU
	cudaError_t r = cudaMalloc((void **)&outp, sizeof(double) * matrix.getSize());
	cout << r;
	cudaMalloc(&dMat, sizeof(double) * matrix.getSize());
	cudaMalloc(&dSpins, sizeof(double) * matrix.getSize() * matrix.getSize());
	cudaMalloc((void**) &dSInd, sizeof(int));
	cudaMalloc((void**) &size, sizeof(int));

	//Copy data
	cudaMemcpy(dMat, matrix.getArray(), sizeof(dMat), cudaMemcpyHostToDevice);
	cudaMemcpy(dSpins, spinset, sizeof(dSpins), cudaMemcpyHostToDevice);
	cudaMemcpy(&dSInd, &spinIndex, sizeof(dSInd), cudaMemcpyHostToDevice);
	int ss = matrix.getSize();
	cudaMemcpy(&size, &ss, sizeof(size), cudaMemcpyHostToDevice);

	//Start
	calcF<<<1, matrix.getSize()>>>(dMat, dSpins, dSInd, size, outp);

	cudaDeviceSynchronize();
	double result = 0;
	double* ress = new double[matrix.getSize()];
	cudaMemcpy(ress, outp, sizeof(double) * matrix.getSize(),
			cudaMemcpyDeviceToHost);
	for (int i = 0; i < matrix.getSize(); i++) {
		result += ress[i];
	}

	//Free memory of GPU
	cudaFree(outp);
	cudaFree(dMat);
	cudaFree(dSpins);
	cudaFree(&dSInd);
	cudaFree(&size);
	return result;
}
