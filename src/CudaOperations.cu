#include "Matrix.h"
#include "Spinset.h"
#include "CudaOperations.h"
#include <cuda_runtime.h>

double CudaOperations::getEnergy(Matrix matrix, double* spinset) {
}

__global__ void calcF(double *matrix, double *spinset, int* spinIndex,
		int* size, double *calc) {
	int i = threadIdx.x;
	if (i >= *size || i == *spinIndex)
		return;
	if (i > *spinIndex)
		calc[i] = spinset[i] * matrix[*spinIndex * *size + i];
	else
		calc[i] = spinset[i] * matrix[i * *size + *spinIndex];
}

void checkError(cudaError_t err, string arg = ""){
	if(err != cudaSuccess){
		cout << "Error: "<< cudaGetErrorString(err)<<endl;
		if(arg != "")
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
	checkError(cudaMalloc((void **) &outp, sizeof(double) * matrix.getSize()), "malloc");
	cudaMalloc((void**) &dMat, sizeof(double) * matrix.getSize());
	cudaMalloc((void**) &dSpins,
			sizeof(double) * matrix.getSize() * matrix.getSize());
	cudaMalloc((void**) &dSInd, sizeof(int));
	cudaMalloc((void**) &size, sizeof(int));

	//Copy data
	checkError(cudaMemcpy(dMat, matrix.getArray(), sizeof(double) * matrix.getSize(),
			cudaMemcpyHostToDevice), "memcpy");
	cudaMemcpy(dSpins, spinset,
			sizeof(double) * matrix.getSize() * matrix.getSize(),
			cudaMemcpyHostToDevice);
	cudaMemcpy(dSInd, &spinIndex, sizeof(int), cudaMemcpyHostToDevice);
	int* ss = (int*) malloc(sizeof(int));
	ss[0] = matrix.getSize();
	cudaMemcpy(size, ss, sizeof(int), cudaMemcpyHostToDevice);

	//Start
	calcF<<<1, matrix.getSize()>>>(dMat, dSpins, dSInd, size, outp);

	cudaDeviceSynchronize();
	double result = 0;
	double* ress = new double[matrix.getSize()];
	cudaMemcpy(ress, outp, sizeof(double) * matrix.getSize(),
			cudaMemcpyDeviceToHost);
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
