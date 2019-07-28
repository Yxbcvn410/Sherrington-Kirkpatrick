/*
 * CudaOperations.h
 *
 *  Created on: Feb 6, 2019
 *      Author: alexander
 */

#ifndef CUDAOPERATIONS_H_
#define CUDAOPERATIONS_H_

#include "Matrice.h"
#include "Spinset.h"

class CudaOperator {
private:
	//GPU pointers
	float* devSpins; //Spinset
	float* devMat; //Matrix
	int* devUnemptyMat; //UnemptyMat field of Matrix object
	float* devTemp; //Temperature
	float* meanFieldElems; //Temporary storage for meanfield computation
	bool* continueIteration;
	double* hamiltonianElems; //Temporary storage for hamiltonian computation
	//CPU variables
	int size;
	int blockSize;
	int blockCount;
	float minDiff;
public:
	CudaOperator(Matrice matrix, int blockCount, float _minDiff);
	void cudaLoadSpinset(Spinset spinset, int index);
	void cudaPull(float pStep, float linearCoef);
	double extractHamiltonian(int index);
	Spinset extractSpinset(int index);
	void cudaClear();
};

#endif /* CUDAOPERATIONS_H_ */
