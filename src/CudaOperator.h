/*
 * CudaOperations.h
 *
 *  Created on: Feb 6, 2019
 *      Author: root
 */

#ifndef CUDAOPERATIONS_H_
#define CUDAOPERATIONS_H_

#include "Matrix.h"
#include "Spinset.h"

class CudaOperator {
private:
	//GPU pointers
	float* devSpins = NULL; //Spinset
	float* devMat = NULL; //Matrix
	float* devTemp = NULL; //Temperature
	float* meanFieldElems = NULL; //Temporary storage for force computation
	bool* continueIteration = NULL;
	double* energyElems = NULL; //Temporary storage for energy computation
	//CPU variables
	int size;
	int blockSize;
	int blockCount;
	float minDiff;
public:
	CudaOperator(Matrix matrix, int blockCount, float _minDiff);
	void cudaLoadSpinset(Spinset spinset, int index);
	void cudaPull(float pStep);
	double extractHamiltonian(int index);
	Spinset extractSpinset(int index);
	void cudaClear();
};

#endif /* CUDAOPERATIONS_H_ */
