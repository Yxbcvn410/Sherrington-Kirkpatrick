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
	double* devSpins = NULL; //Spinset
	double* devMat = NULL; //Matrix
	double* devTemp = NULL; //Temperature
	double* meanFieldElems = NULL; //Temporary storage for force computation
	double* delta = NULL;
	double* energyElems = NULL; //Temporary storage for energy computation
	//CPU variables
	int size;
	int blockSize;
	int blockCount;
public:
	CudaOperator(Matrix matrix, int blockCount);
	void cudaLoadSpinset(Spinset spinset, int index);
	void cudaPull(double pStep);
	double extractEnergy(int index);
	Spinset extractSpinset(int index);
	void cudaClear();
};

#endif /* CUDAOPERATIONS_H_ */
