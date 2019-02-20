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
	int* devSize = NULL; //Size
	double* devTemp = NULL; //Temperature
	double* meanFieldElems = NULL; //Temporary storage for force computation
	double* delta = NULL;
	double* energyElems = NULL; //Temporary storage for energy computation
	double* energy = NULL;
	//CPU variables
	double temp;
	int size;
	int blockSize;
public:
	CudaOperator(Matrix matrix);
	void cudaLoadSpinset(Spinset spinset);
	void cudaPull(double pStep);
	double extractEnergy();
	Spinset extractSpinset();
	void cudaClear();
};

#endif /* CUDAOPERATIONS_H_ */
