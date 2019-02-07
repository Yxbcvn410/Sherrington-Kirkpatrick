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

namespace CudaOperations{
double getEnergy(Matrix matrix, double* spinset);
double getForce(Matrix matrix, double* spinset, int spinIndex);
void iterateSpinset(Matrix matrix, Spinset spinset);
}



#endif /* CUDAOPERATIONS_H_ */
