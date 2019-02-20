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
void cudaInit(Matrix matrix);
void cudaSetBlock();
void cudaLoadSpinset(Spinset spinset);
void cudaPull(double pStep);
double extractEnergy();
Spinset extractSpinset();
void cudaClear();
}



#endif /* CUDAOPERATIONS_H_ */
