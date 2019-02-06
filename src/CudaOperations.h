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
__global__ double getEnergy(Matrix matrix, Spinset spinset);
__global__ double getForce(Matrix matrix, Spinset spinset, int spinIndex);
}



#endif /* CUDAOPERATIONS_H_ */
