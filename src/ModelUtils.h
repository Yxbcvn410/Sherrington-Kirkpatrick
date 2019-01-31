/*
 * ModelFrontend.h
 *
 *  Created on: Jan 19, 2019
 *      Author: alexander
 */
#include "Matrix.h"
#include "Spinset.h"
#ifndef MODELFRONTEND_H_
#define MODELFRONTEND_H_

namespace ModelUtils {
void Stabilize(Matrix matrix, Spinset spinset, double coef, double requiredAccu);
void Stabilize(Matrix matrix, Spinset spinset);
void PullToZeroTemp(Matrix matrix, Spinset spinset, double step);
void PullToZeroTemp(Matrix matrix, Spinset spinset);
void RoundSpins(Spinset spinset);
}

#endif /* MODELFRONTEND_H_ */
