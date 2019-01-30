/*
 * ModelFrontend.cpp
 *
 *  Created on: Jan 19, 2019
 *      Author: alexander
 */

#include "ModelUtils.h"
#include "Matrix.h"
#include "Spinset.h"
#include <stdio.h>
#include <iostream>
#include <cmath>

double iterate(Matrix matrix, Spinset spinset, double coef) {
	double out = 0;
	for (int i = 0; i < matrix.getSize(); ++i) {
		if (out < std::abs(spinset.getPreferredSpin(i, matrix) - spinset.getSpin(i)))
			out = std::abs(spinset.getPreferredSpin(i, matrix) - spinset.getSpin(i));
		spinset.SetSpin(i,
				spinset.getSpin(i)
						+ (spinset.getPreferredSpin(i, matrix) - spinset.getSpin(i))
								* coef);
	}
	return out;
}

void ModelUtils::Stabilize(Matrix matrix, Spinset spinset, double coef, double requiredAccu) {
	while (iterate(matrix, spinset, coef) > requiredAccu) {
	}
}

void ModelUtils::Stabilize(Matrix matrix, Spinset spinset){
	ModelUtils::Stabilize(matrix, spinset, 0.95, 0.000001);
}

void ModelUtils::PullToZeroTemp(Matrix matrix, Spinset spinset){
	ModelUtils::Stabilize(matrix, spinset);
	while (spinset.temp>0){
		spinset.temp-=0.005;
		ModelUtils::Stabilize(matrix, spinset);
	}
	spinset.temp=0;
}
