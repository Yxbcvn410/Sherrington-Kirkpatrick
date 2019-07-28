/*
 * Spinset.cpp
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#include "Spinset.h"
#include "Matrice.h"
#include "CudaOperator.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cmath>

Spinset::Spinset(int size) {
	this->size = size;
	spins = new float[size];
	temp = 0;
}

void Spinset::Randomize() {
	for (int i = 0; i < size; ++i)
			spins[i] = (rand() / (float) RAND_MAX) * 2 - 1;
}

void Spinset::SetSpin(int index, float value) {
	spins[index] = value;
}

const string Spinset::getSpins() {
	ostringstream out;
	for (int i = 0; i < size; ++i) {
		out << spins[i] << " ";
	}
	return out.str();
}

const float* Spinset::getArray() {
	return spins;
}
