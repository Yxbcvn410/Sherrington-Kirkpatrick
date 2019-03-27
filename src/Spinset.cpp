/*
 * Spinset.cpp
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#include "Spinset.h"
#include "Matrix.h"
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

void Spinset::Randomize(bool bin) {
	for (int i = 0; i < size; ++i) {
		if (bin)
			if (rand() > 0.5)
				spins[i] = 1;
			else
				spins[i] = -1;
		else
			spins[i] = (rand() / (float) RAND_MAX) * 2 - 1;
	}
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
