/*
 * Spinset.cpp
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#include "Spinset.h"
#include "Matrix.h"
#include "CudaOperations.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cmath>

Spinset::Spinset(int size) {
	this->size = size;
	spins = new double[size];
	temp = 0;
}

void Spinset::Randomize(bool bin) {
	double f;
	for (int i = 0; i < size; ++i) {
		f = (random() / (double) RAND_MAX) / (double) 2;
		f = f * 2 - 1;
		if (bin)
			if (f > 0)
				spins[i] = 1;
			else
				spins[i] = -1;
		else
			spins[i] = f;
	}
}

void Spinset::seed(int seed) {
	random.seed(seed);
}

void Spinset::SetSpin(int index, double value) {
	spins[index] = value;
}

const string Spinset::getSpins() {
	ostringstream out;
	for (int i = 0; i < size; ++i) {
		out << spins[i] << " ";
	}
	return out.str();
}

const double* Spinset::getArray() {
	return spins;
}
