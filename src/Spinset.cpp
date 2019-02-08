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

void Spinset::SetSpin(int index, double value) {
	spins[index] = value;
}

void Spinset::Randomize(bool bin) {
	double f;
	for (int i = 0; i < size; ++i) {
		f = (random() / (double) RAND_MAX)/(double)2;
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

void Spinset::seed(int seed){
	random.seed(seed);
}

const int Spinset::getSize() {
	int s;
	s = size;
	return s;
}

const double Spinset::getSpin(int index) {
	double d;
	d = spins[index];
	return d;
}

const string Spinset::getSpins() {
	ostringstream out;
	for (int i = 0; i < size; ++i) {
		out << spins[i] << " ";
	}
	return out.str();
}

const double Spinset::getForce(const int index, Matrix matrix) {
	return CudaOperations::getForce(matrix, spins, index);
	/*double out = 0;
	for (int i = 0; i < size; i++) {
		if (i < index) {
			out += spins[i] * matrix.getCell(i, index);
		} else
			out += spins[i] * matrix.getCell(index, i);
	}
	cout << out << endl;
	string buf;
	cin >> buf;
	return out;//*/
}

const double Spinset::getEnergy(Matrix matrix) {
	double out = 0;
	for (int i = 0; i < size; ++i) {
		for (int j = i + 1; j < size; ++j) {
			out += spins[i] * spins[j] * matrix.getCell(i,j);
		}
	}
	return out;
}

const double Spinset::getPreferredSpin(int index, Matrix matrix) {
	if (temp <= 0) {
		if (getForce(index, matrix) > 0)
			return -1;
		else if (getForce(index, matrix) < 0)
			return 1;
		else
			return 0;
	} else
		return tanh((-1 * getForce(index, matrix)) / temp);
}

const double* Spinset::getArray(){
	return spins;
}
