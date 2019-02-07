/*
 * Spinset.h
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#ifndef SPINSET_H_
#define SPINSET_H_
#include <iostream>
#include <stdio.h>
#include <random>
#include "Matrix.h"
using namespace std;

class Spinset {
private:
	int size;
	double* spins;
	const double getForce(int index, Matrix matrix);
	mt19937 random;
public:
	double temp;
	Spinset(int size);
	void seed(int seed);
	void Randomize(bool bin);
	void SetSpin(int index, double value);
	const double getEnergy(Matrix matrix);
	const double getPreferredSpin(int index, Matrix matrix);
	const double getSpin(int index);
	const string getSpins();
	const int getSize();
	const double* getArray();
};

#endif /* SPINSET_H_ */
