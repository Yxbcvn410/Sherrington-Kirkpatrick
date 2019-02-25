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
#include "Matrix.h"
using namespace std;

class Spinset {
private:
	int size;
	double* spins;
public:
	double temp;
	Spinset(int size);
	void Randomize(bool bin);
	void SetSpin(int index, double value);
	const string getSpins();
	const double* getArray();
};

#endif /* SPINSET_H_ */
