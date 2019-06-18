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
	float* spins;
public:
	float temp;
	Spinset(int size);
	void Randomize();
	void SetSpin(int index, float value);
	const string getSpins();
	const float* getArray();
};

#endif /* SPINSET_H_ */
