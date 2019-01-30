/*
 * Spinset.h
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#ifndef SPINSET_H_
#define SPINSET_H_
#include <iostream>
#include "Matrix.h"
using namespace std;

class Spinset {
private:
	int size;
	double* spins;
	double getForce(int index, Matrix matrix);
public:
	double temp;
	Spinset(int size);
	void Randomize(bool bin); //ss
	void SetSpin(int index, double value); //ss
	double getEnergy(Matrix matrix); //ss(m)
	double getPreferredSpin(int index, Matrix matrix); //ss(m)
	double getSpin(int index); //ss
	string getSpins(); //ss
	int getSize(); //m, ss
};

#endif /* SPINSET_H_ */
