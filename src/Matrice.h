/*
 * Matrix.h
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#ifndef MATRIX_H_
#define MATRIX_H_
#include <iostream>
#include <fstream>
using namespace std;

class Matrice {
private:
	int size;
	float* matrix;
	int* unemptyMat;
	float sum;
public:
	Matrice(int size);
	Matrice(ifstream fs);
	void buildMat(ifstream ifs);
	void Randomize();
	string getMatrix();
	int getSize();
	float* getArray();
	float getSum();
	int* getUnemptyMat();
};

#endif /* MATRIX_H_ */
