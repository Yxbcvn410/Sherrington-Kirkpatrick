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

class Matrix {
private:
	int size;
	float* matrix;
	int* unemptyMat;
	float sum;
public:
	Matrix(int size);
	Matrix(ifstream fs);
	void buildMat(ifstream ifs);
	void Randomize();
	string getMatrix();
	int getSize();
	float* getArray();
	float getSum();
	int* getUnemptyMat();
};

#endif /* MATRIX_H_ */
