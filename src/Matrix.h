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
	double* matrix;
public:
	Matrix(int size);
	Matrix(string s);
	Matrix(ifstream fs);
	void Randomize();
	const double getCell(int x, int y);
	string getMatrix();
	 int getSize() const;
	const double* getArray();
};

#endif /* MATRIX_H_ */
