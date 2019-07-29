/*
 * Matrice.h
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#ifndef MATRICE_H_
#define MATRICE_H_
#include <iostream>
#include <fstream>
using namespace std;

class Matrice {
private:
	int size;
	float* matrice;
	int* unemptyMat;
	float sum;
public:
	Matrice(int size);
	Matrice(ifstream fs);
	void buildMat(ifstream ifs);
	void Randomize();
	string getMatriceText();
	int getSize();
	float* getArray();
	float getSum();
	int* getUnemptyMat();
};

#endif /* MATRICE_H_ */
