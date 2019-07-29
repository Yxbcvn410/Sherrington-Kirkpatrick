/*
 * Matrix.cpp
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#include "Matrice.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

Matrice::Matrice(int size) {
	this->size = size;
	matrice = new float[size * size];
	unemptyMat = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
		if (var < size * size)
			matrice[var] = 0.;
	}
	sum = 0;
}

Matrice::Matrice(ifstream fs) {
	int ss;
	fs >> ss;
	size = ss;
	sum = 0;
	matrice = new float[size * size];
	unemptyMat = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
		if (var < size * size)
			matrice[var] = 0.;
	}
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			float cell;
			fs >> cell;
			if (i <= j) {
				matrice[i * size + j] = cell;
				matrice[j * size + i] = cell;
				if (cell != 0.) {
					unemptyMat[i * (size + 1)]++;
					unemptyMat[i * (size + 1) + unemptyMat[i * (size + 1)]] = j;
					unemptyMat[j * (size + 1)]++;
					unemptyMat[j * (size + 1) + unemptyMat[j * (size + 1)]] = i;
				}
				sum += cell;
			}
		}
	}
}

void Matrice::Randomize() {
	float f;
	sum = 0;
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
	}
	for (int i = 0; i < size; ++i) {
		for (int j = i + 1; j < size; ++j) {
			f = rand() / (float) RAND_MAX;
			f = f * 2 - 1;
			matrice[i * size + j] = f;
			matrice[j * size + i] = matrice[i * size + j];
			if (f != 0.) {
				unemptyMat[i * (size + 1)]++;
				unemptyMat[i * (size + 1) + unemptyMat[i * (size + 1)]] = j;
				unemptyMat[j * (size + 1)]++;
				unemptyMat[j * (size + 1) + unemptyMat[j * (size + 1)]] = i;
			}
			sum += matrice[i * size + j];
		}
	}
}

int Matrice::getSize() {
	int s;
	s = size;
	return s;
}

string Matrice::getMatriceText() {
	ostringstream out;
	out << size << "\n";
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			out << matrice[i * size + j] << " ";
		}
		out << "\n";
	}
	return out.str();
}

void Matrice::buildMat(ifstream ifs) {
	sum = 0;
	ifs >> size;
	matrice = new float[size * size];
	unemptyMat = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
		if (var < size * size)
			matrice[var] = 0.;
	}
	int i, j, val, edges;
	ifs >> edges;
	for (int var = 0; var < edges; ++var) {
		ifs >> i;
		ifs >> j;
		ifs >> val;
		i -= 1;
		j -= 1;
		matrice[i * size + j] = val;
		matrice[j * size + i] = val;
		if (val != 0.) {
			unemptyMat[i * (size + 1)]++;
			unemptyMat[i * (size + 1) + unemptyMat[i * (size + 1)]] = j;
			unemptyMat[j * (size + 1)]++;
			unemptyMat[j * (size + 1) + unemptyMat[j * (size + 1)]] = i;
		}
		sum += val;
	}
}

float Matrice::getSum() {
	return sum;
}

float* Matrice::getArray() {
	return matrice;
}

int* Matrice::getUnemptyMat() {
	return unemptyMat;
}
