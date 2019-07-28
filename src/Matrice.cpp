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
	matrix = new float[size * size];
	unemptyMat = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
		if (var < size * size)
			matrix[var] = 0.;
	}
	sum = 0;
}

Matrice::Matrice(ifstream fs) {
	int ss;
	fs >> ss;
	size = ss;
	sum = 0;
	matrix = new float[size * size];
	unemptyMat = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
		if (var < size * size)
			matrix[var] = 0.;
	}
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			float cell;
			fs >> cell;
			if (i <= j) {
				matrix[i * size + j] = cell;
				matrix[j * size + i] = cell;
				unemptyMat[i * (size + 1)]++;
				unemptyMat[i * (size + 1) + unemptyMat[i * (size + 1)]] = j;
				unemptyMat[j * (size + 1)]++;
				unemptyMat[j * (size + 1) + unemptyMat[j * (size + 1)]] = i;
				sum += cell * 2;
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
			matrix[i * size + j] = f;
			matrix[j * size + i] = matrix[i * size + j];
			unemptyMat[i * (size + 1)]++;
			unemptyMat[i * (size + 1) + unemptyMat[i * (size + 1)]] = j;
			unemptyMat[j * (size + 1)]++;
			unemptyMat[j * (size + 1) + unemptyMat[j * (size + 1)]] = i;
			sum += matrix[i * size + j];
		}
	}
}

int Matrice::getSize() {
	int s;
	s = size;
	return s;
}

string Matrice::getMatrix() {
	ostringstream out;
	out << size << "\n";
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			out << matrix[i * size + j] << " ";
		}
		out << "\n";
	}
	return out.str();
}

void Matrice::buildMat(ifstream ifs) {
	sum = 0;
	ifs >> size;
	matrix = new float[size * size];
	unemptyMat = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		unemptyMat[var] = 0;
		if (var < size * size)
			matrix[var] = 0.;
	}
	int i, j, val, edges;
	ifs >> edges;
	for (int var = 0; var < edges; ++var) {
		ifs >> i;
		ifs >> j;
		ifs >> val;
		i -= 1;
		j -= 1;
		matrix[i * size + j] = val;
		matrix[j * size + i] = val;
		unemptyMat[i * (size + 1)]++;
		unemptyMat[i * (size + 1) + unemptyMat[i * (size + 1)]] = j;
		unemptyMat[j * (size + 1)]++;
		unemptyMat[j * (size + 1) + unemptyMat[j * (size + 1)]] = i;
		sum += val;
	}
}

float Matrice::getSum() {
	return sum;
}

float* Matrice::getArray() {
	return matrix;
}

int* Matrice::getUnemptyMat() {
	return unemptyMat;
}
