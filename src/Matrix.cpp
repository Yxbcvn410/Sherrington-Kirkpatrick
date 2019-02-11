/*
 * Matrix.cpp
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#include "Matrix.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

Matrix::Matrix(int size) {
	this->size = size;
	matrix = new double[size * size];
}

Matrix::Matrix(string s) {
	istringstream iss(s);
	int ss;
	iss >> ss;
	size = ss;
	matrix = new double[size * size];
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			iss >> matrix[i * size + j];
		}
	}
}

Matrix::Matrix(ifstream fs) {
	int ss;
	fs >> ss;
	size = ss;
	matrix = new double[size * size];
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			fs >> matrix[i * size + j];
		}
	}
}

void Matrix::Randomize() {
	double f;
	for (int i = 0; i < size; ++i) {
		for (int j = i + 1; j < size; ++j) {
			f = rand() / (double) RAND_MAX;
			f = f * 2 - 1;
			matrix[i * size + j] = f;
		}
	}
}

int Matrix::getSize() const{
	int s;
	s = size;
	return s;
}

string Matrix::getMatrix() {
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

const double Matrix::getCell(int x, int y) {
	return matrix[x * size + y];
}

const double* Matrix::getArray(){
	return matrix;
}
