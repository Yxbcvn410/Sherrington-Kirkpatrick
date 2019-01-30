/*
 * GlobalUtils.cpp
 *
 *  Created on: Jan 29, 2019
 *      Author: alexander
 */

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <unistd.h>

#include "StartupUtils.h"
using namespace std;

bool StartupUtils::grabFromCLI(double& startRef, double& endRef, double& stepRef,
		Matrix& modelRef, int& thrCountRef, string& wDirRef) {
	double tmp1;
	cout << "Working dir? (-c to use current working directory)\n"
			<< getCurrentWorkingDir() << endl;
	cin >> wDirRef;
	if (wDirRef == "-c" || wDirRef == "-C")
		wDirRef = getCurrentWorkingDir();
	cout << "Lower temperature limit?" << endl;
	cin >> tmp1;
	startRef = tmp1;
	cout << "Upper temperature limit?" << endl;
	cin >> tmp1;
	endRef = tmp1;
	cout << "Calculation step?" << endl;
	cin >> stepRef;
	cout << "Thread count?" << endl;
	cin >> thrCountRef;
	cout << "Model file? (-r to randomize)" << endl;
	string resp;
	cin >> resp;
	if (resp == "-r" || resp == "-R") {
		int msize;
		cout << "Model size?" << endl;
		cin >> msize;
		modelRef = Matrix(msize);
		modelRef.Randomize();
		return true;
	} else
		modelRef = Matrix(ifstream(resp));
	return false;
}

bool StartupUtils::grabFromFile(double& startRef, double& endRef,
		double& stepRef, Matrix& modelRef, int& thrCountRef, string& wDirRef,
		string confLocation) {
	ifstream ifs;
	ifs.open(confLocation);
	if (ifs.good())
		cout << "Config file detected, stay calm..." << endl;
	else {
		cout << "Config file not detected, error." << endl;
		return false;
	}
	int prg = 6;
	string s;
	bool needSave;
	while (prg > 0) {
		ifs >> s;
		if (s == "&start") {
			ifs >> startRef;
		} else if (s == "&end") {
			ifs >> endRef;
		} else if (s == "&step") {
			ifs >> stepRef;
		} else if (s == "&tc"||s == "&tcount") {
			ifs >> thrCountRef;
		} else if (s == "&msize"||s=="&ms") {
			int size;
			ifs >> size;
			modelRef = Matrix(size);
			modelRef.Randomize();
			needSave = true;
		} else if (s == "&wdir" || s == "&wd") {
			ifs >> wDirRef;
		} else if (s == "&mloc") {
			needSave = false;
			string loc;
			ifs >> loc;
			ifstream mfs;
			mfs.open(loc);
			if (!mfs.good()) {
				cout << "Error while parsing config:\n"
						<< "Bad matrix file location." << endl;
				return false;
			}
			modelRef = Matrix(ifstream(loc));
		} else {
			cout << "Error while parsing config:\n"
					<< "Unknown command at config: " << s << endl;
			return false;
		}
		prg -= 1;
	}
	cout << "Config parsing complete, success." << endl;
	return needSave;
}

string StartupUtils::getCurrentWorkingDir() {
	char buff[FILENAME_MAX];
	getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}
