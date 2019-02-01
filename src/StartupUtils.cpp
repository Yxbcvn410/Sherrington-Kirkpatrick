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

bool StartupUtils::grabFromCLI(double& startRef, double& endRef,
		double& stepRef, double& pStepRef, Matrix& modelRef, int& thrCountRef,
		bool& randRef, string& wDirRef) {
	cout << "Do you want to init randomizer?(yes/no) ";
	string resp = "";
	cin >> resp;
	randRef = false;
	if (resp == "yes" || resp == "y") {
		randRef = true;
		cout << "Randomizer initialized." << endl;
	} else
		cout << "Randomizer was not initialized" << "\n";
	cout << "Working dir? (-c to use current working directory)\n"
			<< getCurrentWorkingDir() << endl;
	cin >> wDirRef;
	if (wDirRef == "-c" || wDirRef == "-C")
		wDirRef = getCurrentWorkingDir();
	cout << "Lower temperature limit?" << endl;
	cin >> startRef;
	cout << "Upper temperature limit?" << endl;
	cin >> endRef;
	cout << "Calculation step?" << endl;
	cin >> stepRef;
	cout << "When moving to zero step?" << endl;
	cin >> pStepRef;
	cout << "Thread count?" << endl;
	cin >> thrCountRef;
	cout << "Model file? (-r to randomize)" << endl;
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
		double& stepRef, double& pStepRef, Matrix& modelRef, int& thrCountRef,
		bool& randRef, string& wDirRef, string confLocation) {
	randRef = true;
	stepRef = 0.001;
	pStepRef = 0.02;
	ifstream ifs;
	ifs.open(confLocation);
	if (ifs.good())
		cout << "Config file detected, stay calm..." << endl;
	else {
		cout << "Config file not detected, error." << endl;
		return false;
	}
	string s;
	bool needSave = false;
	while (ifs.peek() != EOF) {
		ifs >> s;
		if (s == "&start") {
			ifs >> startRef;
		} else if (s == "&end") {
			ifs >> endRef;
		} else if (s == "&step") {
			ifs >> stepRef;
		} else if (s == "&pstep" || s == "&ps") {
			ifs >> pStepRef;
		} else if (s == "&tc" || s == "&tcount") {
			ifs >> thrCountRef;
		} else if (s == "&msize" || s == "&ms") {
			int size;
			ifs >> size;
			modelRef = Matrix(size);
			modelRef.Randomize();
			needSave = true;
		} else if (s == "&wdir" || s == "&wd" || s == "&dir") {
			ifs >> wDirRef;
		} else if (s == "&mloc" || s == "&ml") {
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
		} else if (s == "&ird" || s == "&irand" || s == "&initrd"
				|| s == "&initrand") {
			string buf;
			ifs >> buf;
			if (buf == "true" || buf == "t")
				randRef = true;
			else if (buf == "false" || buf == "f")
				randRef = false;
			else {
				cout << "Error while parsing config:\n"
						<< "Bad word after &irand: " << s << endl;
				return false;
			}
			if (needSave) {
				modelRef.Randomize();
			}
		} else if (s[0] == '#') {
			string buf;
			getline(ifs, buf);
		} else {
			cout << "Error while parsing config:\n"
					<< "Unknown word at config: " << s << endl;
			return false;
		}
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
