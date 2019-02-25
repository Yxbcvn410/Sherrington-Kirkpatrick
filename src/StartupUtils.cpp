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
#include "StartupUtils.h"
#include "FilesystemProvider.h"

using namespace std;
using namespace FilesystemProvider;

int StartupUtils::grabFromCLI(double& startRef, double& endRef, double& stepRef,
		double& pStepRef, Matrix& modelRef, int& blockCountRef, bool& randRef,
		string& wDirRef) {
	cout << "Do you want to init randomizer? (yes/no) ";
	string resp = "";
	cin >> resp;
	randRef = false;
	bool mkDir = false;
	if (resp == "yes" || resp == "y") {
		randRef = true;
		cout << "Randomizer initialized." << endl;
	} else
		cout << "Randomizer was not initialized" << "\n";
	cout << "Working dir? (-a to auto-create in current working directory)\n"
			<< getCurrentWorkingDirectory() << endl;
	cin >> wDirRef;
	if (wDirRef == "-a" || wDirRef == "-A") {
		mkDir = true;
		wDirRef = "";
	}
	cout << "Lower temperature limit?" << endl;
	cin >> startRef;
	cout << "Upper temperature limit?" << endl;
	cin >> endRef;
	cout << "Calculation step?" << endl;
	cin >> stepRef;
	cout << "When moving to zero step?" << endl;
	cin >> pStepRef;
	cout << "CUDA block count?" << endl;
	cin >> blockCountRef;
	cout << "Model file? (-r to randomize)" << endl;
	cin >> resp;
	int ocode = 0;
	if (resp == "-r" || resp == "-R") {
		int msize;
		cout << "Model size?" << endl;
		cin >> msize;
		modelRef = Matrix(msize);
		ocode = 1;
	} else
		modelRef = Matrix(ifstream(resp));
	if (mkDir) {
		ostringstream oss;
		oss << "calc" << modelRef.getSize() << "_";
		int dirIndex = FreeFileIndex(getCurrentWorkingDirectory(), oss.str(),
				"", false);
		oss << dirIndex;
		wDirRef = getCurrentWorkingDirectory() + "/" + oss.str();
		makeDirectory(wDirRef, "");
	}
	return ocode;
}

int StartupUtils::grabFromFile(double& startRef, double& endRef,
		double& stepRef, double& pStepRef, Matrix& modelRef, int& blockCountRef,
		bool& randRef, string& wDirRef, string confLocation) {
	ifstream ifs;
	ifs.open(confLocation);
	if (ifs.good())
		cout << "Config file detected, stay calm..." << endl;
	else {
		cout << "Error 00: Config file not detected.\n"
				<< "Is it in the working directory?" << endl;
		return -1;
	}
	string s;
	bool mkDir = false;
	bool cStep = false;
	int needSave = 1;
	ifs >> s;
	while (ifs.peek() != EOF) {
		if (s == "&start") {
			ifs >> startRef;
		} else if (s == "&end" || s == "&e") {
			ifs >> endRef;
		} else if (s == "&step") {
			ifs >> stepRef;
			cStep = false;
		} else if (s == "&c" || s == "&count") {
			ifs >> stepRef;
			cStep = true;
		} else if (s == "&pstep" || s == "&ps") {
			ifs >> pStepRef;
		} else if (s == "&bc" || s == "&bcount") {
			ifs >> blockCountRef;
		} else if (s == "&msize" || s == "&ms") {
			int size;
			ifs >> size;
			modelRef = Matrix(size);
			modelRef.Randomize();
			needSave = 1;
		} else if (s == "&wdir" || s == "&wd" || s == "&dir") {
			ifs >> wDirRef;
			if (wDirRef == "-a" || wDirRef == "-A") {
				mkDir = true;
				wDirRef = "";
			} else
				mkDir = false;
		} else if (s == "&mloc" || s == "&ml" || s == "&ml_b"
				|| s == "&mloc_b") {
			needSave = 0;
			string loc;
			ifs >> loc;
			ifstream mfs;
			mfs.open(loc);
			if (!mfs.good()) {
				cout << "Error 02 in launch config:\n"
						<< "No matrix file found at specified location."
						<< endl;
				return -1;
			}
			if (s == "&ml_b" || s == "&mloc_b")
				modelRef.buildMat(ifstream(loc));
			else
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
				cout << "Error 04 in launch config:\n"
						<< "Bad word after &irand: " << buf << endl;
				return -1;
			}
		} else if (s[0] == '#') {
			string buf;
			getline(ifs, buf);
		} else {
			cout << "Error 01 in launch config:\n" << "Unknown word: " << s
					<< endl;
			return -1;
		}
		ifs >> s;
	}
	if (!randRef)
		cout << "WARNING: Random generator was not initialized." << endl;
	if (mkDir) {
		ostringstream oss;
		oss << "calc" << modelRef.getSize() << "_";
		int dirIndex = FreeFileIndex(getCurrentWorkingDirectory(), oss.str(),
				"", false);
		oss << dirIndex;
		wDirRef = getCurrentWorkingDirectory() + "/" + oss.str();
		makeDirectory(wDirRef, "");
	}
	if (cStep)
		stepRef = (endRef - startRef) / stepRef;
	cout << "Config parsing complete, success." << endl;
	return needSave;
}
