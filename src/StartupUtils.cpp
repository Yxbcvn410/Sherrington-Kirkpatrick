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
	if (startRef != -1)
		cout << "Current value: " << startRef << ", -c to leave unchanged."
				<< endl;
	cin >> resp;
	if (resp != "-c")
		startRef = stod(resp);
	cout << "Upper temperature limit?" << endl;
	if (endRef != -1)
		cout << "Current value: " << endRef << ", -c to leave unchanged."
				<< endl;
	cin >> resp;
	if (resp != "-c")
		endRef = stod(resp);
	cout << "Calculation step?" << endl;
	if (stepRef != -1)
		cout << "Current value: " << stepRef << ", -c to leave unchanged."
				<< endl;
	cin >> resp;
	if (resp != "-c")
		stepRef = stod(resp);
	cout << "When moving to zero step?" << endl;
	if (pStepRef != -1)
		cout << "Current value: " << pStepRef << ", -c to leave unchanged."
				<< endl;
	cin >> resp;
	if (resp != "-c")
		pStepRef = stod(resp);
	cout << "CUDA block count?" << endl;
	if (blockCountRef != -1)
			cout << "Current value: " << blockCountRef << ", -c to leave unchanged."
					<< endl;
		cin >> resp;
		if (resp != "-c")
			blockCountRef = stod(resp);
	cout << "Matrix file path? (-r to randomize, -b to build)" << endl;
	cin >> resp;
	int ocode = 0;
	if (resp == "-r" || resp == "-R") {
		int msize;
		cout << "Matrix size?" << endl;
		cin >> msize;
		modelRef = Matrix(msize);
		ocode = 1;
	} else if(resp == "-b" || resp == "-B"){
		cout << "Matrix builder file path?" << endl;
		cin >> resp;
		modelRef.buildMat(ifstream(resp));
	}else
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
		bool& randRef, string& wDirRef, bool& useCLI, string confLocation) {
	ifstream ifs;
	ifs.open(confLocation);
	if (ifs.good())
		cout << "Config file detected, stay calm..." << endl;
	else {
		cout << "Error 00: Config file not detected.\n"
				<< "Is it in the working directory?" << endl;
		return -1;
	}
	string buf;
	bool mkDir = false;
	bool cStep = false;
	double pCount = 1;
	int needSave = 1;
	ifs >> buf;
	while (ifs.peek() != EOF) {
		if (buf == "&start") {
			ifs >> startRef;
		} else if (buf == "&end" || buf == "&e") {
			ifs >> endRef;
		} else if (buf == "&step") {
			ifs >> stepRef;
			cStep = false;
		} else if (buf == "&c" || buf == "&count") {
			ifs >> pCount;
			cStep = true;
		} else if (buf == "&pstep" || buf == "&ps") {
			ifs >> pStepRef;
		} else if (buf == "&bc" || buf == "&bcount") {
			ifs >> blockCountRef;
		} else if (buf == "&msize" || buf == "&ms") {
			int size;
			ifs >> size;
			modelRef = Matrix(size);
			modelRef.Randomize();
			needSave = 1;
		} else if (buf == "&wdir" || buf == "&wd" || buf == "&dir") {
			ifs >> wDirRef;
			if (wDirRef == "-a" || wDirRef == "-A") {
				mkDir = true;
				wDirRef = "";
			} else
				mkDir = false;
		} else if (buf == "&mloc" || buf == "&ml" || buf == "&ml_b"
				|| buf == "&mloc_b") {
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
			if (buf == "&ml_b" || buf == "&mloc_b")
				modelRef.buildMat(ifstream(loc));
			else
				modelRef = Matrix(ifstream(loc));
		} else if (buf == "&ird" || buf == "&irand" || buf == "&initrd"
				|| buf == "&initrand") {
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
		} else if (buf == "&cli" || buf == "&usecli") {
			string buf;
			ifs >> buf;
			if (buf == "true" || buf == "t")
				useCLI = true;
			else if (buf == "false" || buf == "f")
				useCLI = false;
			else {
				cout << "Error 05 in launch config:\n"
						<< "Bad word after &cli: " << buf << endl;
				return -1;
			}
		} else if (buf[0] == '#') {
			string buf;
			getline(ifs, buf);
		} else {
			cout << "Error 01 in launch config:\n" << "Unknown word: " << buf
					<< endl;
			return -1;
		}
		ifs >> buf;
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
		stepRef = (endRef - startRef) / pCount;
	cout << "Config parsing complete, success." << endl;
	return needSave;
}
