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

int StartupUtils::grabInteractive(long double& startRef, long double& endRef,
		long double& stepRef, double& pStepRef, Matrix& modelRef,
		int& blockCountRef, bool& randRef, string& wDirRef) {
	cout << "Do you want to init randomizer? (yes/no) ";
	string resp = "";
	cin >> resp;
	randRef = false;
	bool mkDir = false;
	if (resp == "yes" || resp == "y") {
		randRef = true;
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
	} else if (resp == "-b" || resp == "-B") {
		cout << "Matrix builder file path?" << endl;
		cin >> resp;
		modelRef.buildMat(ifstream(resp));
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

int StartupUtils::grabFromString(string inp, long double& startRef,
		long double& endRef, long double& stepRef, double& pStepRef,
		Matrix& modelRef, int& blockCountRef, bool& randRef, string& wDirRef) {
	istringstream ifs = istringstream(inp);
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
			string loc;
			ifs >> loc;
			ifstream mfs;
			mfs.open(loc);
			if (!mfs.good()) {
				cout << "Error 02 while parsing:\n"
						<< "No matrix file found at specified location."
						<< endl;
				return -1;
			}
			if (buf == "&ml_b" || buf == "&mloc_b") {
				modelRef.buildMat(ifstream(loc));
				needSave = 2;
			} else {
				modelRef = Matrix(ifstream(loc));
				needSave = 0;
			}
		} else if (buf == "&ird" || buf == "&irand" || buf == "&initrd"
				|| buf == "&initrand") {
			string buf;
			ifs >> buf;
			if (buf == "true" || buf == "t")
				randRef = true;
			else if (buf == "false" || buf == "f")
				randRef = false;
			else {
				cout << "Error 04 while parsing:\n" << "Bad word after &irand: "
						<< buf << endl;
				return -1;
			}
		} else if (buf[0] == '#') {
			string buf;
			getline(ifs, buf);
		} else {
			cout << "Error 01 while parsing:\n" << "Unknown word: " << buf
					<< endl;
			return -1;
		}
		ifs >> buf;
	}
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
	return needSave;
}
