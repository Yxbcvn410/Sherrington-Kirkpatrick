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
#include <cmath>
#include "StartupUtils.h"
#include "FilesystemProvider.h"

using namespace std;
using namespace FilesystemProvider;

int StartupUtils::grabFromString(string inp, long double& startRef,
		long double& endRef, long& pointCountRef, double& pStepRef,
		Matrix& matrixRef, int& blockCountRef, string& wDirRef, bool& cliRef,
		float& minDiffRef, int& appendConfigRef, float& linearCoefRef) {
	istringstream ifs = istringstream(inp);
	string buffer;
	bool countDefinedAsStep = false;
	double cStep = 0.001;
	ifs >> buffer;
	while (ifs.peek() != EOF) {
		if (buffer == "%start") {
			ifs >> startRef;
		} else if (buffer == "%end" || buffer == "%e") {
			ifs >> endRef;
		} else if (buffer == "%step") {
			ifs >> cStep;
			countDefinedAsStep = true;
		} else if (buffer == "%md" || buffer == "%mdiff"
				|| buffer == "%mindiff") {
			ifs >> minDiffRef;
		} else if (buffer == "%c" || buffer == "%count") {
			ifs >> pointCountRef;
			countDefinedAsStep = false;
		} else if (buffer == "%lincoef" || buffer == "%lc") {
			ifs >> linearCoefRef;
		} else if (buffer == "%pstep" || buffer == "%ps") {
			ifs >> pStepRef;
		} else if (buffer == "%bc" || buffer == "%bcount") {
			ifs >> blockCountRef;
		} else if (buffer == "%msize" || buffer == "%ms") {
			int size;
			ifs >> size;
			matrixRef = Matrix(size);
			matrixRef.Randomize();
		} else if (buffer == "%wdir" || buffer == "%wd" || buffer == "%dir") {
			ifs >> wDirRef;
		} else if (buffer == "%mloc" || buffer == "%ml" || buffer == "%ml_b"
				|| buffer == "%mloc_b") {
			string loc;
			ifs >> loc;
			ifstream mfs;
			mfs.open(loc);
			if (!mfs.good()) {
				cout << "InputParser: FATAL ERROR: "
						<< "No matrix file found at specified location. Program will be terminated."
						<< endl;
				return -1;
			} else if (buffer == "%ml_b" || buffer == "%mloc_b") {
				cout << "InputParser: MESSAGE: Started building matrix."
						<< endl;
				matrixRef.buildMat(ifstream(loc));
			} else {
				cout << "InputParser: MESSAGE: Started building matrix."
						<< endl;
				matrixRef = Matrix(ifstream(loc));
			}
			cout << "InputParser: MESSAGE: Matrix loaded successfully" << endl;
		} else if (buffer == "%ird" || buffer == "%irand" || buffer == "%initrd"
				|| buffer == "%initrand") {
			string buf;
			ifs >> buf;
			if (buf == "true" || buf == "t"){
				srand(time(0));
				cout << "InputParser: MESSAGE: Random initialized." << endl;
			}
			else if (buf == "false" || buf == "f")
				srand(0);
			else {
				cout << "InputParser: CRITICAL WARNING: "
						<< "Bad word after %irand: " << buf << ", line ignored."
						<< endl;
			}
		} else if (buffer == "%cli") {
			ifs >> buffer;
			if (buffer == "true" || buffer == "t")
				cliRef = true;
			else if (buffer == "false" || buffer == "f")
				cliRef = false;
			else {
				cout << "InputParser: CRITICAL WARNING: "
						<< "Bad word after %cli: " << buffer
						<< ", line ignored." << endl;
			}
		} else if (buffer == "%ac" || buffer == "%appconf") {
			ifs >> buffer;
			if (buffer == "none" || buffer == "n" || buffer == "false" || buffer == "f")
				appendConfigRef = 0;
			else if (buffer == "upper" || buffer == "u")
				appendConfigRef = 1;
			else if (buffer == "both" || buffer == "b")
				appendConfigRef = 2;
			else {
				cout << "InputParser: CRITICAL WARNING: "
						<< "Bad word after %appconf: " << buffer
						<< ", line ignored." << endl;
			}
		} else if (buffer[0] == '#') {
			getline(ifs, buffer);
		} else {
			cout << "InputParser: CRITICAL WARNING: " << "Unknown word: \""
					<< buffer << "\", ignored" << endl;
		}
		ifs >> buffer;
	}
	if (countDefinedAsStep)
		pointCountRef = round((endRef - startRef) / cStep);
	int exitCode = 0;
	if (wDirRef == "") {
		cout << "InputParser: WARNING: " << "Directory not defined" << endl;
		exitCode = 1;
	}
	if (endRef == -1) {
		cout << "InputParser: WARNING: "
				<< "Upper temperature limit not defined" << endl;
		exitCode = 1;
	}
	if (blockCountRef == -1) {
		cout << "InputParser: WARNING: " << "CUDA block count not defined"
				<< endl;
		exitCode = 1;
	}
	if (matrixRef.getSize() == 2) {
		cout << "InputParser: WARNING: " << "Matrix not defined" << endl;
		exitCode = 1;
	}
	return exitCode;
}
