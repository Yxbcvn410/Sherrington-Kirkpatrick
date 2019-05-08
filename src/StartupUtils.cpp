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

int StartupUtils::grabInteractive(long double& startRef, long double& endRef,
		long& pointCountRef, double& pStepRef, Matrix& matrixRef,
		int& blockCountRef, string& wDirRef, bool& cliRef, float& minDiffRef, bool& appendConfigRef) {
	cliRef = true;
	cout << "Do you want to init randomizer? (yes/no) ";
	string resp = "";
	getline(cin, resp);
	srand(0);
	if (resp == "yes" || resp == "y") {
		srand(time(0));
	} else
		cout << "InputParser: WARNING: Randomizer was not initialized" << "\n";
	if (wDirRef == "") {
		cout
				<< "Working dir? (-a to auto-create in current working directory)\n"
				<< getCurrentWorkingDirectory() << endl;
		getline(cin, wDirRef);
	}
	cout << "Lower temperature limit?" << endl;
	if (startRef != -1)
		cout << "Current value: " << startRef << ", press Enter to leave unchanged."
				<< endl;
	getline(cin, resp);
	if (resp != "")
		startRef = stod(resp);
	cout << "Upper temperature limit?" << endl;
	if (endRef != -1)
		cout << "Current value: " << endRef << ", press Enter to leave unchanged."
				<< endl;
	getline(cin, resp);
	if (resp != "")
		endRef = stod(resp);
	cout << "Point quantity?" << endl;
	if (pointCountRef != -1)
		cout << "Current value: " << pointCountRef << ", press Enter to leave unchanged."
				<< endl;
	getline(cin, resp);
	if (resp != "")
		pointCountRef = stol(resp);
	cout << "When moving to zero step?" << endl;
	if (pStepRef != -1)
		cout << "Current value: " << pStepRef << ", press Enter to leave unchanged."
				<< endl;
	getline(cin, resp);
	if (resp != "")
		pStepRef = stod(resp);
	cout << "CUDA block count?" << endl;
	if (blockCountRef != -1)
		cout << "Current value: " << blockCountRef << ", press Enter to leave unchanged."
				<< endl;
	getline(cin, resp);
	if (resp != "")
		blockCountRef = stod(resp);
	cout << "Minimum iteration delta?" << endl;
	if (minDiffRef != -1)
		cout << "Current value: " << minDiffRef << ", press Enter to leave unchanged."
				<< endl;
	getline(cin, resp);
	if (resp != "")
		minDiffRef = stod(resp);
	cout << "Matrix file path? (-r to randomize, -b to build)" << endl;
	getline(cin, resp);
	if (resp == "-r" || resp == "-R") {
		int msize;
		cout << "Matrix size?" << endl;
		cin >> msize;
		matrixRef = Matrix(msize);
	} else if (resp == "-b" || resp == "-B") {
		cout << "Matrix builder file path?" << endl;
		getline(cin, resp);
		matrixRef.buildMat(ifstream(resp));
	} else
		matrixRef = Matrix(ifstream(resp));
	cout << "Do you want to append new temperature ranges to config after session? (yes/no)" << endl;
	cin >> resp;
	if (resp == "yes" || resp == "y") {
			appendConfigRef = true;
		} else
			appendConfigRef = false;
	return 0;
}

int StartupUtils::grabFromString(string inp, long double& startRef,
		long double& endRef, long& pointCountRef, double& pStepRef,
		Matrix& matrixRef, int& blockCountRef, string& wDirRef, bool& cliRef,
		float& minDiffRef, bool& appendConfigRef) {
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
		} else if (buffer == "%md" || buffer == "%mdiff" || buffer == "%mindiff") {
			ifs >> minDiffRef;
		} else if (buffer == "%c" || buffer == "%count") {
			ifs >> pointCountRef;
			countDefinedAsStep = false;
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
				cout << "InputParser: WARNING: "
						<< "No matrix file found at specified location, ignored."
						<< endl;
			} else if (buffer == "%ml_b" || buffer == "%mloc_b") {
				matrixRef.buildMat(ifstream(loc));
			} else {
				matrixRef = Matrix(ifstream(loc));
			}
		} else if (buffer == "%ird" || buffer == "%irand" || buffer == "%initrd"
				|| buffer == "%initrand") {
			string buf;
			ifs >> buf;
			if (buf == "true" || buf == "t")
				srand(time(0));
			else if (buf == "false" || buf == "f")
				srand(0);
			else {
				cout << "InputParser: ERROR: "
						<< "Bad word after %irand: " << buf << endl;
				return -1;
			}
		} else if (buffer == "%cli") {
			ifs >> buffer;
			if (buffer == "true" || buffer == "t")
				cliRef = true;
			else if (buffer == "false" || buffer == "f")
				cliRef = false;
			else {
				cout << "InputParser: ERROR: "
						<< "Bad word after %cli: " << buffer << endl;
				return -1;
			}
		}	else if (buffer == "%ac" || buffer == "%appconf") {
					ifs >> buffer;
					if (buffer == "true" || buffer == "t")
						appendConfigRef = true;
					else if (buffer == "false" || buffer == "f")
						appendConfigRef = false;
					else {
						cout << "InputParser: ERROR: "
								<< "Bad word after %appconf: " << buffer << endl;
						return -1;
					}
				}
		else if (buffer[0] == '#') {
			getline(ifs, buffer);
		} else {
			cout << "InputParser: WARNING: " << "Unknown word: \""
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
