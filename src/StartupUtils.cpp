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
		int& blockCountRef, string& wDirRef, bool& cliRef, float& minDiffRef) {
	cliRef = true;
	cout << "Do you want to init randomizer? (yes/no) ";
	string resp = "";
	cin >> resp;
	srand(0);
	if (resp == "yes" || resp == "y") {
		srand(time(0));
	} else
		cout << "InputParser: WARNING: Randomizer was not initialized" << "\n";
	if (wDirRef == "") {
		cout
				<< "Working dir? (-a to auto-create in current working directory)\n"
				<< getCurrentWorkingDirectory() << endl;
		cin >> wDirRef;
		if (wDirRef == "-a" || wDirRef == "-A") {
			ostringstream oss;
			oss << "calc" << modelRef.getSize() << "_";
			int dirIndex = FreeFileIndex(getCurrentWorkingDirectory(),
					oss.str(), "", false);
			oss << dirIndex;
			wDirRef = getCurrentWorkingDirectory() + "/" + oss.str();
		}
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
	cout << "Minimum iteration delta?" << endl;
	if (minDiffRef != -1)
		cout << "Current value: " << minDiffRef << ", -c to leave unchanged."
				<< endl;
	cin >> resp;
	if (resp != "-c")
		minDiffRef = stod(resp);
	cout << "Matrix file path? (-r to randomize, -b to build)" << endl;
	cin >> resp;
	if (resp == "-r" || resp == "-R") {
		int msize;
		cout << "Matrix size?" << endl;
		cin >> msize;
		modelRef = Matrix(msize);
	} else if (resp == "-b" || resp == "-B") {
		cout << "Matrix builder file path?" << endl;
		cin >> resp;
		modelRef.buildMat(ifstream(resp));
	} else
		modelRef = Matrix(ifstream(resp));
	return 0;
}

int StartupUtils::grabFromString(string inp, long double& startRef,
		long double& endRef, long double& stepRef, double& pStepRef,
		Matrix& modelRef, int& blockCountRef, string& wDirRef, bool& cliRef,
		float& minDiffRef) {
	istringstream ifs = istringstream(inp);
	string buffer;
	bool stepDefinedAsCount = false;
	double pCount = 1;
	ifs >> buffer;
	while (ifs.peek() != EOF) {
		if (buffer == "%start") {
			ifs >> startRef;
		} else if (buffer == "%end" || buffer == "%e") {
			ifs >> endRef;
		} else if (buffer == "%step") {
			ifs >> stepRef;
			stepDefinedAsCount = false;
		} else if (buffer == "%md") {
			ifs >> minDiffRef;
		} else if (buffer == "%c" || buffer == "%count") {
			ifs >> pCount;
			stepDefinedAsCount = true;
		} else if (buffer == "%pstep" || buffer == "%ps") {
			ifs >> pStepRef;
		} else if (buffer == "%bc" || buffer == "%bcount") {
			ifs >> blockCountRef;
		} else if (buffer == "%msize" || buffer == "%ms") {
			int size;
			ifs >> size;
			modelRef = Matrix(size);
			modelRef.Randomize();
		} else if (buffer == "%wdir" || buffer == "%wd" || buffer == "%dir") {
			ifs >> wDirRef;
			if (wDirRef == "-a" || wDirRef == "-A") {
				ostringstream oss;
				oss << "calc" << modelRef.getSize() << "_";
				int dirIndex = FreeFileIndex(getCurrentWorkingDirectory(),
						oss.str(), "", false);
				oss << dirIndex;
				wDirRef = getCurrentWorkingDirectory() + "/" + oss.str();
			}
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
				modelRef.buildMat(ifstream(loc));
			} else {
				modelRef = Matrix(ifstream(loc));
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
			string buf;
			ifs >> buf;
			if (buf == "true" || buf == "t")
				cliRef = true;
			else if (buf == "false" || buf == "f")
				cliRef = false;
			else {
				cout << "InputParser: ERROR: "
						<< "Bad word after %cli: " << buf << endl;
				return -1;
			}
		}

		else if (buffer[0] == '#') {
			string buf;
			getline(ifs, buf);
		} else {
			cout << "InputParser: WARNING: " << "Unknown word: \""
					<< buffer << "\", ignored" << endl;
		}
		ifs >> buffer;
	}
	if (stepDefinedAsCount)
		stepRef = (endRef - startRef) / pCount;
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
	if (modelRef.getSize() == 2) {
		cout << "InputParser: WARNING: " << "Matrix not defined" << endl;
		exitCode = 1;
	}
	return exitCode;
}
