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
#include "FilesystemProvider.h"
using namespace std;

int StartupUtils::grabFromCLI(double& startRef, double& endRef, double& stepRef,
		double& pStepRef, Matrix& modelRef, int& thrCountRef, bool& randRef,
		string& wDirRef) {
	cout << "Do you want to init randomizer? (yes/no) ";
	string resp = "";
	cin >> resp;
	randRef = false;
	if (resp == "yes" || resp == "y") {
		randRef = true;
		cout << "Randomizer initialized." << endl;
	} else
		cout << "Randomizer was not initialized" << "\n";
	cout << "Working dir? (-c to auto-create working directory)\n"
			<< getCurrentWorkingDir() << endl;
	cin >> wDirRef;
	if (wDirRef == "-c" || wDirRef == "-C") {
		int dirIndex = 0;
		do {
			dirIndex++;
			ostringstream oss;
			oss << getCurrentWorkingDir() << "/calc" << dirIndex;
			wDirRef = oss.str();
		} while (FilesystemProvider::FileExists(wDirRef));
		ostringstream oss;
		oss << "mkdir " << wDirRef;
		system(oss.str().c_str());
	}
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
		return 1;
	} else
		modelRef = Matrix(ifstream(resp));
	return 0;
}

int StartupUtils::grabFromFile(double& startRef, double& endRef,
		double& stepRef, double& pStepRef, Matrix& modelRef, int& thrCountRef,
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
	int needSave = 1;
	ifs >> s;
	while (ifs.peek() != EOF) {
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
			needSave = 1;
		} else if (s == "&wdir" || s == "&wd" || s == "&dir") {
			ifs >> wDirRef;
			if (wDirRef == "-a" || wDirRef == "-A") {
				int dirIndex = 0;
				do {
					dirIndex++;
					ostringstream oss;
					oss << getCurrentWorkingDir() << "/calc" << dirIndex;
					wDirRef = oss.str();
				} while (FilesystemProvider::FileExists(wDirRef));
				ostringstream oss;
				oss << "mkdir " << wDirRef;
				system(oss.str().c_str());
			}
		} else if (s == "&mloc" || s == "&ml") {
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
	cout << "Config parsing complete, success." << endl;
	return needSave;
}

string StartupUtils::getCurrentWorkingDir() {
	char buff[FILENAME_MAX];
	getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}
