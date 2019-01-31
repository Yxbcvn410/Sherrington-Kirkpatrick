#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include "Matrix.h"
#include "Spinset.h"
#include "ModelUtils.h"
#include "FilesystemProvider.h"
#include "Plotter.h"
#include "StartupUtils.h"

using namespace std;
using namespace FilesystemProvider;

void analyzeTempInterval(const Matrix &matrix, double start, double end,
		double step, double pullStep, const string &dir, const string &fname,
		double &progress) {
	Spinset spinset(matrix.getSize());
	ofstream ofs;
	progress = 0;

	int findex = FreeFileIndex(dir, fname, ".txt");
	ofs.open(ComposeFilename(dir, fname, findex, ".txt"), ios::out);
	ofs << "t e" << endl;
	ofs.flush();
	Plotter::AddDatafile(ComposeFilename(dir, fname, findex, ".txt"),
			Plotter::POINTS);

	double t = start;
	while (t < end) {
		spinset.temp = t;
		progress = (t - start) * (t + start) / ((end - start) * (end + start));
		spinset.Randomize(false);
		ModelUtils::PullToZeroTemp(matrix, spinset, pullStep);
		ofs << t << " \t" << spinset.getEnergy(matrix) << endl;
		t += step;
	}
	ofs.flush();
	ofs.close();
	progress = -1;
}

string composeThreadStatus(int id, double state, int pbLen) {
	ostringstream os;
	os << "Thread #" << id << ": \t";
	if (state == -1)
		os << "Dead.";
	else {
		os << "[";
		for (int i = 0; i < pbLen; i++)
			if (i / (double) pbLen < state)
				os << "#";
			else
				os << "-";
		os << "] ";
		os << (int) (state * 100);
		os << "%";
	}
	return os.str();
}

int main(int argc, char* argv[]) {
	cout << "Calc program by Yxbcvn410, version 2.5, build 15" << endl;

	//Init model
	Matrix m(2);
	string dir;
	double dTemp;
	double upTemp;
	double step;
	double pullStep;
	int thrC;
	bool nSave;
	if (argc == 2) {
		//Acquire init config from config
		string wd = StartupUtils::getCurrentWorkingDir();
		nSave = StartupUtils::grabFromFile(ref(dTemp), ref(upTemp), ref(step), ref(pullStep),
				ref(m), ref(thrC), ref(dir), wd + "/config");

	} else {
		//Acquire init config from cin
		nSave = StartupUtils::grabFromCLI(ref(dTemp), ref(upTemp), ref(step), ref(pullStep),
				ref(m), ref(thrC), ref(dir));
	}

	if (nSave) { //Export matrix if needed
		fstream fs;
		fs.open(
				ComposeFilename(dir, "mat", FreeFileIndex(dir, "mat", ".txt"),
						".txt"), ios::out);
		fs << m.getMatrix();
		fs.flush();
	}

	//Init plot
	Plotter::InitScriptfile(
			ComposeFilename(dir, "img", FreeFileIndex(dir, "img", ".png"),
					".png"), "");

	//Launch threads
	double* statuses = new double[thrC];
	for (int i = 0; i < thrC; ++i) {
		const int j = i;
		statuses[j] = 0;
		thread ht(analyzeTempInterval, m, dTemp + step * j, upTemp,
				(step * thrC), pullStep, dir, "log", ref(statuses[j]));
		ht.detach();
	}

	//Beautiful output
	bool flag = true;
	while (flag) {
		this_thread::sleep_for(1000ms);
		system("clear");
		flag = false;
		for (int i = 0; i < thrC; ++i) {
			if (statuses[i] != -1)
				flag = true;
			cout << composeThreadStatus(i, statuses[i], 60) << endl;
		}
	}
	Plotter::doPlot();
	Plotter::clearScriptfile();
}
