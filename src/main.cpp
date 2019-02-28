#define VERSION 4.4
#define BUILD 77

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <ctime>
#include <mutex>
#include "Matrix.h"
#include "Spinset.h"
#include "FilesystemProvider.h"
#include "Plotter.h"
#include "StartupUtils.h"
#include "CudaOperator.h"

using namespace std;
using namespace FilesystemProvider;

//Minimum storage
double minEnergy;
int minCount;
string minSpins;
mutex mcMutex;
mutex mesMutex;

double progress;
time_t start;

string composeProgressbar(double state, int pbLen) {
	ostringstream os;
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

string getTimeString(double time) {
	if (time <= 0)
		return "0 h 0 m 0 s";
	ostringstream oss;
	int d = (int) (time / (24 * 3600));
	int h = (int) ((time - d * 24 * 3600) / 3600);
	int m = (int) ((time - d * 24 * 3600 - h * 3600) / 60);
	int s = (int) (time - d * 24 * 3600 - h * 3600 - m * 60);
	if (d != 0) {
		oss << d << " d ";
	}
	oss << h << " h " << m << " m " << s << " s";
	return oss.str();

}

void CLIControl() {
	while (progress != -1) {
		int termsize = 75;
		try {
			system("clear");
			system("tput cols > /tmp/lololol.lol");
			ostringstream iss;
			iss << ifstream("/tmp/lololol.lol").rdbuf();
			termsize = stoi(iss.str());
		} catch (exception & e) {
			termsize = 75;
		}
		cout << "Progress: " << composeProgressbar(progress, termsize - 20)
				<< endl;
		cout << "Time elapsed: " << getTimeString(time(NULL) - start) << endl;
		cout << "ETA: "
				<< getTimeString(
						(time(NULL) - start) * (1 - progress) / progress)
				<< endl;
		this_thread::sleep_for(std::chrono::seconds(1));
	}
	cout << "Computation complete in " << getTimeString(time(NULL) - start)
			<< endl;
}

int main(int argc, char* argv[]) {
	cout << "Calc program by Yxbcvn410, version " << VERSION << ", build "
			<< BUILD << endl;

	//Init model
	Matrix matrix(2);
	string dir;
	double dTemp = 0;
	double upTemp = -1;
	double step = 0.001;
	double pullStep = 0.1;
	int blockCount = -1;
	int nSave;
	bool doRand = true;
	ofstream logWriter;
	if (argc >= 2) {
		//Acquire init config from config
		try {
			matrix = Matrix(stoi(argv[1]));
		} catch (exception & e) {
		}
		if (argc >= 3) {
			upTemp = stod(argv[2]);
		}
		string wd = FilesystemProvider::getCurrentWorkingDirectory();
		nSave = StartupUtils::grabFromFile(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(matrix), ref(blockCount), ref(doRand),
				ref(dir), wd + "/config");
	} else {
		//Acquire init config from cin
		nSave = StartupUtils::grabFromCLI(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(matrix), ref(blockCount), ref(doRand),
				ref(dir));
	}

	logWriter.open(dir + "/log.txt", ios::out | ios::app);

	if (nSave == -1) { // If an error occured while parsing
		cout << "Errors in launch config detected. Fallback to CLI input."
				<< endl;
		logWriter << "Errors in launch config detected. Fallback to CLI input."
				<< endl;
		nSave = StartupUtils::grabFromCLI(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(matrix), ref(blockCount), ref(doRand),
				ref(dir));
	}

	logWriter << "Starting with ";
	if (doRand)
		srand(time(NULL));

	if (nSave == 1) { //Export matrix if needed
		matrix.Randomize();
		fstream fs;
		fs.open(ComposeFilename(dir, "mat", ".txt"), ios::out);
		fs << matrix.getMatrix();
		fs.flush();
		logWriter << "new matrix, size " << matrix.getSize() << endl;
	} else
		logWriter << "existing matrix, size " << matrix.getSize() << endl;

	// Init plot, clock, CUDA, CLI
	FilesystemProvider::makeFile(ComposeFilename(dir, "img", ".png"));
	Plotter::InitScriptfile(dir + "/plot.txt",
			ComposeFilename(dir, "img", ".png"), "Hamiltonian");
	ofstream hamiltonianWriter(
			ComposeFilename(dir, "data_hamiltonian", ".txt"));
	hamiltonianWriter << "t e" << endl;
	Plotter::AddDatafile(dir + "/plot.txt",
			ComposeFilename(dir, "data_hamiltonian", ".txt"), Plotter::POINTS);

	ofstream maxcutWriter(ComposeFilename(dir, "data_maxcut", ".txt"));
	maxcutWriter << " " << endl;

	CudaOperator op = CudaOperator(matrix, blockCount);
	start = time(NULL);
	thread th(CLIControl);
	th.detach();

	// Start calculations
	Spinset spins(matrix.getSize());
	spins.temp = dTemp;
	while (spins.temp < upTemp) {
		logWriter << "[" << getTimeString(time(NULL) - start) << "] "
				<< "Starting pull session. Loading spinsets..." << endl;
		for (int i = 0; i < blockCount; i++) {
			spins.Randomize(false);
			op.cudaLoadSpinset(spins, i);
			spins.temp += step;
		}
		logWriter << "[" << getTimeString(time(NULL) - start) << "] "
				<< "Spinset loading complete. Starting CUDA kernel function..."
				<< endl;
		op.cudaPull(pullStep);
		logWriter << "[" << getTimeString(time(NULL) - start) << "] "
				<< "Kernel returned success. Acquiring data..." << endl;
		spins.temp -= step * blockCount;
		for (int i = 0; i < blockCount; i++) {
			if (spins.temp >= upTemp)
				continue;
			double nrg = op.extractEnergy(i);
			if (nrg < minEnergy) {
				minEnergy = nrg;
				minCount = 1;
				minSpins = op.extractSpinset(i).getSpins();
			} else if (nrg == minEnergy)
				minCount++;
			hamiltonianWriter << abs(spins.temp) << "\t" << nrg << "\n";
			maxcutWriter << (matrix.getSum() - nrg) / 2.0 << ", \n";
			spins.temp += step;
		}
		progress = (spins.temp * spins.temp - dTemp * dTemp)
				/ (upTemp * upTemp - dTemp * dTemp);
		logWriter << "[" << getTimeString(time(NULL) - start) << "] "
				<< "Pull session complete, current temperature: " << spins.temp
				<< endl << endl;
		hamiltonianWriter.flush();
		maxcutWriter.flush();
	}

	// Disable output && write data to log
	progress = -1;
	logWriter << "Calculation complete in "
			<< getTimeString(difftime(time(NULL), start)) << endl;
	ofstream spinWriter;
	spinWriter.open(dir + "/spins.txt", ios::out);
	spinWriter << "Matrix size: " << matrix.getSize() << endl;
	spinWriter << "Minimum energy: " << minEnergy << endl;
	spinWriter << "Maximum cut: " << (matrix.getSum() - minEnergy) / 2.0
			<< endl;
	spinWriter << "Hit count: " << minCount << endl;
	spinWriter << "Temperature bounds: from " << dTemp << " to " << upTemp
			<< ", " << (int) ((upTemp - dTemp) / step) << " points in total"
			<< endl;
	spinWriter << "Spin assessment:" << endl << minSpins << endl;
	spinWriter << "Computed using " << blockCount << " thread blocks in "
			<< getTimeString(difftime(time(NULL), start)) << endl;
	spinWriter.flush();
	spinWriter.close();
	Plotter::doPlot(dir + "/plot.txt");
}
