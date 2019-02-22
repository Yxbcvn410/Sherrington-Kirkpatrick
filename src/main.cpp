#define VERSION 4.2
#define BUILD 65

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
	else if (state == 0) {
		os << "Idle.";
	} else {
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
		system("clear");
		system("tput cols > /tmp/lololol.lol");
		int termsize = 75;
		try {
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
}

int main(int argc, char* argv[]) {
	cout << "Calc program by Yxbcvn410, version " << VERSION << ", build "
			<< BUILD << endl;

	//Init model
	Matrix matrix(2);
	string dir;
	double dTemp = 0;
	double upTemp;
	double step = 0.001;
	double pullStep = 0.1;
	int blockCount;
	int nSave;
	bool doRand = true;
	ofstream logWriter;
	if (argc >= 2) {
		//Acquire init config from config
		matrix = Matrix(stoi(argv[1]));
		if (argc >= 3) {
			upTemp = stod(argv[2]);
		}
		logWriter << "Parsing init config..." << endl;
		string wd = StartupUtils::getCurrentWorkingDir();
		nSave = StartupUtils::grabFromFile(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(matrix), ref(blockCount), ref(doRand),
				ref(dir), wd + "/config");

	} else {
		//Acquire init config from cin
		nSave = StartupUtils::grabFromCLI(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(matrix), ref(blockCount), ref(doRand),
				ref(dir));
	}
	logWriter.open(dir + "/l.log", ios::out | ios::app);

	if (nSave == -1) { // If an error occured while parsing
		cout << "Program terminated due to an error in launch config." << endl;
		logWriter << "Program terminated due to an error in launch config."
				<< endl;
		return -1;
	}

	logWriter << "Starting with ";

	srand(time(0));

	if (nSave == 1) { //Export matrix if needed
		matrix.Randomize();
		fstream fs;
		fs.open(
				ComposeFilename(dir, "mat",
						FreeFileIndex(dir, "mat", ".txt", true), ".txt"),
				ios::out);
		fs << matrix.getMatrix();
		fs.flush();
		logWriter << "new matrix, size " << matrix.getSize() << endl;
	} else
		logWriter << "existing matrix, size " << matrix.getSize() << endl;

	if (!doRand) { // Init random if needed
		srand(1);
		logWriter << "Random de-initialized." << endl;
	}
	// Init plot, clock, CUDA, CLI
	Plotter::InitScriptfile(
			ComposeFilename(dir, "img", FreeFileIndex(dir, "img", ".png", true),
					".png"), "");
	start = time(NULL);
	CudaOperator op(matrix, blockCount);
	int dataIndex = FreeFileIndex(dir, "data", ".txt", true);
	ofstream dataStream(ComposeFilename(dir, "data", dataIndex, ".txt"));
	dataStream << "t e" << endl;
	Plotter::AddDatafile(ComposeFilename(dir, "data", dataIndex, ".txt"),
			Plotter::POINTS);
	thread th(CLIControl);
	th.detach();

	// Start calculations
	Spinset spins(matrix.getSize());
	spins.temp = dTemp;
	while (spins.temp < upTemp) {
		for (int i = 0; i < blockCount; i++) {
			spins.Randomize(false);
			op.cudaLoadSpinset(spins, i);
			spins.temp += step;
		}
		op.cudaPull(pullStep);
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
			dataStream << spins.temp << " " << nrg << "\n";
			spins.temp += step;
		}
		progress = (spins.temp * spins.temp - dTemp * dTemp)
				/ (upTemp * upTemp - dTemp * dTemp);
		logWriter << "[" << getTimeString(time(NULL) - start) << "] ETA: "
							<< getTimeString(
									(time(NULL) - start) * (1 - progress) / progress)
							<< endl;
		dataStream.flush();
	}

	progress = -1;
	logWriter << "Calculation complete in "
			<< getTimeString(difftime(time(NULL), start)) << endl;
	ofstream spinWriter;
	spinWriter.open(dir + "/spins.txt", ios::out);
	spinWriter << "Minimum energy: " << minEnergy << endl;
	spinWriter << "Hit count: " << minCount << endl;
	spinWriter << "Temperature bounds: from " << dTemp << " to " << upTemp
			<< ", " << (int) ((upTemp - dTemp) / step) << " points in total"
			<< endl;
	spinWriter << "Spin assessment:" << endl << minSpins << endl;
	spinWriter << "Computed using " << blockCount << " threads in "
			<< getTimeString(difftime(time(NULL), start)) << endl;
	spinWriter.flush();
	spinWriter.close();
	Plotter::doPlot();
	Plotter::clearScriptfile();
}
