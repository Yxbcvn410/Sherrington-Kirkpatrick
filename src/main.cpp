#define VERSION 4.6
#define BUILD "87.3.2"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <ctime>
#include <cmath>
#include "Matrix.h"
#include "Spinset.h"
#include "FilesystemProvider.h"
#include "Plotter.h"
#include "StartupUtils.h"
#include "CudaOperator.h"

using namespace std;
using namespace FilesystemProvider;

//Minimum storage
double minHamiltonian;
int minCount;
string minSpins;
float minTemp;
long double hamSum;

float progress;
time_t start;

string composeProgressbar(float state, int pbLen) {
	ostringstream os;
	if (state == -1)
		os << "Dead.";
	else {
		os << "[";
		for (int i = 0; i < pbLen; i++)
			if (i / (float) pbLen < state)
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

void saveMinData(string dir, Matrix matrix, long double dTemp,
		long double upTemp, long double step, int blockCount) {
	ofstream spinWriter;
	if (FilesystemProvider::FileExists(dir + "/spins.txt"))
		system(("rm -r " + dir + "/spins.txt").c_str());
	spinWriter.open(dir + "/spins.txt", ios::out);
	spinWriter << "Matrix size: " << matrix.getSize() << endl;
	spinWriter << "Minimum hamiltonian: " << minHamiltonian << endl;
	spinWriter << "Maximum cut: " << (matrix.getSum() - minHamiltonian) / 2.0
			<< endl;
	spinWriter << "Hit count: " << minCount << endl;
	spinWriter << "Hit temperature: " << minTemp << endl;
	spinWriter << "Middle hamiltonian: " << hamSum * step / (upTemp - dTemp)
			<< endl;
	spinWriter << "Middle max-cut: "
			<< (matrix.getSum() - (hamSum) * step / (upTemp - dTemp)) * 0.5
			<< endl;
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

int main(int argc, char* argv[]) {
	cout << "NMFA-like analysis by Yxbcvn410, version " << VERSION << ", build "
			<< BUILD << endl;

	//Init model
	Matrix matrix(2);
	string dir = "";
	long double dTemp = 0;
	long double upTemp = -1;
	long double step = 0.001;
	double pullStep = 1;
	int blockCount = -1;
	int nSave = -2;
	bool doRand = true;
	ofstream logWriter;
	bool displayData = false;

	//Append config file
	cout << "Trying to parse init config... \n"
			"Make sure it is located in current working directory and is called \"config\"!" << endl;
	ostringstream oss;
	try {
		oss
				<< ifstream(
						FilesystemProvider::getCurrentWorkingDirectory()
								+ "/config").rdbuf();
		cout << "Done." << endl;
	} catch (exception & e) {
		cout << "Failed. \nCheck config file existence." << endl;
	}

	//Append arguments
	if (argc >= 2) {
		cout << "Trying to parse arguments... " << endl;
		for (int i = 1; i < argc; i++) {
			oss << argv[i] << " ";
		}
		cout << "Done." << endl;
	}

	cout << "Parsing..." << endl;
	nSave = StartupUtils::grabFromString(oss.str(), ref(dTemp), ref(upTemp),
			ref(step), ref(pullStep), ref(matrix), ref(blockCount),
			ref(doRand), ref(dir));
	cout << "Complete." << endl;

	if (dir == "" || upTemp == -1 || blockCount == -1
			|| matrix.getSize() == 2) {
		cout
				<< "Not all init parameters were assigned. Fallback to interactive mode."
				<< endl;
		nSave = StartupUtils::grabInteractive(ref(dTemp), ref(upTemp),
				ref(step), ref(pullStep), ref(matrix), ref(blockCount),
				ref(doRand), ref(dir));
		displayData = true;
	}

	if (!doRand)
		cout << "WARNING: Random generator was not initialized." << endl;
	else
		srand(time(0));

	logWriter.open(dir + "/log.txt", ios::out | ios::app);

	logWriter << "Starting with ";

	if (nSave == 1) { //Export matrix if needed
		matrix.Randomize();
		logWriter << "new matrix, size " << matrix.getSize() << endl;
	} else
		logWriter << "existing matrix, size " << matrix.getSize() << endl;
	if (nSave == 1 || nSave == 2) {
		fstream fs;
		fs.open(ComposeFilename(dir, "mat", ".txt"), ios::out);
		fs << matrix.getMatrix();
		fs.flush();
	}

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
	if (displayData) {
		thread th(CLIControl);
		th.detach();
	}

// Start calculations
	Spinset spins(matrix.getSize());
	spins.temp = dTemp;
	do {
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
			double nrg = op.extractHamiltonian(i);
			hamSum += nrg;
			if (nrg < minHamiltonian) {
				minHamiltonian = nrg;
				minCount = 1;
				minSpins = op.extractSpinset(i).getSpins();
				minTemp = spins.temp;
			} else if (nrg == minHamiltonian)
				minCount++;
			hamiltonianWriter << fabs(spins.temp) << "\t" << nrg << "\n";
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
		saveMinData(dir, matrix, dTemp, spins.temp, step, blockCount);
	} while (spins.temp + step * 0.001 < upTemp);

// Disable output && write data to log
	progress = -1;
	logWriter << "Calculation complete in "
			<< getTimeString(difftime(time(NULL), start)) << endl;
	saveMinData(dir, matrix, dTemp, upTemp, step, blockCount);
}
