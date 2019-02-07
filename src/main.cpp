#define VERSION 2.8
#define BUILD 32

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
#include "ModelUtils.h"
#include "FilesystemProvider.h"
#include "Plotter.h"
#include "StartupUtils.h"

using namespace std;
using namespace FilesystemProvider;

//Minimum storage
double minEnergy;
int minCount;
string minSpins;
mutex mcMutex;
mutex mesMutex;

void analyzeTempInterval(const Matrix &matrix, double start, double end,
		double step, double pullStep, const string &dir, const string &fname,
		int seed, double &progress) {
	Spinset spinset(matrix.getSize());
	spinset.seed(seed);
	ofstream ofs;
	progress = 0;

	int findex = FreeFileIndex(dir, fname, ".txt");
	ofs.open(ComposeFilename(dir, fname, findex, ".txt"), ios::out);
	ofs << "t e" << endl;
	ofs.flush();
	Plotter::AddDatafile(ComposeFilename(dir, fname, findex, ".txt"),
			Plotter::POINTS);

	double currentTemp = start;
	while (currentTemp < end) {
		spinset.temp = currentTemp;
		progress = (currentTemp - start) * (currentTemp + start)
				/ ((end - start) * (end + start));
		spinset.Randomize(false);
		ModelUtils::PullToZeroTemp(matrix, spinset, pullStep);
		ofs << currentTemp << " \t" << spinset.getEnergy(matrix) << endl;
		if (spinset.getEnergy(matrix) < minEnergy) {
			mesMutex.lock();
			if (spinset.getEnergy(matrix) < minEnergy) {
				minEnergy = spinset.getEnergy(matrix);
				minSpins = spinset.getSpins();
				minCount = 0;
			}
			mesMutex.unlock();
		} else if (spinset.getEnergy(matrix) == minEnergy) {
			mcMutex.lock();
			if (spinset.getEnergy(matrix) == minEnergy)
				minCount++; //*/
			mcMutex.unlock();
		}
		currentTemp += step;
	}
	ofs.flush();
	ofs.close();
	progress = -1;
}

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
	if (time < 0)
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

int main(int argc, char* argv[]) {
	cout << "Calc program by Yxbcvn410, version " << VERSION << ", build "
			<< BUILD << endl;

	//Init model
	Matrix m(2);
	string dir;
	double dTemp;
	double upTemp;
	double step = 0.01;
	double pullStep = 0.01;
	int thrC;
	int nSave;
	bool doRand = true;
	ofstream logWriter;
	if (argc == 2) {
		//Acquire init config from config
		try{
		m = Matrix(stod(argv[1]));
		}
		catch(exception* e){
			cout << "Argument not double";
		}
		logWriter << "Parsing init config..." << endl;
		string wd = StartupUtils::getCurrentWorkingDir();
		nSave = StartupUtils::grabFromFile(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(m), ref(thrC), ref(doRand), ref(dir),
				wd + "/config");

	} else {
		//Acquire init config from cin
		nSave = StartupUtils::grabFromCLI(ref(dTemp), ref(upTemp), ref(step),
				ref(pullStep), ref(m), ref(thrC), ref(doRand), ref(dir));
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
		m.Randomize();
		fstream fs;
		fs.open(
				ComposeFilename(dir, "mat", FreeFileIndex(dir, "mat", ".txt"),
						".txt"), ios::out);
		fs << m.getMatrix();
		fs.flush();
		logWriter << "new matrix, size " << m.getSize() << endl;
	} else
		logWriter << "existing matrix, size " << m.getSize() << endl;

	if (!doRand) { // Init random if needed
		srand(1);
		logWriter << "Random de-initialized." << endl;
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
				(step * thrC), pullStep, dir, "log", rand(), ref(statuses[j]));
		ht.detach();
		logWriter << "Launched thread #" << i + 1 << endl;
	}

	//Launch clock
	time_t start = time(NULL);

	//Beautiful output
	bool flag = true;
	int count = 1;
	double progr;
	while (flag) {
		this_thread::sleep_for(1000ms);
		system("clear");
		flag = false;
		progr = 0;
		for (int i = 0; i < thrC; ++i) {
			if (statuses[i] == -1)
				progr += 1;
			else
				progr += statuses[i];
			if (statuses[i] != -1)
				flag = true;
			cout << "Thread #" << i << ":\t"
					<< composeProgressbar(statuses[i], 60) << endl;
		}
		progr = progr / thrC;
		cout << "Time elapsed: " << getTimeString(difftime(time(NULL), start))
				<< "\n";
		cout << "ETA: "
				<< getTimeString(
						((1 - progr) / progr) * difftime(time(NULL), start))
				<< endl;
		count++;
		if (count > 60) {
			count = 1;
			logWriter << "[" << getTimeString(difftime(time(NULL), start))
					<< "]:\t Progress " << composeProgressbar(progr, 60)
					<< "\n";
			logWriter << "ETA: "
					<< getTimeString(
							((1 - progr) / progr) * difftime(time(NULL), start))
					<< endl;
		}
	}

	logWriter << "Calculation complete in "
			<< getTimeString(difftime(time(NULL), start)) << endl;
	ofstream spinWriter;
	spinWriter.open(dir + "/spins.txt", ios::out);
	spinWriter << "Minimum energy: " << minEnergy << endl;
	spinWriter << "Hit count: " << minCount << endl;
	spinWriter << "Spin assessment:" << endl << minSpins << endl;
	spinWriter.flush();
	spinWriter.close();
	Plotter::doPlot();
	Plotter::clearScriptfile();
}
