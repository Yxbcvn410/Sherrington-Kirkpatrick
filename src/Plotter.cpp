/*
 * Plotter.cpp
 *
 *  Created on: Jan 22, 2019
 *      Author: alexander
 */

#include "Plotter.h"
#include <fstream>
#include <mutex>
using namespace std;
using namespace Plotter;

void Plotter::InitScriptfile(string scriptFile, string outputFile, string title) {
	ofstream ofs;
	ofs.open(scriptFile, ios::out);
	ofs << "set autoscale" << endl;
	ofs << "set terminal png size 3200,2400" << endl;
	ofs << "set output '" << outputFile << "'" << endl;
	ofs << "set title \"" << title << "\"" << endl;
	ofs << "set grid x" << endl;
	ofs << "set grid y" << endl;
	ofs << "plot ";
	ofs.flush();
	ofs.close();
}

int countColumns(string file) {
	ifstream ifs;
	ifs.open(file);
	int out = 0;
	char b;
	bool f = false;
	while (b != '\n') {
		b=ifs.get();
		if (b != '\t' && b != ' ' && b != '\n')
			f = true;
		else {
			if (f)
				out++;
			f = false;
		}
	}
	return out;
}

mutex aplot_lock;

void Plotter::AddDatafile(string scriptFile, string dataFile, Plotter::PlotType type, string color) {
	aplot_lock.lock();
	int c = countColumns(dataFile);
	ofstream ofs;
	ofs.open(scriptFile, ios::out | ios::app);
	for (int i = 2; i <= c; ++i) {
		ofs << "\"" << dataFile << "\" using 1:" << i << " title '' with ";
 		switch (type) {
			case LINES:
				ofs << "lines";
				break;
			case POINTS:
				ofs << "points pointtype 13 ps 0.7 lt rgb \"#" << color << "\" ";
				break;
			default:
				break;
		}
 		ofs << ", \\\n";
	}
	ofs.flush();
	ofs.close();
	aplot_lock.unlock();
}

void Plotter::doPlot(string scriptFile){
	system(("gnuplot -c " + scriptFile).c_str());
}
