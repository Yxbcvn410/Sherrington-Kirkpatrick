/*
 * Plotter.h
 *
 *  Created on: Jan 22, 2019
 *      Author: alexander
 */

#include <iostream>
#include<stdio.h>
#ifndef PLOTTER_H_
#define PLOTTER_H_
using namespace std;

namespace Plotter {
enum PlotType {
	LINES, POINTS
};
void InitScriptfile(string scriptFile, string outputFile, string title = "");
void AddDatafile(string scriptFile, string dataFile, PlotType type, string color = "0000FF");
void doPlot(string scriptFile);
}

#endif /* PLOTTER_H_ */
