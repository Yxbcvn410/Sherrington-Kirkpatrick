/*
 * GlobalUtils.h
 *
 *  Created on: Jan 29, 2019
 *      Author: alexander
 */

#ifndef STARTUPUTILS_H_
#define STARTUPUTILS_H_
#include "Matrix.h"

namespace StartupUtils {
int grabFromFile(double& startRef, double& endRef, double& stepRef,
		double& pStepRef, Matrix& modelRef, int& blockCountRef, bool& randRef,
		string& wDirRef, string confLocation);
int grabFromCLI(double& startRef, double& endRef, double& stepRef,
		double& pStepRef, Matrix& modelRef, int& blockCountRef, bool& randRef,
		string& wDirRef);
string getCurrentWorkingDir();
}

#endif /* STARTUPUTILS_H_ */
