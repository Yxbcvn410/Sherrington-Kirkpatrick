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
bool grabFromFile(double& startRef, double& endRef, double& stepRef, Matrix& modelRef,
		int& thrCountRef, string& wDirRef, string confLocation);
bool grabFromCLI(double& startRef, double& endRef, double& stepRef, Matrix& modelRef,
		int& thrCountRef, string& wDirRef);
string getCurrentWorkingDir();
}

#endif /* STARTUPUTILS_H_ */
