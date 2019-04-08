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
int grabFromString(string inp, long double& startRef, long double& endRef,
		long double& stepRef, double& pStepRef, Matrix& matrixRef,
		int& blockCountRef, string& wDirRef, bool& cliRef, float& minDiffRef);
int grabInteractive(long double& startRef, long double& endRef,
		long double& stepRef, double& pStepRef, Matrix& matrixRef,
		int& blockCountRef, string& wDirRef, bool& cliRef, float& minDiffRef);
}

#endif /* STARTUPUTILS_H_ */

/*
 * INFO:
 * Return 0 if success,
 * -1 if error
 * 1 if data incomplete
 */
