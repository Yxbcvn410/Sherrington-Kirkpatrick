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
		long double& stepRef, double& pStepRef, Matrix& modelRef,
		int& blockCountRef, bool& randRef, string& wDirRef);
int grabInteractive(long double& startRef, long double& endRef,
		long double& stepRef, double& pStepRef, Matrix& modelRef,
		int& blockCountRef, bool& randRef, string& wDirRef);
}

#endif /* STARTUPUTILS_H_ */
