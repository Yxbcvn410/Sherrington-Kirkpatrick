/*
 * GlobalUtils.h
 *
 *  Created on: Jan 29, 2019
 *      Author: alexander
 */

#ifndef STARTUPUTILS_H_
#define STARTUPUTILS_H_
#include "Matrice.h"

namespace StartupUtils {
int grabFromString(string inp, long double& startRef, long double& endRef,
		long& pointCountRef, double& pStepRef, Matrice& matriceRef,
		int& blockCountRef, string& wDirRef, bool& cliRef, float& minDiffRef, int& appendConfigRef, float& linearCoefRef, bool& doPlot);
}

#endif /* STARTUPUTILS_H_ */

/*
 * INFO:
 * Return 0 if success,
 * -1 if error
 * 1 if data incomplete
 */
