/*
 * FilesystemProvider.h
 *
 *  Created on: Jan 20, 2019
 *      Author: alexander
 */

#include <iostream>
#ifndef FILESYSTEMPROVIDER_H_
#define FILESYSTEMPROVIDER_H_
using namespace std;

namespace FilesystemProvider {
bool FileExists(string filename);
int FreeFileIndex(string parentName, string key, string extention);
string ComposeFilename(string parentName, string key, int index,
		string extention);
}

#endif /* FILESYSTEMPROVIDER_H_ */
