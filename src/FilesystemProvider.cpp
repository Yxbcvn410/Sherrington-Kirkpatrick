/*
 * FilesystemProvider.cpp
 *
 *  Created on: Jan 20, 2019
 *      Author: alexander
 */

#include "FilesystemProvider.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <mutex>
using namespace std;

bool FilesystemProvider::FileExists(string Filename) {
	ifstream str(Filename.c_str());
	return str.good();
}

string FilesystemProvider::ComposeFilename(string ParentName, string key,
		int index, string extention) {
	ostringstream out;
	out << ParentName << "/" << key << index << extention;
	return out.str();
}

mutex fs_lock;

int FilesystemProvider::FreeFileIndex(string ParentName, string key,
		string extention) {
	fs_lock.lock();
	int out = 0;
	while (FileExists(ComposeFilename(ParentName, key, out, extention)))
		out++;
	ofstream ifs;
	ifs.open(ComposeFilename(ParentName, key, out, extention), ios::out);
	ifs << " ";
	ifs.flush();
	ifs.close();
	fs_lock.unlock();
	return out;
}
