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

string FilesystemProvider::ComposeFilename(string ParentName, string key,
		string extention = "") {
	ostringstream out;
	out << ParentName << "/" << key << extention;
	return out.str();
}

mutex fs_lock;

int FilesystemProvider::FreeFileIndex(string ParentName, string key,
		string extention, bool reserve) {
	fs_lock.lock();
	int out = 0;
	while (FileExists(ComposeFilename(ParentName, key, out, extention)))
		out++;
	if (reserve) {
		makeFile(ComposeFilename(ParentName, key, out, extention));
	}
	fs_lock.unlock();
	return out;
}

void FilesystemProvider::makeDirectory(string pathTo, string name) {
	ostringstream oss;
	oss << "mkdir " << pathTo << "/" << name;
	system(oss.str().c_str());
}

void FilesystemProvider::makeFile(string filename) {
	ofstream ifs;
	ifs.open(filename, ios::out);
	ifs << " ";
	ifs.flush();
	ifs.close();
}
