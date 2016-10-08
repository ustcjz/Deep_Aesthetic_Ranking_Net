#include"Head.h"

void consAesthIndexFile();

void consOneAesthIndexFile(float scoreGap, vector<string>saveFileVec, vector<int>numVec, 
						   vector<float>&scoreVec, vector<string>&imgNamesVec, vector<string>&namesServerVec, map<string,int>& imgToIndex);

void consOneAesthIndexFile_random(float scoreGap, vector<string>saveFileVec, vector<int>numVec, vector<int> bili,  
						   vector<float>&scoreVec, vector<string>&imgNamesVec, vector<string>&namesServerVec, map<string,int>& imgToIndex);

void writeAesthLine(ofstream& fs, int gIndex, int bIndex, string base, vector<string>&imgNamesVec);