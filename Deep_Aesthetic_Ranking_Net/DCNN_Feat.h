#pragma once

#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include<time.h>

#include<opencv2/opencv.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/contrib/contrib.hpp>
#include<opencv2/core/core.hpp>

using namespace std;
using namespace cv;

struct ImgInfo{
	string imgName;
//	vector<float> dcnn_feat;
	float dis;
}
;


// read *.sift file, and normalization
bool readBinaryFile(string fileNames, int featNum, vector<float>& feat);

/*
 * form all imginfo
 */
bool readImgInfoVec( string qImg, string srcFold, vector<ImgInfo>& imgInfoVec);


bool readImgInfoVec( string qImg, string srcFold, vector<ImgInfo>& imgInfoVec, vector<vector<float>>& dcnnFeatVec);
/*
	test retrieval function
 */
void retrievalTest( );


/*
	retrieval and save topK img 
 */
void retrievalTopK(string qImg, int topK, string srcFold, string saveFold);

/*
	retrieval on training set and save table
 */
void retrievalAndSave(string qImg, string srcFold, string saveFile);

void retrievalAndSave(string qImg, string srcFold, string saveFile, vector<vector<float>>& dcnnFeatVec);

/*
	calculate the distance between 2 vectors
 */
float disCal(vector<float>& src, vector<float>& dst);

/*
	userdefined sort of class ImgInfo
 */
bool ImgInfoSort( ImgInfo a, ImgInfo b);


/*----------------------------------------------------------*/
/*
	transform all .feat file to a XML file
 */
void feat2XML();

/*
	read *.sift file, and normalization
*/
bool readBinaryFile(string fileNames, int featNum, Mat& feat);

/*
	read .aesthFeat file
*/
bool readAesthFeatFile(string fileName, int featNum, vector<float>& feat);

/*
	read .aesthLabel file
*/
bool readAesthLabelFile(string fileName, int& label);

/*
 * form all imginfo
 */
bool readImgInfoVecXML( string qImg, string srcFold, vector<ImgInfo>& imgInfoVec);


int getImgNum(string srcFold);


/*
	read .featIndex file
*/
bool readFeatIndexFile(string fileName, int& featIndex);


/*
	read .score file, which save the original score of each image
*/
bool readScoreFile(string fileName, float& score);

vector<vector<float>> readAllDcnnFeat(string srcFold);