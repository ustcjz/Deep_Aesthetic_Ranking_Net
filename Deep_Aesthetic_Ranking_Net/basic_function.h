#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<fstream>
#include<direct.h>
#include<io.h>

#include<opencv2/ml/ml.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/contrib/contrib.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cv.h>

using namespace std;
using namespace cv;

class JpgFileInFolder{
private:
	int jpgNum ;
	vector<string>fileNames ;
	
public:
	
	JpgFileInFolder( string folderPath)
	{
		Directory dir;
		fileNames = dir.GetListFiles( folderPath, "*.jpg", false);
		jpgNum = fileNames.size();
	}
	
	vector<string> getJPGFile()
	{
		return fileNames;
	}

	int getJPGNum()
	{
		return jpgNum;
	}
};

class FormatTransform{
public:
	static float string2float( string src);
	static vector<float> vectorString2Float( vector<string>);

	static int string2int( string src);
	static string int2str( int a);
	static vector<int> vectorString2Int( vector<string>);
	static string floatToString( float src);

};

class CrossValiParams{

public:

	/**
	cross validation to find best params of SVR model
	metrics is the kendall_tau coefficients
	*/
	static vector<float> svrParamsRBFforRank( Mat trainMat, Mat trainScore, int cMin, int cMax, int gMin,
								  int gMax, int pMin, int pMax, int cStep, int gStep, int pStep, int folderNum, vector<float>oriRank) ;

	static vector<float> svrParamsLinearforRank( Mat trainMat, Mat trainScrore, int cMin, int cMax, int pMin,
									 int pMax, int cStep, int pStep, int folderNum, vector<float>oriRank);

	/**
	return params order: c 
	*/
	static float svmParams_Linear( Mat trainMat, Mat trainLabel, int cMin, int cMax, int cStep,int folderNum);
	
	/**
	return params order : c p g
	*/
	static vector<float> svmParams_RBF( Mat trainMat, Mat trainLabel, int cMin, int cMax, int pMin,
								  int pMax, int gMin, int gMax, int cStep, int pStep, int gStep, int folderNum);

	/**
	split mat into train and test, used for cross validation
	*/
	static vector<Mat>MatSplit(Mat srcMat, int num, int startPos);

	/**
	split vector into train and test, used for cross validation	
	*/
	static vector<float>vecSplit(vector<float> src, int num, int startPos);
};

class ReadDataFromFile{
public:
	
	// read all lines in file, the return format is string, specified the rows
	static vector<string>readAllLines( string fileFullPath, int rows); 
	static vector<string>readAllLines( string fileFullPath, int rows, int cols, int colWanted);


	// read feat feature from binary file
	static vector<float> readBinaryFile( string fileName, int numFeat);

	// read all lines in file with wanted format: float
	static vector<float>readFloatLines( string fileFullPath, int rows);
	static vector<float>readFloatLines( string fileFullPath, int rows, int cols, int col);

	static Mat readFloatLines2Mat( string fileFullPath, int rows);
	static Mat readFloatLines2Mat( string fileFullPath, int rows, int cols, int col);

	static Mat readMatFromXML( string xmlFile, string saveName);
};

class KendallTau{
public:
	// rank of a vector, bigger rank value for a bigger original value
	static vector<int>rankOfVector( vector<float> a);

	static float kendallCal( vector<float>preRank, vector<float>oriRank);
};

class SvmSvrModel{
private:
	CvSVMParams params;
	CvSVM svm;
	Mat trainMat, trainTag;
	Mat testMat, testTag;
	Mat predTag;

public:
	SvmSvrModel(Mat trainMatt, Mat trainTagg, CvSVMParams paramss)
	{
		trainMat	= trainMatt;
		trainTag	= trainTagg;
		params		= paramss;

		svm.train( trainMat, trainTag, Mat(), Mat(), params);
	}
	CvSVMParams getParams()
	{
		return params;
	}	
	bool setParams( CvSVMParams paramss)
	{
		params = paramss;
		svm.train( trainMat, trainTag, Mat(), Mat(), params);
		return true;
	}
	Mat getPredTag()
	{
		return predTag;
	}
	bool predict(Mat testMat, bool flag)
	{
		if(testMat.rows == 0)
			return false;

		//TODO predict
		predTag		 = Mat::zeros(testMat.rows, 1, CV_32FC1);
		Mat rowTmp = Mat::zeros(1, testMat.cols, CV_32FC1);
		
		for (int i = 0; i < testMat.rows; i++)
		{
			testMat.row( i ).copyTo(rowTmp);

			float *dataP	= predTag.ptr<float>( i );
			dataP[i]			= svm.predict( rowTmp, flag);
		}
		return true;
	}
	bool predict(Mat testMat, string file, bool flag)
	{
		if(testMat.rows == 0)
			return false;

		//TODO predict
		ofstream fs(file);

		predTag		 = Mat::zeros(testMat.rows, 1, CV_32FC1);
		Mat rowTmp = Mat::zeros(1, testMat.cols, CV_32FC1);
		
		for (int i = 0; i < testMat.rows; i++)
		{
			testMat.row( i ).copyTo(rowTmp);

			float *dataP	= predTag.ptr<float>( i );
			dataP[0]		= svm.predict( rowTmp, flag);
			
			fs<<dataP[0]<<endl;
		}
		fs.close();

		return true;
	}
};

class FileTransform{
public:
	static bool feat2xml();
	static bool xml2txt( string xmlFile, string saveName, string txtFile);
};

int ImageToDataset( ); // drag images from whole dataset to 900 dataset


// is image valid in trainset and test; return Mat or null Mat 
Mat isImgExist( string trainSetPath, string testSetPath, string imgName);


vector<int>vectorForm(int min, int max, int step);
vector<float>vectorForm( float min, float max, float step);

// the cols of input mat should be 1
float mseCal4Mat( Mat mat1, Mat mat2) ;


bool exeCommand(string cmd);

/*
make new dir
*/
bool makeDir (string dir);