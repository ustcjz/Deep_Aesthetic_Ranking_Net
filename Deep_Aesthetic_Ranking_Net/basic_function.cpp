#include"basic_function.h"

using namespace std;

/*----------------------------class ReadDataFromFile----------------------------------------*/

// read all lines in file, the return format is string, specified the rows
vector<string> ReadDataFromFile::readAllLines( string fileFullPath, int rows)
{
	vector<string>lines(rows,"");
	ifstream fs( fileFullPath);

	string temp;
	for( int i = 0; i < rows; i++){
		fs>>temp;
		lines[ i ] = temp;
	}
	fs.close();
	return lines;
};

vector<string> ReadDataFromFile::readAllLines( string fileFullPath, int rows, int cols, int colWanted)
{
	vector<string>lines(rows, "");
	ifstream fs( fileFullPath);

	string temp;
	int count = 0;
	for( int i = 0; i < rows; i ++){
		for( int j = 0; j < cols; j++){
			fs>>temp;
			if( j == colWanted){
				lines[count] = temp;
				count++;
			}
		}
	}
	return lines;
}

// read feat feature from binary file
vector<float> ReadDataFromFile::readBinaryFile( string fileName, int numFeat)
{
	//---- feature of DCNN Linear
	vector<float>feat( numFeat , -1);
	ifstream fs( fileName, ios::binary);

	float a;
	float sum = 0;
	for( int i = 0; i < numFeat; i++){
		fs.read((char*)&a, sizeof(float));
		sum += a;
		feat[ i ] = a;
	}
	fs.close();
	//---- normalization of DCNN feature
	sum = sqrt(sum);
	for( int i = 0; i < numFeat; i++){
		feat[ i ] = sqrt( feat[ i ] )/sum;
	}

	return feat;
}

// read all lines in file with wanted format: float
vector<float> ReadDataFromFile::readFloatLines( string fileFullPath, int rows)
{
	vector<float>lines(rows, 0);
	ifstream fs( fileFullPath);

	string temp;
	for( int i = 0; i < rows; i++){
		fs>>temp;
		lines[ i ] = FormatTransform::string2float(temp);
	}
	fs.close();
	return lines;
}

vector<float> ReadDataFromFile::readFloatLines( string fileFullPath, int rows, int cols, int col)
{
	vector<string>stringLines = readAllLines( fileFullPath, rows, cols, col);
	vector<float>lines(rows, 0);

	for( int i = 0; i < stringLines.size(); i++){
		lines[ i ] = FormatTransform::string2float( stringLines[ i ] );
	}

	return lines;
}

Mat ReadDataFromFile::readFloatLines2Mat( string fileFullPath, int rows)
{
	vector<float>lines = readFloatLines( fileFullPath, rows);
	Mat linesMat = Mat::zeros( rows, 1, CV_32FC1);

	for( int i = 0; i < rows; i ++){
		float* dataL = linesMat.ptr<float>( i );
		dataL[ 0 ] = lines[ i ];
	}
	return linesMat;
}

Mat ReadDataFromFile::readFloatLines2Mat( string fileFullPath, int rows, int cols, int col)
{
	vector<float>lines = readFloatLines( fileFullPath, rows, cols, col);
	Mat linesMat = Mat::zeros( rows, 1, CV_32FC1);

	for( int i = 0; i < rows; i ++){
		float* dataL = linesMat.ptr<float>( i );
		dataL[ 0 ] = lines[ i ];
	}
	return linesMat;
}

Mat ReadDataFromFile::readMatFromXML( string xmlFile, string saveName)
{
	Mat data;
	FileStorage fs( xmlFile, FileStorage::READ);
	fs[saveName]>>data;
	fs.release();
	if( data.rows == 0){
		cout<<endl;
		cout<<"read Mat from file failed"<<endl;
		cout<<"xml file: "<<xmlFile<<endl;
		cout<<"xml save name"<<saveName<<endl<<endl;
	}
	return data;
}



/*---------------------------class FormatTransform--------------------------------------------------*/

float FormatTransform::string2float( string src)
{
	char stringTemp[ 256 ];
	for( int i = 0; i < src.size(); i++ )
	{
		stringTemp[ i ] = src[ i ];
	}
	stringTemp[ src.size() ] = '\0';

	float des = atof( stringTemp );
	return des;
}

vector<float> FormatTransform::vectorString2Float( vector<string>src)
{
	vector<float>vec( src.size(), 0);
	for( int i = 0; i < src.size(); i++){
		vec[ i ] = string2float( src[ i ]);
	}
	return vec;
}

int FormatTransform::string2int( string src)
{
	char stringTemp[ 256 ];
	for( int i = 0; i < src.size(); i++ )
	{
		stringTemp[ i ] = src[ i ];
	}
	stringTemp[ src.size() ] = '\0';

	int des = atoi( stringTemp );
	return des;
}

string FormatTransform::int2str(int a)
{
	char buffer[20];
	_itoa( a, buffer, 10 );
	string s(buffer);
	return s;
}

vector<int> FormatTransform::vectorString2Int( vector<string> src) 
{
	vector<int>vec( src.size(), 0);
	for( int i = 0; i < src.size(); i++){
		vec[ i ] = string2int( src[ i ]);
	}
	return vec;
}

string FormatTransform::floatToString( float src)
{
	char strf[256];
	_gcvt_s( strf, 256, src, 25);
	string str = strf;
	return str;
}



/*----------------------------class CrossValiParams------------------------------------*/

// cross-validation to find best params of SVR model
vector<float> CrossValiParams::svrParamsRBFforRank( Mat trainMat, Mat trainScore, int cMin, int cMax, int gMin,
												   int gMax, int pMin, int pMax, int cStep, int gStep, int pStep, int folderNum, vector<float>oriRank) 
{
	vector<int>cVector = vectorForm( cMin, cMax, cStep);
	vector<int>gVector = vectorForm( gMin, gMax, gStep);
	vector<int>pVector = vectorForm( pMin, pMax, pStep);

	vector<float>bestParams = vector<float>(3, -1);

	int cTemp = -1;
	int gTemp = -1;
	int pTemp = -1;

	float maxKendall = -1;

	for( int i = 0; i < cVector.size(); i++)
	{
		cTemp = cVector[ i ];
		for( int j = 0; j < gVector.size(); j++)
		{
			gTemp = gVector[ j ];
			for( int k = 0; k < pVector.size(); k++)
			{
				pTemp = pVector[ k ];

				cout<<"C: "<<cTemp<<", g: "<<gTemp<<", p: "<<pTemp<<"\t";
				//------------ set the parameters of SVM
				CvSVMParams params;
				params.svm_type = CvSVM::EPS_SVR;
				params.kernel_type = CvSVM::RBF;
				params.C = pow( 2.0, cTemp);
				params.gamma = pow( 2.0, gTemp); 
				params.p = pow( 2.0, pTemp);
				params.term_crit = TermCriteria( CV_TERMCRIT_ITER, (int)1e4, 1e-6);


				float curKendall = 0;
				for( int m = 0; m < folderNum; m++)
				{
					Mat curTrainMat, curTestMat;
					Mat curTrainScore, curTestScore;
					vector<float>curTestRank;

					vector<Mat>mat12 = MatSplit(trainMat, folderNum, m);
					curTrainMat = mat12[ 0 ];
					curTestMat =mat12[ 1 ];

					mat12 = MatSplit(trainScore, folderNum, m);
					curTrainScore = mat12[ 0 ];
					curTestScore = mat12[ 1 ];

					curTestRank = vecSplit(oriRank, folderNum, m);
					//------------- train svm model
					CvSVM svm;
					svm.train( curTrainMat, curTrainScore, Mat(), Mat(), params);

					//------------ test the testMat
					vector<float> predictedScore = vector<float>( curTestScore.rows, -1);
					for( int i = 0; i < curTestMat.rows; i++)
					{
						predictedScore[ i ] = svm.predict(curTestMat.row(i), true);
					} 
					curKendall  += KendallTau::kendallCal( predictedScore, curTestRank);
				}
				curKendall /= folderNum;
				cout<<"\tkendall: "<<curKendall<<endl;

				if( curKendall > maxKendall)
				{
					maxKendall = curKendall;
					bestParams[ 0 ] = cTemp;
					bestParams[ 1 ] = gTemp;
					bestParams[ 2 ] = pTemp;
				}
			}
		}
	}
	return bestParams;
}

vector<float> CrossValiParams::svrParamsLinearforRank( Mat trainMat, Mat trainScore, int cMin, int cMax, int gMin,
													  int gMax, int cStep, int gStep, int folderNum, vector<float>oriRank)
{
	vector<int>cVector = vectorForm( cMin, cMax, cStep);
	vector<int>pVector = vectorForm( gMin, gMax, gStep);

	vector<float>bestParams = vector<float>(2, -1);

	int cTemp = -1;
	int pTemp = -1;

	float maxKendall = -1;

	for( int i = 0; i < cVector.size(); i++)
	{
		cTemp = cVector[ i ];
		for( int j = 0; j < pVector.size(); j++)
		{
			pTemp = pVector[ j ];

			cout<<"C: "<<cTemp<<", g: "<<pTemp<<"\t";
			//------------ set the parameters of SVM
			CvSVMParams params;
			params.svm_type = CvSVM::EPS_SVR;
			params.kernel_type = CvSVM::LINEAR;
			params.C = pow( 2.0, cTemp);
			params.p = pow( 2.0, pTemp); 
			params.term_crit = TermCriteria( CV_TERMCRIT_ITER, (int)1e4, 1e-6);


			float curKendall = 0;
			for( int m = 0; m < folderNum; m++)
			{
				Mat curTrainMat, curTestMat;
				Mat curTrainScore, curTestScore;
				vector<float>curTestRank;

				vector<Mat>mat12 = MatSplit(trainMat, folderNum, m);
				curTrainMat = mat12[ 0 ];
				curTestMat = mat12[ 1 ];

				mat12 = MatSplit(trainScore, folderNum, m);
				curTrainScore = mat12[ 0 ];
				curTestScore = mat12[ 1 ];

				curTestRank = vecSplit(oriRank, folderNum, m);
				//------------- train svm model
				CvSVM svm;
				svm.train( curTrainMat, curTrainScore, Mat(), Mat(), params);

				//------------ test the testMat
				vector<float> predictedScore = vector<float>( curTestScore.rows, -1);
				for( int i = 0; i < curTestMat.rows; i++)
				{
					predictedScore[ i ] = svm.predict(curTestMat.row(i), true);
				} 
				curKendall  += KendallTau::kendallCal( predictedScore, curTestRank);
			}
			curKendall /= folderNum;
			cout<<"\tkendall: "<<curKendall<<endl;

			if( curKendall > maxKendall)
			{
				maxKendall = curKendall;
				bestParams[ 0 ] = cTemp;
				bestParams[ 1 ] = pTemp;
			}
		}
	}
	return bestParams;
}

float CrossValiParams::svmParams_Linear( Mat trainMat, Mat trainLabel, int cMin, int cMax,int cStep,int folderNum)
{
	vector<int>cVector = vectorForm( cMin, cMax, cStep);

	float bestParams	= -1;
	int cTemp				= -1;
	float maxPrec		= -1;

	for( int i = 0; i < cVector.size(); i++)
	{
		cTemp = cVector[ i ];

		cout<<"C: "<<cTemp<<"\t";
		//------------ set the parameters of SVM
		CvSVMParams params;
		params.svm_type		 = CvSVM::C_SVC;
		params.kernel_type	 = CvSVM::LINEAR;
		params.C			 = pow( 2.0, cTemp);
		params.term_crit	 = TermCriteria( CV_TERMCRIT_ITER, (int)1e4, 1e-6);

		float curPrec = 0;
		for( int m = 0; m < folderNum; m++)
		{
			Mat curTrainMat, curTestMat;
			Mat curTrainScore, curTestScore;
			vector<float>curTestRank;

			vector<Mat>mat12 = MatSplit(trainMat, folderNum, m);
			curTrainMat				= mat12[ 0 ];
			curTestMat				= mat12[ 1 ];

			mat12				= MatSplit(trainLabel, folderNum, m);
			curTrainScore	= mat12[ 0 ];
			curTestScore	= mat12[ 1 ];

			//------------- train svm model
			CvSVM svm;
			svm.train( curTrainMat, curTrainScore, Mat(), Mat(), params);

			//------------ test the testMat
			vector<int> predictedScore = vector<int>( curTestScore.rows, -1);
			float* dataP		= NULL;
			int correct		= 0;
			for( int i = 0; i < curTestMat.rows; i++)
			{
				predictedScore[ i ] = svm.predict(curTestMat.row(i), false);

				dataP = curTestScore.ptr<float>(i);
				if( dataP[0] == predictedScore[i])
					correct ++;
			} 
			curPrec  += (float)correct/curTestMat.rows;
		}	
		curPrec /= folderNum;
		cout<<"precision: "<<curPrec<<endl;

		if( curPrec > maxPrec)
		{
			maxPrec		= curPrec;
			bestParams	= cTemp;
		}
	}

	return bestParams;
}

vector<float> CrossValiParams::svmParams_RBF( Mat trainMat, Mat trainLabel, int cMin, int cMax, int pMin,
											 int pMax, int gMin, int gMax, int cStep, int pStep, int gStep, int folderNum)
{
	vector<int>cVector = vectorForm( cMin, cMax, cStep);
	vector<int>pVector = vectorForm( pMin, pMax, pStep);
	vector<int>gVector = vectorForm( gMin, gMax, gStep);

	vector<float>bestParams = vector<float>(3, -1);

	int cTemp = -1;
	int pTemp = -1;
	int gTemp = -1;

	float maxPrec = -1;

	for ( int i = 0; i < cVector.size(); i++)
	{
		cTemp = cVector[ i ];
		for ( int j = 0; j < pVector.size(); j++)
		{
			pTemp = pVector[ j ];
			for ( int k = 0; k < gVector.size(); k++)
			{
				gTemp = gVector[ k ];

				cout<<"C: "<<cTemp<<", p: "<<pTemp<<", g: "<<gTemp<<"\t";
				//------------ set the parameters of SVM
				CvSVMParams params;
				params.svm_type		= CvSVM::C_SVC;
				params.kernel_type	= CvSVM::RBF;
				params.C					= pow( 2.0, cTemp);
				params.p					= pow( 2.0, pTemp); 
				params.gamma			= pow( 2.0, gTemp);
				params.term_crit		= TermCriteria( CV_TERMCRIT_ITER, (int)1e4, 1e-6);

				float curPrec = 0;
				for ( int m = 0; m < folderNum; m++)
				{
					Mat curTrainMat, curTestMat;
					Mat curTrainScore, curTestScore;
					vector<float>curTestRank;

					vector<Mat>mat12 = MatSplit(trainMat, folderNum, m);
					curTrainMat				= mat12[ 0 ];
					curTestMat				= mat12[ 1 ];

					mat12				= MatSplit(trainLabel, folderNum, m);
					curTrainScore	= mat12[0];
					curTestScore	= mat12[1];

					//------------- train svm model
					CvSVM svm;
					svm.train( curTrainMat, curTrainScore, Mat(), Mat(), params);

					//------------ test the testMat
					vector<int> predictedScore = vector<int>( curTestScore.rows, -1);
					float	*dataP							= NULL;
					int correct							= 0;
					for ( int i = 0; i < curTestMat.rows; i++)
					{
						predictedScore[ i ] = svm.predict(curTestMat.row(i), false);

						dataP = curTestScore.ptr<float>( i );
						if( dataP[ 0 ] == predictedScore[ i ])
							correct ++;
					} 
					curPrec  += (float)correct/curTestMat.rows;
				}
				curPrec /= folderNum;
				cout<<"\tprecision: "<<curPrec<<endl;

				if (curPrec > maxPrec)
				{
					maxPrec			 = curPrec;
					bestParams[ 0 ] = cTemp;
					bestParams[ 1 ] = pTemp;
					bestParams[ 2 ] = gTemp;
				}
			}
		}
	}
	return bestParams;
}

vector<Mat> CrossValiParams::MatSplit(Mat srcMat, int num, int startPos)
{
	int testNum;
	if( srcMat.rows % num > startPos && srcMat.rows % num != 0)
	{
		testNum = srcMat.rows  / num + 1; 
	}else
	{
		testNum = srcMat.rows  / num;
	}
	int trainNum = srcMat.rows - testNum;

	Mat trainMat = Mat::zeros(trainNum, srcMat.cols, CV_32FC1);
	Mat testMat = Mat::zeros(testNum, srcMat.cols, CV_32FC1);

	int trainIndex = 0;
	int testIndex = 0;
	for( int i = 0; i < srcMat.rows; i++)
	{
		if( i % num == startPos)
		{
			srcMat.row( i ).copyTo( testMat.row(testIndex));
			testIndex ++;
		}else
		{
			srcMat.row( i ).copyTo( trainMat.row(trainIndex));
			trainIndex ++;
		}
	}
	vector<Mat>res = vector<Mat>(2, Mat());
	res[ 0 ] = trainMat;
	res[ 1 ] = testMat;
	return res;
}

vector<float> CrossValiParams::vecSplit( vector<float>src, int num, int startPos)
{
	int testNum;
	if( src.size() % num > startPos && src.size() % num != 0)
	{
		testNum = src.size() / num + 1; 
	}else
	{
		testNum = src.size() / num;
	}
	int trainNum = src.size() - testNum;

	vector<float>trainVec = vector<float>( trainNum, -1);
	vector<float>testVec = vector<float>( testNum, -1);

	int trainIndex = 0;
	int testIndex = 0;
	for( int i = 0; i < src.size(); i++)
	{
		if( i % num == startPos)
		{
			testVec[ testIndex ] = src[ i ];
			testIndex ++;
		}else
		{
			trainVec[ trainIndex ] = src[ i ];
			trainIndex ++;
		}
	}

	return testVec;
}



/*---------------------------------class KendallTau--------------------------------------------------------------*/

float KendallTau::kendallCal( vector<float>preRank, vector<float>oriRank)
{
	if( preRank.size() != oriRank.size())
	{
		cout<<"the size of two rank is not match"<<endl;
		return -1;
	}

	int concordant = 0;
	int discordant = 0;
	for( int i = 0; i < preRank.size() - 1; i++)
	{
		for( int j = i + 1; j < preRank.size(); j++)
		{
			float preD = preRank[ i ] - preRank[ j ];
			float oriD = oriRank[ i ] - oriRank[ j ];
			if( preD * oriD > 0)
				concordant ++;
			else discordant ++;
		}
	}
	return (float)(concordant - discordant) * 2 / (preRank.size() * (preRank.size() -1));
}

// rank of a vector
vector<int> KendallTau::rankOfVector( vector<float> src)
{
	vector<float>a = src;
	sort(a.begin(), a.end());

	// correspond each element in sorted vector with original vector
	// the first element in sorted has smallest rank value
	vector<int>rankValue( src.size(), -1);
	for( int i = 0; i < src.size(); i++){
		int index = -1;
		// correspond each element in sorted vector with original vector
		for(int j = 0; j < src.size(); j++){
			if( src[ j ] == a[ i ]){
				index = j;
				break;
			}
		}

		if( index < 0 || index >= src.size()){
			cout<<"error, no found element"<<a[ i ]<<endl;
		}
		rankValue[ index ] = i + 1;
	}
	return rankValue;
}


/*---------------------------------class FileTransform----------------------------------------------------------*/
bool FileTransform::xml2txt( string xmlFile, string saveName, string txtFile)
{
	Mat srcMat = ReadDataFromFile::readMatFromXML( xmlFile, saveName);
	float *dataP = NULL;

	if (srcMat.rows == 0)
		return false;

	ofstream fs( txtFile);
	for (int i = 0; i < srcMat.rows; i++)
	{
		dataP = srcMat.ptr<float>(i);
		for ( int j = 0; j < srcMat.cols; j++)
		{
			fs<<dataP[j]<<"\t";
		}
		fs<<endl;
	}
	fs.close();
	return true;
}

/*--------------------------------------------------------------------------------*/
// is image valid in trainset and test; return Mat or null Mat 
Mat isImgExist( string trainSetPath, string testSetPath, string imgName)
{
	Mat img = imread( trainSetPath + imgName + "G.jpg");
	if( img.rows == 0)
		img = imread( trainSetPath + imgName + "B.jpg");
	if( img.rows == 0)
		img = imread( testSetPath + imgName + "G.jpg");
	if( img.rows == 0)
		img = imread( testSetPath + imgName + "B.jpg");

	return img;
}

// the cols of input mat should be 1
float mseCal4Mat( Mat mat1, Mat mat2) 
{ 
	float mse = 0;
	if( mat1.rows != mat2.rows){
		cout<<"MSE calculation the rows of two Mat is not match "<<endl<<endl;
	}

	for( int i = 0; i < mat1.rows; i++){
		float* dataP1 = mat1.ptr<float>( i );
		float* dataP2 = mat2.ptr<float>( i );

		mse += ( dataP1[ 0 ] - dataP2[ 0 ]) * ( dataP1[ 0 ] - dataP2[ 0 ]);
	}
	mse = sqrt( mse / mat1.rows);
	return mse;
}

vector<int>vectorForm( int min, int max, int step)
{
	int size = ( max - min ) / step + 1;
	vector<int>vec(size, 0);
	vec[ 0 ] = min;

	for( int i = 1; i < size; i++){
		vec[ i ] = vec[ i - 1 ] + step;
	}
	return vec;
}

vector<float>vectorForm( float min, float max, float step)
{
	int size = ( max - min ) / step + 1;
	vector<float>vec(size, 0);
	vec[ 0 ] = min;
	for( int i = 1; i < size; i++){
		vec[ i ] = vec[ i - 1 ] + step;
	}
	return vec;
}

// exe a string command
bool exeCommand(string str)
{
	char cmd[500];
	for (int i = 0; i < str.size(); i++)
	{
		cmd[i] = str[i];
	}
	cmd[str.size()] = '\0';
	system(cmd);
	return true;
}


bool makeDir(string dir)
{
	char tmp[300];
	for (int i = 0; i < dir.size(); i++)
	{
		tmp[i] = dir[i];
	}
	tmp[dir.size()] = '\0';
	//----- folder already exists
	if (_access(tmp, 0)!=-1)
		return true;

	if(_mkdir(tmp)==0)	
		return true;
	else {
		cout<<"cannot create folder:"<<dir<<endl;
		return false;
	}
}
