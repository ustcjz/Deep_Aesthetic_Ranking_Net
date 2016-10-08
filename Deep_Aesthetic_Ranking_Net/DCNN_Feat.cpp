#include"DCNN_Feat.h"
#include"basic_function.h"

/*
	retrieval and save topK img 
 */
void retrievalTest()
{
//	string pathBase = "";
//	string qImg = srcFold + "animal\\B1112.jpg";
//	int topK = 10;


//	const string saveFold = "F:\\Lab\\My_CUHK_Dataset\\DCNN_test\\category_based_t\\";
//	string queryFold = "F:\\Lab\\My_CUHK_Dataset\\category_based_test\\";
	const string saveFold = "E:\\Lab\\Category_Ranking_AVA_12\\DCNN_test\\category_based_train\\";
	const string queryFold = "F:\\Lab\\Category_Ranking_AVA_12\\category_based_train\\";

	const string srcFold = "F:\\Lab\\Category_Ranking_AVA_12\\category_based_train\\";

	cout<<"reading all dcnnFeatVec...";
	vector<vector<float>> dcnnFeatVec = readAllDcnnFeat(srcFold);
	cout<<"done!"<<endl;

	Directory dir;
	vector<string> foldNames = dir.GetListFolders(queryFold, "*", false);
	for (int i=0; i<foldNames.size(); i++)
	{
		vector<string> imgNames = dir.GetListFiles(queryFold+foldNames[i], "*.jpg", false);
		for (int j=0; j<imgNames.size(); j++)
		{
			string queryImg = queryFold+foldNames[i]+"\\"+imgNames[j];
			string saveFile = saveFold+foldNames[i]+"\\"+imgNames[j]+".txt";
			cout<<queryImg<<endl;
			retrievalAndSave(queryImg, srcFold, saveFile, dcnnFeatVec);
		}
	}

//	retrievalTopK(qImg, topK, srcFold, saveFold);
}

/*
	retrieval and save topK img 
 */
void retrievalTopK(string qImg, int topK, string srcFold, string saveFold)
{
	//--- initialization
	int featNum = 4096;
	ImgInfo imgInfo;
	imgInfo.imgName = "-1";
	imgInfo.dis = -1;

	//--- get all imgInfoVec, whick stores the distance to the quety img
	const int imgNum = 8845;
	vector<ImgInfo> imgInfoVec(imgNum, imgInfo);

	clock_t start, end;
	start = clock();

	readImgInfoVec( qImg, srcFold, imgInfoVec);
//	readImgInfoVecXML(qImg, srcFold, imgInfoVec);
	end = clock();
	cout<<"time consuming of read imgInfoVec is: "<<(double)(end-start)/CLOCKS_PER_SEC;
	cout<<endl<<endl;

	//--- sort based on distance
	start = clock();
	sort(imgInfoVec.begin(), imgInfoVec.end(), ImgInfoSort);
	end = clock();
	cout<<"time consuming of sort imgInfoVec is: "<<(double)(end-start)/CLOCKS_PER_SEC;
	cout<<endl<<endl;

	start = clock();
	ofstream fs(saveFold+"retrieval.txt");
	for (int i=0; i<imgInfoVec.size(); i++)
	{
		fs<<imgInfoVec[i].imgName<<endl;
	}
	fs.close();
	end = clock();
	cout<<"time consuming of writing file is: "<<(double)(end-start)/CLOCKS_PER_SEC;
	cout<<endl<<endl;

//	//--- save the topK img to saveFold
//	for (int i=0; i<topK; i++)
//	{
//		//--- TODO save img
//		Mat img = imread(srcFold + imgInfoVec[i].imgName);
//		string saveName = saveFold + "_" + FormatTransform::int2str(i) + 
//			"_" + imgInfoVec[i].imgName;
//		imwrite(saveName, img);
//	}
}


/*
	retrieval on training set and save table
	input:
		qImg: name of query image
		srcFold: the path of the image set 
		saveFile: the sort result to be saved
 */
void retrievalAndSave(string qImg, string srcFold, string saveFile)
{
	//--- initialization
	int featNum = 4096;
	ImgInfo imgInfo;
	imgInfo.imgName = "-1";
	imgInfo.dis = -1;

	//--- get all imgInfoVec, whick stores the distance to the quety img
	int imgNum = getImgNum(srcFold);
	vector<ImgInfo> imgInfoVec(imgNum, imgInfo);

	//--- read imginfo in training set
	readImgInfoVec( qImg, srcFold, imgInfoVec);

	//--- sort based on distance
	sort(imgInfoVec.begin(), imgInfoVec.end(), ImgInfoSort);

	ofstream fs(saveFile);
	//for (int i=0; i<imgInfoVec.size(); i++)
	for (int i=1; i<imgInfoVec.size(); i++)
	{
		fs<<imgInfoVec[i].imgName<<endl;
	}
	fs.close();
}

void retrievalAndSave(string qImg, string srcFold, string saveFile, vector<vector<float>>& dcnnFeatVec)
{
	//--- initialization
	int featNum = 4096;
	ImgInfo imgInfo;
	imgInfo.imgName = "-1";
	imgInfo.dis = -1;

	//--- get all imgInfoVec, whick stores the distance to the quety img
	int imgNum = getImgNum(srcFold);
	vector<ImgInfo> imgInfoVec(imgNum, imgInfo);

	//--- read imginfo in training set
	readImgInfoVec( qImg, srcFold, imgInfoVec, dcnnFeatVec);

	//--- sort based on distance
	sort(imgInfoVec.begin(), imgInfoVec.end(), ImgInfoSort);

	ofstream fs(saveFile);
	//for (int i=0; i<imgInfoVec.size(); i++)
	for (int i=1; i<imgInfoVec.size(); i++)
	{
		fs<<imgInfoVec[i].imgName<<endl;
	}
	fs.close();
}

/*---------------------------------------------------------------------------------*/
bool readBinaryFile(string fileNames, int featNum, vector<float>& feat)
{
	feat.clear();
	feat = vector<float>(featNum, -1);

	//--- feature of DCNN
	ifstream fs(fileNames, ios::binary);
	if (!fs.is_open())
		return false;

	float a;
	float sum = 0;

	for (int i=0; i < featNum; i++)
	{
		fs.read((char*)&a, sizeof(float));
		sum += a;
		feat[i] = a;
	}
	fs.close();

	//--- normalization of DCNN feature
	sum = sqrt(sum);
	for (int i=0; i < featNum; i++)
	{
		feat[i] = sqrt( feat[i] ) / sum;
	}
	return true;
}

/*
 * form all imginfo
	input:
		qImg: query image
		srcFold: image dataset
		imgInfoVec: class to save distance to query image
 */
bool readImgInfoVec( string qImg, string srcFold, vector<ImgInfo>& imgInfoVec)
{
	//--- the feat of query image
	const int featNum = 4096;
	vector<float> qImgFeat(featNum, -1);
	readBinaryFile(qImg+".feat", featNum, qImgFeat);

	//--- the distance to each img in training set
	Directory dir;
	vector<string> foldNames = dir.GetListFolders(srcFold, "*", false);

	string dcnnFileName = "";
	vector<float> currImgFeat(featNum, -1);

	int index = 0;
	for (int i=0; i<foldNames.size(); i++)
	{
		vector<string> imgNames = dir.GetListFiles(srcFold+foldNames[i], "*.jpg", false);
		for (int j=0; j<imgNames.size(); j++)
		{
			imgInfoVec[index].imgName = foldNames[i] + "\\" + imgNames[j];
			
			dcnnFileName = srcFold + foldNames[i] + "\\" + imgNames[j] + ".feat";
			readBinaryFile(dcnnFileName, featNum, currImgFeat);
			imgInfoVec[index].dis = disCal(currImgFeat, qImgFeat);

			index ++;
		}
	}
	return true;
}


bool readImgInfoVec( string qImg, string srcFold, vector<ImgInfo>& imgInfoVec, vector<vector<float>>& dcnnFeatVec)
{
	//--- the feat of query image
	const int featNum = 4096;
	vector<float> qImgFeat(featNum, -1);
	readBinaryFile(qImg+".feat", featNum, qImgFeat);

	//--- the distance to each img in training set
	Directory dir;
	vector<string> foldNames = dir.GetListFolders(srcFold, "*", false);

	int index = 0;
	for (int i=0; i<foldNames.size(); i++)
	{
		vector<string> imgNames = dir.GetListFiles(srcFold+foldNames[i], "*.jpg", false);
		for (int j=0; j<imgNames.size(); j++)
		{
			imgInfoVec[index].imgName = foldNames[i] + "\\" + imgNames[j];
			imgInfoVec[index].dis = disCal(qImgFeat, dcnnFeatVec[index]);
			index ++;
		}
	}
	return true;
}

/*
	calculate the distance between 2 vectors
 */
float disCal(vector<float>& src, vector<float>& dst)
{
	if (src.size() != dst.size())
	{
		cout<<"size of two vectors not match"<<endl;
		return -1;
	}

	float dis = 0;
	for (int i=0; i<src.size(); i++)
	{
		dis += (src[i]-dst[i])*(src[i]-dst[i]);
	}
	return dis;
}

/*
	userdefined sort of class ImgInfo
 */
bool ImgInfoSort( ImgInfo a, ImgInfo b)
{
	return (a.dis < b.dis);
}


/*-------------------------------------------------------------------------------*/
/*
	transform all .feat file to a XML file
 */
void feat2XML()
{
	string path = "F:\\Lab\\My_CUHK_Dataset\\category_based_train\\animal\\";
	string savePath = "F:\\Lab\\My_CUHK_Dataset\\DCNN_test\\animal\\";
	int featNum = 4096;
	// mat to save .feat file
	Directory dir;
	vector<string> imgNames = dir.GetListFiles(path, "*.jpg", false);

	Mat featMat = Mat::zeros(imgNames.size(), featNum, CV_32FC1);

	Mat rowMat = Mat::zeros(1, featNum, CV_32FC1);
	// load .sift to Mat
	vector<float> imgFeat(4096, -1);
	for (int i=0; i<imgNames.size(); i++)
	{
		readBinaryFile( path+imgNames[i]+".feat", featNum, rowMat); 
		rowMat.copyTo(featMat.row(i));
	}

	// save Mat to XML file
	FileStorage fs(savePath+"featMat.XML", FileStorage::WRITE);
	fs<<"feat"<<featMat;
	fs.release();
}

bool readBinaryFile(string fileNames, int featNum, Mat& feat)
{
	//--- feature of DCNN
	ifstream fs(fileNames, ios::binary);
	if (!fs.is_open())
		return false;

	float a;
	float sum = 0;

	float* dataP = feat.ptr<float>(0);
	for (int i=0; i < featNum; i++)
	{
		fs.read((char*)&a, sizeof(float));
		sum += a;
		dataP[i] = a;
	}
	fs.close();

	//--- normalization of DCNN feature
	sum = sqrt(sum);
	for (int i=0; i < featNum; i++)
	{
		dataP[i] = sqrt( dataP[i] ) / sum;
	}
	return true;
}

/*
 * form all imginfo
 */
bool readImgInfoVecXML( string qImg, string srcFold, vector<ImgInfo>& imgInfoVec)
{
	clock_t st, end;
	st = clock();

	// read xml file to get mat
	string path = "F:\\Lab\\My_CUHK_Dataset\\DCNN_test\\animal\\";
	Mat featMat;
	FileStorage fs(path+"featMat.xml", FileStorage::READ);
	fs["feat"]>>featMat;
	fs.release();

	//---get image names
	Directory dir;
	string imgPath = "";
	vector<string> imgNames = dir.GetListFiles(imgPath, "*.jpg", false);

	int featNum = 4096;
	vector<float> qImgFeat(featNum, -1);
	readBinaryFile(qImg, featNum, qImgFeat);
	
	for (int i=0; i<featMat.rows; i++)
	{
		imgInfoVec[i].imgName = imgNames[i];

		float dis = 0;
		float* dataP = featMat.ptr<float>(i);
		for (int j=0; j<featMat.cols; j++)
		{
			dis += (dataP[j]-qImgFeat[j])*(dataP[j]-qImgFeat[j]);
		}

		imgInfoVec[i].dis = dis;
	}
	end = clock();
	cout<<"time consuming is: "<<(double)(end-st)/CLOCKS_PER_SEC<<endl;
	return true;
}


int getImgNum(string srcFold)
{
	Directory dir;
	vector<string> foldNames = dir.GetListFolders( srcFold, "*", false);

	int count = 0;
	for (int i=0; i<foldNames.size(); i++)
	{
		vector<string> imgNames = dir.GetListFiles( srcFold+foldNames[i], "*.jpg", false);
		count += imgNames.size();
	}
	return count;

}

/*
	read .aesthFeat file
*/
bool readAesthFeatFile(string fileName, int featNum, vector<float>& feat)
{
	ifstream fs(fileName);
	if (!fs.is_open())
	{
		cout<<"connnot open file: "<<fileName;
		return false;
	}

	feat = vector<float> (featNum, -1);
	string tmp = "";
	
	for (int i=0; i<featNum; i++)
	{
		fs>>tmp;
		feat[i] = FormatTransform::string2float(tmp);
	}
	fs.close();
	return true;
}

/*
	read .aesthLabel file
*/
bool readAesthLabelFile(string fileName, int& label)
{
	ifstream fs(fileName);
	if (!fs.is_open())
	{
		cout<<"connnot open file: "<<fileName;
		return false;
	}

	string tmp = "";
	fs>>tmp;
	label = FormatTransform::string2int(tmp);
	fs.close();
	return true;
}

/*
	read .featIndex file
*/
bool readFeatIndexFile(string fileName, int& featIndex)
{
	ifstream fs(fileName);
	if (!fs.is_open())
	{
		cout<<"cannot open file: "<<fileName<<endl;
		return false;
	}
	fs>>featIndex;
	fs.close();
	return true;
}


/*
	read .score file, which save the original score of each image
*/
bool readScoreFile(string fileName, float& score)
{
	ifstream fs(fileName);
	if (!fs.is_open())
	{
		cout<<"cannot open file : "<<fileName<<endl;
		return false;
	}
	fs>>score;
	fs.close();
	return true;
}


vector<vector<float>> readAllDcnnFeat(string srcFold)
{
	int imgNum = getImgNum(srcFold);
	vector<float> dcnnFeat(4096, -1);
	vector<vector<float>> dcnnFeatVec(imgNum, dcnnFeat);
	int index = 0;

	Directory dir;
	vector<string> dirNames = dir.GetListFolders(srcFold, "*", false);
	for (int i = 0; i < dirNames.size(); i++)
	{
		string currDir = srcFold + dirNames[i] + "\\";
		vector<string> dcnnFiles = dir.GetListFiles(currDir, "*.feat", false);

		for (int j = 0; j < dcnnFiles.size(); j++)
		{
			readBinaryFile(currDir+dcnnFiles[j], 4096, dcnnFeat);
			dcnnFeatVec[index] = dcnnFeat;
			index ++;
		}
	}
	return dcnnFeatVec;
}
