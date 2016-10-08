#include "Head.h"
#include "basic_function.h"

// only positive numver
string IntToStr(int num)
{
	if (num==0)
		return "0";
	string res = "";
	while (num != 0)
	{
		res.push_back( '0'+num%10 );
		num -= num%10;
		num /= 10;
	}
	reverse(res.begin(), res.end());
	return res;
}

void SaveTopK()
{
	string cate = "cate_12\\";
	string qImg = "G300222.jpg";
	int topK = 25;

	string trainPath = "E:\\Lab\\Category_Ranking_AVA_14\\DCNN_test\\category_based_train\\"; 
	string dataPath = "E:\\Dataset_AVA_14\\category_based_train\\";

	string savePath = "E:\\Lab\\paper_AVA_14\\" + qImg + "\\";
	makeDir(savePath);

	string tmp = "";
	ifstream fs(trainPath+cate+qImg+".txt");
	for (int i=0; i<=topK; i++)
	{
		if (i<topK)
		{
			fs>>tmp;
			Mat img = imread(dataPath+tmp);
			imwrite(savePath + IntToStr(i) + ".jpg", img);
		}else
		{
			Mat img = imread(dataPath+cate+qImg);
			imwrite(savePath + qImg, img);
		}
	}
}


void test()
{

	cout << "first run after more than three month" << endl;
//	SaveTopK();

//	string cmd = "F:\\Google\\torrent\\extract.deep.feat\\extract.deep.feat\\extract_features2files.exe F:\\ModelTmp\\ava_train_full_iter_20000_noSigmoid.caffemodel F:\\ModelTmp\\proto_pair_ranking_train_noSigmoid.prototxt reg3_g H:\\Lab\\DeepAesthRankNet\\image_score_pred_20000_noSigmoid.txt 5387";
//	exeCommand(cmd);

	/*
	Directory dir;
	string pathTest = "F:\\Lab\\Category_Ranking_AVA_14\\category_based_train\\";
	vector<string>cateNames = dir.GetListFolders(pathTest, "*", false);

	string pathBase = "E:\\Dataset_AVA_14\\category_based_train\\";
	int count = 0;
	for (int i = 0; i < cateNames.size(); i++)
	{
		string currPath = pathTest + cateNames[i] + "\\";
		cout<<currPath<<endl;
		vector<string> imgNames = dir.GetListFiles(currPath, "*.jpg", false);
		
		count += imgNames.size();
		for (int j = 0; j < imgNames.size(); j++)
		{

			Mat img = imread(pathBase+cateNames[i]+"\\"+imgNames[j]);
			if (img.rows == 0) // not valid
				cout<<cateNames[i]+"\\"+imgNames[j]<<endl;
		}	
	}
	cout<<"total number: "<<count<<endl;
	*/
	/*
	Directory dir;
	string savePath = "E:\\Lab\\DeepAesthRankNet\\Pred_cate\\";
	vector<string> cateNames = dir.GetListFolders(savePath, "*", false);
	
	for (int i = 0; i < cateNames.size(); i++)
	{
		string currPath = savePath + cateNames[i] + "\\";
		makeDir(currPath + "models_10_29_2\\");
	}
	*/
}
