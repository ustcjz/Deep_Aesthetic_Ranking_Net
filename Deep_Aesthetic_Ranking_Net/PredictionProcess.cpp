#include "PredictionProcess.h"
#include "basic_function.h"
/**
	divide the prediction
*/
void DividePred()
{
	Directory dir;

	const string pathBase = "F:\\Lab\\Category_Ranking_AVA_14\\category_based_test\\";
	vector<string> cateNames = dir.GetListFolders(pathBase, "*", false);		
	vector<int> boundVec(cateNames.size(), -1);
	
	int count = 0;
	for (int i = 0; i < cateNames.size(); i++) // calc the boundVec
	{
		string currPath = pathBase + cateNames[i] + "\\";
		vector<string> imgNames = dir.GetListFiles(currPath, "*.jpg", false);
		count += imgNames.size();
		boundVec[i] = count;
	}

	string path = "H:\\Lab\\DeepAesthRankNet\\PredTest_Pairwise\\";
	string fileName = "";
	string saveName = "";
	string savePath = "E:\\Lab\\DeepAesthRankNet\\Pred_cate\\";
	
	string layerName = "score_g";

	int low = 500000;
	int high = 1000000;
	int step = 100000;

	vector<string> modelNameVec(1, "");
	modelNameVec[0] = "models_01_08\\";
//	modelNameVec[1] = "models_01_11\\";

	for (int j=0; j<modelNameVec.size(); j++)
	{
		string modelsName = modelNameVec[j];
		for (int i = low; i <= high; i+=step)
		{
			fileName = path+modelsName+"image_score_pred_"+FormatTransform::int2str(i)+"_"+layerName+".txt"; // full path of predicted all scores
			saveName = "score_pred_"+FormatTransform::int2str(i)+".txt"; // the name of divided files, not full path
			cout<<fileName<<endl<<endl;
			cout<<saveName<<endl<<endl;
			DividePredOne(fileName, saveName, savePath, modelsName, boundVec);	
		}
	}
}


/**
	divide the one prediction into 14 categories
*/
void DividePredOne(string fileName, string saveName, string savePath, string modelsName, vector<int>& boundVec)
{
	const int cateNum = 14;

	Directory dir;
	vector<string> cateNames = dir.GetListFolders(savePath, "*", false);
	if (cateNames.size() != cateNum)
	{
		cout<<"not the right path to save divied prediction"<<endl;
		cout<<savePath<<endl;
		return;
	}

	ifstream fsIn(fileName);
	if (!fsIn.is_open())
	{
		cout<<"cannot open file: "<<fileName<<endl;
		return;
	}


	string tmp = "";
	int index = 0;
	for (int i = 0; i < cateNames.size(); i++)
	{
		string currPath = savePath + cateNames[i] + "\\" + modelsName;
		makeDir(currPath); // makedir for a set of new caffemodels
		ofstream fsOut(currPath + saveName);
		if (!fsOut.is_open())
		{
			cout<<"cannot open file: "<<currPath+saveName<<endl;
			return;
		}
		cout<<currPath<<endl;

		while(index<boundVec[i])	
		{
			getline(fsIn, tmp);
			fsOut<<tmp<<endl;
			index++; // write a line
		}
		fsOut.close();
	}
	cout<<"total image num: "<<index<<endl;
}

void PredCaffeTest()
{
	string extractCmd = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\extract_features2files.exe";
	//string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test.prototxt";
	//string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test_with_mean.prototxt";

	//string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test_single_channel.prototxt";
	string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test_single_channel_with_mean.prototxt";

	int low = 500000;
	int high = 1000000;
	int step = 100000;

	string modelPathBase = "H:\\Lab\\DeepAesthRankNet\\Models_Pairwise\\";
	string savePathBase = "H:\\Lab\\DeepAesthRankNet\\PredTest_Pairwise\\";

	///////////////////////////////// for narmal use
	/*
	string modelName = "models_12_04\\";
	string modelPath = modelPathBase + modelName; // alread exists
	string savePath = savePathBase + modelName; // need to make new folder
	makeDir(savePath);

	string layerName = "score_g";
	string batchNum = "5387"; // 26935/5=5387

	for (int i = start; i <=  end; i++)
	{
		string iterCaffe = "ava_train_full_iter_"+FormatTransform::int2str(i*step)+".caffemodel";
		string iterPred = "image_score_pred_"+FormatTransform::int2str(i*step)+"_"+layerName+".txt";
	//	cout<<testPrototxt<<endl<<endl;
		cout<<modelPath+iterCaffe<<endl<<endl;
		cout<<savePath+iterPred<<endl<<endl;
		PredCaffeTestOne(extractCmd, modelPath+iterCaffe, testPrototxt, layerName, savePath+iterPred, batchNum);
	}
	*/

	//////////////////////////////// for models vectors
	int modelNum = 1;
	vector<string> modelNameVec(modelNum, "");
	modelNameVec[0] = "models_01_08\\";
//	modelNameVec[1] = "models_01_11\\";
//	modelNameVec[1] = "models_12_22_1_NG\\";
//	modelNameVec[1] = "models_12_21_1_NG\\";
//	modelNameVec[1] = "models_12_14_1_NG\\";

//	modelNameVec[-1] = "models_12_20_NG";
	for (int j=0; j<modelNameVec.size(); j++)
	{
		string modelPath = modelPathBase + modelNameVec[j]; // alread exists
		string savePath = savePathBase + modelNameVec[j]; // need to make new folder
		makeDir(savePath);

		string layerName = "score_g";
		string batchNum = "5387"; // 26935/5=5387

		for (int i = low; i <= high; i+=step)
		{
			string iterCaffe = "ava_train_full_iter_"+FormatTransform::int2str(i)+".caffemodel";
			string iterPred = "image_score_pred_"+FormatTransform::int2str(i)+"_"+layerName+".txt";
			cout<<modelPath+iterCaffe<<endl<<endl;
			cout<<savePath+iterPred<<endl<<endl;
			PredCaffeTestOne(extractCmd, modelPath+iterCaffe, testPrototxt, layerName, savePath+iterPred, batchNum);
		}
	}

}

void testCaffe()
{
//	string extractCmd = "F:\\caffe_quality_assessment\\caffe-quality_assessment\\bin\\MainBuild.exe";
	string extractCmd = "F:\\MainBuild\\MainBuild.exe";
	string extractCmd_s = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\extract_features2files.exe";

	//string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test.prototxt";
	//string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test_with_mean.prototxt";

	//string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test_single_channel.prototxt";
	string testPrototxt = "E:\\Lab\\DeepAesthRankNet\\extract.deep.feat\\extract.deep.feat\\proto_pair_ranking_test_single_channel_with_mean.prototxt";

	int low = 500000;
	int high = 500000;
	int step = 100000;

	string modelPathBase = "H:\\Lab\\DeepAesthRankNet\\Models_Pairwise\\";
	string savePathBase = "H:\\Lab\\DeepAesthRankNet\\PredTest_Pairwise\\";

	///////////////////////////////// for narmal use
	/*
	string modelName = "models_12_04\\";
	string modelPath = modelPathBase + modelName; // alread exists
	string savePath = savePathBase + modelName; // need to make new folder
	makeDir(savePath);

	string layerName = "score_g";
	string batchNum = "5387"; // 26935/5=5387

	for (int i = start; i <=  end; i++)
	{
		string iterCaffe = "ava_train_full_iter_"+FormatTransform::int2str(i*step)+".caffemodel";
		string iterPred = "image_score_pred_"+FormatTransform::int2str(i*step)+"_"+layerName+".txt";
	//	cout<<testPrototxt<<endl<<endl;
		cout<<modelPath+iterCaffe<<endl<<endl;
		cout<<savePath+iterPred<<endl<<endl;
		PredCaffeTestOne(extractCmd, modelPath+iterCaffe, testPrototxt, layerName, savePath+iterPred, batchNum);
	}
	*/

	//////////////////////////////// for models vectors
	int modelNum = 1;
	vector<string> modelNameVec(modelNum, "");
	modelNameVec[0] = "models_01_08\\";
//	modelNameVec[1] = "models_01_11\\";
//	modelNameVec[1] = "models_12_22_1_NG\\";
//	modelNameVec[1] = "models_12_21_1_NG\\";
//	modelNameVec[1] = "models_12_14_1_NG\\";

//	modelNameVec[-1] = "models_12_20_NG";
	for (int j=0; j<modelNameVec.size(); j++)
	{
		string modelPath = modelPathBase + modelNameVec[j]; // alread exists
		string savePath = savePathBase + modelNameVec[j]; // need to make new folder
		makeDir(savePath);

		string layerName = "score_g";
		string batchNum = "50"; // 26935/5=5387

		for (int i = low; i <= high; i+=step)
		{
			string iterCaffe = "ava_train_full_iter_"+FormatTransform::int2str(i)+".caffemodel";
			string iterPred = "image_score_pred_"+FormatTransform::int2str(i)+"_"+layerName+".txt";
			cout<<modelPath+iterCaffe<<endl<<endl;
			cout<<savePath+iterPred<<endl<<endl;
			PredCaffeTestOne(extractCmd, modelPath+iterCaffe, testPrototxt, layerName, savePath+iterPred, batchNum);
		}
	}

}

void PredCaffeTestOne(string extractCmd, string caffeModel, string testPrototxt, 
					  string layerName, string saveFile, string batchNum)
{
	exeCommand(extractCmd+" "+caffeModel+" "+testPrototxt+" "+layerName+" "+saveFile+" "+batchNum);
}
