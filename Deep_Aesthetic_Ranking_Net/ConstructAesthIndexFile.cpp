#include"ConstructAesthIndexFile.h"
#include"DCNN_Feat.h"

void consAesthIndexFile()
{
	string trainTest = "train";
	const string trainPath = "F:\\Lab\\Category_Ranking_AVA_14\\category_based_"+trainTest+"\\"; // path to be changed
	Directory dir;
	vector<string> cateNames = dir.GetListFolders(trainPath, "*", false);


	/*---------- construct the Map<string(imgName), int(featIndex)> -------------*/
	/*---------- construct the scoreVec -------------*/
	/*---------- construct the namesVec -------------*/
	map<string, int> imgToIndex;
	vector<float> scoreVec(getImgNum(trainPath), -1);
	vector<string> namesVec(getImgNum(trainPath), "");
	vector<string> namesServerVec(namesVec.size(), "");

	int index = 0;
	for (int i = 0; i < cateNames.size(); i++)
	{
		string currDir = trainPath + cateNames[i] + "\\";
		vector<string> scoreNames = dir.GetListFiles(currDir, "*.jpg.score", false);
		vector<string> imgNames = dir.GetListFiles(currDir, "*.jpg", false);

		if (imgNames.size() != scoreNames.size())
		{
			cout<<"size not match"<<endl;
			return;
		}

		for (int j = 0; j < scoreNames.size(); j++)
		{
			float score = -1;
			readScoreFile(currDir+scoreNames[j], score);
			scoreVec[index] = score; // save the score
	
			string insStr = cateNames[i] + "\\" + imgNames[j];
			imgToIndex.insert(pair<string, int>(insStr, index)); // insert a pair to map

			namesVec[index] = insStr; // save the image name
			namesServerVec[index] = cateNames[i] + "/" + imgNames[j]; // the names path on the server

			index++;
		}
	}

	const string saveBase = "E:\\"; // path to be changed

	float scoreGap = 0.5;
	vector<string> saveFileVec(1, "");
	vector<int> numBoundVec(1, -1);
	vector<int> biliVec(1, -1);

	for (int i = 0; i < saveFileVec.size(); i++)
	{
		saveFileVec[i] = saveBase + "aesth_index_05_random_valid.txt";
		numBoundVec[i] = 50;
		biliVec[i] = 10;
	}
//	consOneAesthIndexFile(scoreGap, saveFileVec, numBoundVec, scoreVec, namesVec, namesServerVec, imgToIndex);
	consOneAesthIndexFile_random(scoreGap, saveFileVec, numBoundVec, biliVec,
		scoreVec, namesVec, namesServerVec, imgToIndex);
}


/**
input:
	scoreGap: gap to define whether two images are comparable or not
	saveFilevec: save several files once
	numBoundvec: correspond numBound 

pre_calculated, for efficiency
	scoreVec : <imgIndex(int), imgScore(float)>
	imgNamesVec : <imgIndex(int), imgNames(string)>
	imgToIndex : pair<imgNames(string), imgIndex(int)>

output:
	save into saveFileVec
*/
void consOneAesthIndexFile(float scoreGap, vector<string>saveFileVec, vector<int>numBoundVec, 
						   vector<float>&scoreVec, vector<string>&imgNamesVec, vector<string>&namesServerVec, map<string,int>& imgToIndex)
{
	string trainTest = "train";
	// fixed variables
	const string trainPath = "F:\\Lab\\Category_Ranking_AVA_14\\category_based_"+trainTest+"\\";
	const string retrTblPath = "E:\\Lab\\Category_Ranking_AVA_14\\DCNN_test\\category_based_"+trainTest+"\\";
	int numRow = 100;
	int imgNumTotal = getImgNum(trainPath);

	vector<ofstream> fsVec(saveFileVec.size()); // open saveFileVec
	for (int i = 0; i < fsVec.size(); i++)
	{
		fsVec[i].open(saveFileVec[i]);
		if ( !fsVec[i].is_open() )
		{
			cout<<"cannot open file: "<<saveFileVec[i]<<endl;
			return;
		}
	}


	Directory dir;
	vector<string> cateNames = dir.GetListFolders( trainPath, "*", false );

//	const string aesthBase = "/home/LvHao/DeepAesthRankingNet/Dataset_AVA_14/category_based_"+trainTest+"/";
	const string aesthBase = "/home/lvhao/DeepAesthRankingNet/Dataset_AVA_14/category_based_"+trainTest+"/";
	string aesthIndex = "";

	int featIndexQ = -1;
	int qid = 0;
	for (int i = 0; i < cateNames.size(); i++) // traverse for each query image
	{
		string currDir = trainPath + cateNames[i] + "\\";
		vector<string> imgNames = dir.GetListFiles(currDir, "*.jpg", false);
		
		for (int j = 0; j < imgNames.size(); j++)
		{
			featIndexQ++;
			qid ++;
			cout<<cateNames[i] + "\\" + imgNames[j] + "	\t"<<scoreGap<<" \t"<<featIndexQ<<endl;

//			for (int k = 0; k < fsVec.size(); k++) // write query img info to filesVec
//			{
//				fsVec[k]<<"1 qid:"<<qid<<" 1:"<<featIndexQ<<endl;
//			}

			vector<int> aRow(numBoundVec.back()+10, -1);
			vector<vector<int>> fIndexVec(fsVec.size(), aRow); // the vector to save the featIndex smaller than query image
			vector<int> indexB(fsVec.size(), 0); //index  of image that with bigger score
			vector<int> indexS(fsVec.size(), 0); //index  of image that with smaller score

			ifstream fsNei(retrTblPath + cateNames[i] + "\\" + imgNames[j] + ".txt");// open neighbor name table
			if (!fsNei.is_open())
			{
				cout<<"cannot open file: "<<retrTblPath + cateNames[i] + "\\" + imgNames[j]<<endl;
				return;
			}

			float scoreQ = scoreVec[featIndexQ];
			string tmp1 = "";
			int times = 0;
			bool flagAll = false;// all file are full or not flag

			while(!fsNei.eof() && times < imgNumTotal) // find all the bigger images
			{
				fsNei>>tmp1;
				times++;

				int featIndexN = -1;
				float scoreN = -1;
				featIndexN = imgToIndex[tmp1];
				scoreN = scoreVec[featIndexN]; // the score and featIndex of a neighbor
			//	cout<<"nei:   "<<tmp1<<"\t"<<scoreN<<endl;

				if (scoreN <= scoreQ - scoreGap) // smaller neighbors
				{
					for (int k = 0; k < fsVec.size(); k++)
					{
						if (indexB[k] + indexS[k] >= numBoundVec[k])
						{
							if ( k==fsVec.size()-1 ) flagAll = true; // all saveFile is full
							continue;
						}
						
						fIndexVec[k][indexS[k]] = featIndexN;
						indexS[k]++;
					}
				}
				else if (scoreN >= scoreQ + scoreGap){
					for (int k = 0; k < fsVec.size(); k++)
					{
						if (indexB[k] + indexS[k] >= numBoundVec[k])
						{
							if ( k==fsVec.size()-1 ) flagAll = true; // all saveFile is full
							continue;
						}

					//	cout<<"2 qid:"<<qid<<" 1:"<<featIndexN<<endl;
						writeAesthLine(fsVec[k], featIndexN, featIndexQ, aesthBase, namesServerVec); // featIndexN is better quality
						indexB[k]++;
					}
				}

				if (flagAll) break;
			}
			fsNei.close();

			qid ++;
			for (int k = 0; k < fsVec.size(); k++) // write all the smaller images
			{
			//	fsVec[k]<<"2 qid:"<<qid<<" 1:"<<featIndexQ<<endl;

				for (int m = 0; m < indexS[k]; m++)
				{
				//	cout<<"1 qid:"<<qid<<" 1:"<<fIndexVec[k][m]<<endl;
					writeAesthLine(fsVec[k], featIndexQ, fIndexVec[k][m], aesthBase, namesServerVec); // featIndexQ is better quality
				}
			}
		}
	}

	for (int i = 0; i < fsVec.size(); i++)
	{
		fsVec[i].close();
	}
}


/**
input:
	scoreGap: gap to define whether two images are comparable or not
	saveFilevec: save several files once
	numBoundvec: correspond numBound 
	biliVec: random select with corresponding bili

pre_calculated, for efficiency
	scoreVec : <imgIndex(int), imgScore(float)>
	imgNamesVec : <imgIndex(int), imgNames(string)>
	imgToIndex : pair<imgNames(string), imgIndex(int)>

output:
	save into saveFileVec
*/
void consOneAesthIndexFile_random(float scoreGap, vector<string>saveFileVec, vector<int>numBoundVec, vector<int> biliVec,  
						   vector<float>&scoreVec, vector<string>&imgNamesVec, vector<string>&namesServerVec, map<string,int>& imgToIndex)
{
	string trainTest = "train";
	// fixed variables
	const string trainPath = "F:\\Lab\\Category_Ranking_AVA_14\\category_based_"+trainTest+"\\"; // path to be changed
	const string retrTblPath = "E:\\Lab\\Category_Ranking_AVA_14\\DCNN_test\\category_based_"+trainTest+"\\";
	int numRow = 100;
	int imgNumTotal = getImgNum(trainPath);

	vector<ofstream> fsVec(saveFileVec.size()); // open saveFileVec
	for (int i = 0; i < fsVec.size(); i++)
	{
		fsVec[i].open(saveFileVec[i]);
		if ( !fsVec[i].is_open() )
		{
			cout<<"cannot open file: "<<saveFileVec[i]<<endl;
			return;
		}
	}


	Directory dir;
	vector<string> cateNames = dir.GetListFolders( trainPath, "*", false );

//	const string aesthBase = "F:\\Lab\\Category_Ranking_AVA_14\\category_based_train\\";
//	const string aesthBase = "/home/LvHao/DeepAesthRankingNet/Dataset_AVA_14/category_based_"+trainTest+"/";
	const string aesthBase = "/home/lvhao/DeepAesthRankingNet/Dataset_AVA_14/category_based_"+trainTest+"/";
	string aesthIndex = "";

	int featIndexQ = -1;
	int qid = 0;
	for (int i = 0; i < cateNames.size(); i++) // traverse for each query image
	{
		string currDir = trainPath + cateNames[i] + "/";
		vector<string> imgNames = dir.GetListFiles(currDir, "*.jpg", false);
		
		for (int j = 0; j < imgNames.size(); j++)
		{
			featIndexQ++;
			qid ++;
			cout<<cateNames[i] + "/" + imgNames[j] + "	\t"<<scoreGap<<" \t"<<featIndexQ<<endl;

			vector<int> aRow(numBoundVec.back()+10, -1);
			vector<vector<int>> fIndexVec(fsVec.size(), aRow); // the vector to save the featIndex smaller than query image
			vector<int> indexB(fsVec.size(), 0); //index  of image that with bigger score
			vector<int> indexS(fsVec.size(), 0); //index  of image that with smaller score
			vector<int> currNum(biliVec.size(), 0);

			ifstream fsNei(retrTblPath + cateNames[i] + "/" + imgNames[j] + ".txt");// open neighbor name table
			if (!fsNei.is_open())
			{
				cout<<"cannot open file: "<<retrTblPath + cateNames[i] + "/" + imgNames[j]<<endl;
				return;
			}

			float scoreQ = scoreVec[featIndexQ];
			string tmp1 = "";
			int times = 0;
			bool flagAll = false;// all file are full or not flag

			while(!fsNei.eof() && times < imgNumTotal) // find all the bigger images
			{
				fsNei>>tmp1;
				times++;

				int featIndexN = -1;
				float scoreN = -1;
				featIndexN = imgToIndex[tmp1];
				scoreN = scoreVec[featIndexN]; // the score and featIndex of a neighbor

				if (scoreN < scoreQ - scoreGap) // smaller neighbors
				{
					for (int k = 0; k < fsVec.size(); k++)
					{
						if (indexB[k] + indexS[k] >= numBoundVec[k])
						{
							if ( k==fsVec.size()-1 ) flagAll = true; // all saveFile is full
							continue;
						}

						if (currNum[k] >= numBoundVec[k])
						{
							if ( k==fsVec.size()-1 ) flagAll = true; // all saveFile is full
							continue;
						}

						currNum[k]++;
						if (currNum[k] % biliVec[k] == 0)
						{
							fIndexVec[k][indexS[k]] = featIndexN;
							indexS[k]++;
						}
					}
				}
				else if (scoreN > scoreQ + scoreGap){
					for (int k = 0; k < fsVec.size(); k++)
					{
						if (indexB[k] + indexS[k] >= numBoundVec[k])
						{
							if ( k==fsVec.size()-1 ) flagAll = true; // all saveFile is full
							continue;
						}

						if (currNum[k] >= numBoundVec[k])
						{
							if ( k==fsVec.size()-1 ) flagAll = true; // all saveFile is full
							continue;
						}

						currNum[k]++;
						if (currNum[k] % biliVec[k] == 0)
						{
							writeAesthLine(fsVec[k], featIndexN, featIndexQ, aesthBase, namesServerVec);
							indexB[k]++;
						}
					}
				}

				if (flagAll) break;
			}
			fsNei.close();

			qid ++;
			for (int k = 0; k < fsVec.size(); k++) // write all the smaller images
			{
				for (int m = 0; m < indexS[k]; m++)
				{

					writeAesthLine(fsVec[k], featIndexQ, fIndexVec[k][m], aesthBase, namesServerVec);
				}
			}
		}
	}

	for (int i = 0; i < fsVec.size(); i++)
	{
		fsVec[i].close();
	}
}

void writeAesthLine(ofstream& fs, int gIndex, int bIndex, string base, vector<string>&imgNamesVec)
{
	fs<<base+imgNamesVec[gIndex] + " " + base+imgNamesVec[bIndex]<<endl;
}
