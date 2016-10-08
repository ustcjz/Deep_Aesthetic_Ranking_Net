#include "Head.h"

/**
	divide the prediction
*/
void DividePred();


/**
	divide the one prediction into 14 categories
*/
void DividePredOne(string fileName, string saveName, string savePath, string modelsName, vector<int>& boundVec);



/**
	get the prediction with trained caffemodel
*/
void PredCaffeTest();

void PredCaffeTestOne(string extractCmd, string caffeModel, string testPrototxt, 
					  string layerName, string saveFile, string batchNum);

void testCaffe();