/* 
 * File:   train.h
 * Author: kaiwu
 *
 * Created on July 25, 2014, 11:05 AM
 */

#ifndef train_H
#define	train_H
#include <iostream>
#include <string>
#include <fstream>
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "LossFunction.h"
#include "dataManager.h"
#include "linearSearch.h"
#include "loadModel.h"
#include "directionFunction.h"

class train {
public:
    train(char* fTrain, char* fTest, char* fOut, int nTrainEvents,int nTestEvents,int nClass,int nVariables,
            directionFunction::_TREE_TYPE_ treeType, double shrinkage=1.,int nLeaves=8,int minimumNodeSize=1,int nMaxIteration=1000);
    train(char* fTrain, 
            directionFunction::_TREE_TYPE_ treeType, double shrinkage=1.,int nLeaves=8,int minimumNodeSize=1,int nMaxIteration=1000);
    train(char* fTrain, char* fOldFout,
            directionFunction::_TREE_TYPE_ treeType, double shrinkage=1.,int nLeaves=8,int minimumNodeSize=1,int nMaxIteration=1000);
    
    virtual ~train();
    
    //initialization procedure
    //read data file
    void init();
    int load();
    //start the program
    void start();
    void saveResult();
    
    double _bestAccuracy;
    int    _bestIteration;
    double* _accuracyTestArray;
    double* _accuracyTrainArray;
    double* _lossTestArray;
    double* _lossTrainArray;
    int _nMaxIteration;
private:
    //parameters
    string _fTrain;
    string _fTest;
    string _fOut;
    string _fParam;
    string _fOldOut;
    string _fOldParam;
    directionFunction::_TREE_TYPE_  _treeType;
    double _shrinkage;
    int _nLeaves;
    int _minimumNodeSize;
    
    //dataset parameters
    int _nTrainEvents;
    int _nTestEvents;
    int _nClass;
    int _nVariables;
    
    //private variables
    
    dataManager* _data;
    linearSearch* _linearSearchMinimizer;
    void getDataInformation(char* fileInName,int& nEvent,int& nClass,int& nVariable);
    ofstream* _outf;
    ofstream* _paramf;
};

#endif	/* train_H */

