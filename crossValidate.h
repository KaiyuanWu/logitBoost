/* 
 * File:   crossValidate.h
 * Author: kaiwu
 *
 * Created on January 21, 2014, 5:21 PM
 */

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "stdio.h"
#include "stdlib.h"
#include "LossFunction.h"
#include "dataManager.h"
#include "linearSearch.h"
#include "directionFunction.h"

using namespace std;
#ifndef CROSSVALIDATE_H
#define	CROSSVALIDATE_H

class crossValidate {
public:
    crossValidate(int jobID,int nFold, directionFunction::_TREE_TYPE_ treeType,float  shrinkage=1.,int nLeaves=8,int minimumNodeSize=1,int nMaxIteration=1000);
    virtual ~crossValidate();
    
public:
    string _fileInName;
    string _fileOutName;
    //initialization procedure
    //read data file
    void init(char* prefix, char* outputPrefix);
    //start the program
    void start();
    void saveResult();
    
    float  _bestAccuracy;
    int    _bestIteration;
    float * _accuracyTestArray;
    float * _accuracyTrainArray;
    float * _lossTestArray;
    float * _lossTrainArray;
    int _nMaxIteration;
    int _nEvents;    
private:
    //split samples into different folds
    void splitData();
    void getDataInformation(char* fileInName,int& nEvent,int& nClass,int& nVariable);
    
    //_jobID to discriminate the dataset
    int _jobID;
    float  _shrinkage;
    int _nLeaves;
    int _minimumNodeSize;
    
    //number of classes
    int _nClass;
    //number of variables of the problem
    int _nVariables;
    //number of folds
    int _nFold;
    
    //all data will be stored into an one dimension array
    //x_ij will be x[i*nDimension+j]
    //c_i  will be c[i]
    float * _dataX;
    int*    _classX;
    int     _label;
    
    //index of the folds
    int* _foldIndex;
    int* _numberPerFold;
    //total number of training events
    int  _totalNumberOfTrainEvents;
    dataManager** _data;
    linearSearch** _linearSearchMinimizer;
    
    directionFunction::_TREE_TYPE_ _treeType;
    ofstream* _outf;
};

#endif	/* CROSSVALIDATE_H */

