/* 
 * File:   crossValidate.h
 * Author: kaiwu
 *
 * Created on January 21, 2014, 5:21 PM
 */

#include <iostream>
#include <string>
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
    crossValidate(int jobID,int nFold, LossFunction::_LOSSTYPE lossType,double shrinkage=1.,int nLeaves=8,int minimumNodeSize=1,int nMaxIteration=1000);
    virtual ~crossValidate();
    
public:
    string _fileInName;
    string _fileOutName;
    //initialization procedure
    //read data file
    void init();
    //read data file for the experiments of work AOSALogitBoost
    void init2();
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
    int _nEvents;    
private:
    //split samples into different folds
    void splitData();
    
    //_jobID to discriminate the dataset
    int _jobID;
    double _shrinkage;
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
    double* _dataX;
    int*    _classX;
    int     _label;
    
    //index of the folds
    int* _foldIndex;
    int* _numberPerFold;
    //total number of training events
    int  _totalNumberOfTrainEvents;
    dataManager** _data;
    linearSearch** _linearSearchMinimizer;
    
    LossFunction::_LOSSTYPE _lossType;
    ofstream* _outf;
};

#endif	/* CROSSVALIDATE_H */

