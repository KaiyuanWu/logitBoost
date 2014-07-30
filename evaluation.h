/* 
 * File:   evaluation.h
 * Author: kaiwu
 *
 * Created on July 25, 2014, 11:05 AM
 */

#ifndef EVALUATION_H
#define	EVALUATION_H
#include <iostream>
#include <string>
#include <fstream>
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "LossFunction.h"
#include "dataManager.h"
#include "linearSearch.h"
#include "directionFunction.h"

class evaluation {
public:
    evaluation(char* fTrain, char* fTest, char* fOut, int nTrainEvents,int nTestEvents,int nClass,int nVariables,
            LossFunction::_LOSSTYPE lossType, LossFunction::_PHIFUNCTION phiFunction,double shrinkage=1.,int nLeaves=8,int minimumNodeSize=1,int nMaxIteration=1000);
    virtual ~evaluation();
    
    //initialization procedure
    //read data file
    void init();
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
    LossFunction::_LOSSTYPE _lossType;
    LossFunction::_PHIFUNCTION _phiFunction;
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
    ofstream* _outf;
};

#endif	/* EVALUATION_H */

