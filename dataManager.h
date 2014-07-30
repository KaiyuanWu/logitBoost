/* 
 * File:   dataManager.h
 * Author: kaiwu
 *
 * Created on December 4, 2013, 9:40 AM
 */

#ifndef DATAMANAGER_H
#define	DATAMANAGER_H
#include <iostream>
#include <vector>
#include <string>
#include "stdlib.h"
#include "math.h"
#include "LossFunction.h"
#include "directionFunction.h"

using namespace std;
//dataManager class will be served as a pure memory class
//it will not write/read any data into disk
//The reason for this kind of design is to avoid difficulties when using openMP
//From here, we will assume all dimension are continuous double type
//Although we have all data information in the upper level class, this class seems to have a duplicate copy of data
//But when implement the openMP, it will have some advantages
class dataManager {
public:
    dataManager(int nDimension,int nClass, LossFunction::_LOSSTYPE lossType, LossFunction::_PHIFUNCTION phiFunction, int nTrainEvent,int nTestEvent=0);
    //adding training event, after adding training events, and initialize the logist p/F
    void   addEvent(double* event,int iclass)   ;
    //adding validating event, after adding validating events, and initialize the logist p/F
    void   addValidateEvent(double* event,int iclass)   ;
    //do some check after finishing adding events
    void   finishAddingEvent();
    
    //allocate data space
    void   allocateDataSpace() ;
    //increment the logist p/F
    void   increment(double shrinkage,directionFunction* df,int iRound=0) ;
    virtual ~dataManager() ;

    //space for training data
    double* _trainX;
    int* _trainClass;    
    double* _trainDescendingDirection;    
    double* _trainF;
    
    double* _lossGradient;
    double* _lossHessian;
    double* _loss;
    double  _trainLoss;
    int  _trainCurrentEvent;
    double   _trainAccuracy             ;
    int      _trainCorrectClassification;

    //space for testing data
    double* _testX;
    int* _testClass;
    double* _testDescendingDirection;   
    double* _testF;
    
    int _testCurrentEvent;
    double   _testAccuracy;
    double   _testLoss;
    int      _testCorrectClassification;

    int _nClass;
    int _nDimension;
    int _nTrainEvents;
    int _nTestEvents;
    
public:
    //some variable for calculating the accuracy
    double _maxF;
    int    _maxI;
    //correct/wrongly classified number of training events
    int* _correctOld;
    int* _correctNew;
    int* _wrongOld;
    int* _wrongNew;
    //correct/wrongly classified number of testing events
    int* _correctOldTest;
    int* _correctNewTest;
    int* _wrongOldTest;
    int* _wrongNewTest;
    

    LossFunction* _lossFunction;
    //costSensitiveLossFunction* _lossFunction;

};
#endif	/* DATAMANAGER_H */