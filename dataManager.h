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
//dataManager stores training data & testing data, the order information of the samples, the classifier values of the sample and the discending directions and the gradients&hessian of the samples.
class dataManager {
public:
    dataManager(int nDimension,int nClass, directionFunction::_TREE_TYPE_ treeType, int nTrainEvent,int nTestEvent=0);
    //adding training event, after adding training events, and initialize the logist p/F
    void   addEvent(float * event,int iclass)   ;
    //adding validating event, after adding validating events, and initialize the logist p/F
    void   addValidateEvent(float * event,int iclass)   ;
    //do some check after finishing adding events
    void   finishAddingEvent();
    
    //allocate data space
    void   allocateDataSpace() ;
    //increment the logist p/F
    void   increment(float  shrinkage,int iRound=0) ;
    virtual ~dataManager() ;

    //space for training data
    //training features
    float * _trainX;
    //training labels
    int* _trainClass;
    //discending directions of training samples
    float * _trainDescendingDirection;    
    //classifiers values of the training samples
    float * _trainF;
    //gradient and hessian values of the training samples
    int _nG;
    float * _lossGradient;
    float * _lossHessian;
    //loss value of each sample
    float * _loss;
    //total training loss
    float   _trainLoss;
    int  _trainCurrentEvent;
    float    _trainAccuracy             ;
    int      _trainCorrectClassification;
    //orders information of the training samples
    //original index of the sorted array
    int** _dataIndex;
    //reverse index table
    int** _dataReverseIndex;
    //index of all original data
    int** _dataIndex0;
    //reverse index table;
    int** _dataReverseIndex0;
    //temporary array
    float * _projectedX;
    int* _dataIndexTemp;

    
    //space for testing data
    //features of test samples
    float * _testX;
    //labels of test samples
    int* _testClass;
    //discending directions of test samples
    float * _testDescendingDirection;   
    //classifier values of test samples
    float * _testF;
    
    int _testCurrentEvent;
    float    _testAccuracy;
    float    _testLoss;
    int      _testCorrectClassification;

    int _nClass;
    int _nDimension;
    int _nTrainEvents;
    int _nTestEvents;
    
    void saveF();
private:
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

    directionFunction::_TREE_TYPE_ _treeType;
    LossFunction* _lossFunction;
    //costSensitiveLossFunction* _lossFunction;
    
    //recursively sort the projected data
    void sort(int low,int high,int iDimension,bool atInit=false);
    void swap(int i,int j,int iDimension,bool atInit=false);
};
#endif	/* DATAMANAGER_H */