/* 
 * File:   dataManager.cpp
 * Author: kaiwu
 * 
 * Created on December 4, 2013, 9:40 AM
 */

#include <string.h>
#include "dataManager.h"

dataManager::dataManager(int nDimension,int nClass, directionFunction::_TREE_TYPE_ treeType, int nTrainEvent,int nTestEvent){
    _nTrainEvents=nTrainEvent;
    _nTestEvents=nTestEvent;
    _nDimension=nDimension;
    _nClass=nClass;
    _trainAccuracy=0.;
    _testAccuracy=0.;
    _trainLoss=0.;
    _testLoss=0.;
    _treeType=treeType;
    
    allocateDataSpace();
    
    _lossFunction=new LossFunction(_treeType,_nDimension,_nClass);
    //_lossFunction=new costSensitiveLossFunction(phiFunction,_nDimension,_nClass);
}
void dataManager::allocateDataSpace() {
    if (_nTrainEvents > 0) {
        _trainDescendingDirection=new double[_nTrainEvents*_nClass];
        _trainX=new double[_nTrainEvents*_nDimension];
        _trainF=new double[_nTrainEvents*_nClass];
        _loss=new double[_nTrainEvents];
        _trainClass=new int[_nTrainEvents];
        switch(_treeType){
            case directionFunction::_LOGITBOOST_:
            case directionFunction::_MART_:
            case directionFunction::_SLOGITBOOST_:
                _nG=_nClass;
                break;
            case directionFunction::_ABC_LOGITBOOST_:
            case directionFunction::_AOSO_LOGITBOOST_:
                _nG=_nClass*_nClass;
                break;
                
        }
        _lossGradient=new double[_nTrainEvents*_nG];
        _lossHessian=new double[_nTrainEvents*_nG];

        _projectedX = new double[_nTrainEvents];
        _dataIndex = new int*[_nDimension];
        _dataIndex0 = new int*[_nDimension];
        _dataReverseIndex = new int*[_nDimension];
        _dataReverseIndex0 = new int*[_nDimension];
        _dataIndexTemp = new int[_nTrainEvents];
        
        for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
            _dataIndex[iDimension] = new int[_nTrainEvents];
            _dataIndex0[iDimension] = new int[_nTrainEvents];
            _dataReverseIndex[iDimension] = new int[_nTrainEvents];
            _dataReverseIndex0[iDimension] = new int[_nTrainEvents];
        }
    }
    
    if (_nTestEvents > 0) {
        _testDescendingDirection=new double[_nTestEvents*_nClass];
        _testX=new double[_nTestEvents*_nDimension];
        _testF=new double[_nTestEvents*_nClass];
        _testClass=new int[_nTestEvents];
    }
    
    _correctOld=new int[_nClass];
    _correctNew=new int[_nClass];
    _wrongOld=new int[_nClass];
    _wrongNew=new int[_nClass];
    
    _correctOldTest=new int[_nClass];
    _correctNewTest=new int[_nClass];
    _wrongOldTest=new int[_nClass];
    _wrongNewTest=new int[_nClass];
    
    for(int iClass=0;iClass<_nClass;iClass++){
        _correctOld[iClass]=0;
        _correctNew[iClass]=0;
        _wrongOld[iClass]=0;
        _wrongNew[iClass]=0;
        
        _correctOldTest[iClass]=0;
        _correctNewTest[iClass]=0;
        _wrongOldTest[iClass]=0;
        _wrongNewTest[iClass]=0;
    }
    _trainCurrentEvent=0;
    _testCurrentEvent=0;
}

void dataManager::addEvent(double* event,int iclass){
    for(int iDimension=0;iDimension<_nDimension;iDimension++)
        _trainX[_trainCurrentEvent*_nDimension+iDimension]=event[iDimension];
    for(int iClass=0;iClass<_nClass;iClass++)
        _trainF[_trainCurrentEvent * _nClass + iClass] = 0.;
    _trainClass[_trainCurrentEvent]=iclass;
    _trainCurrentEvent++;
}
void dataManager::addValidateEvent(double* event,int iclass){
    for(int iDimension=0;iDimension<_nDimension;iDimension++)
        _testX[_testCurrentEvent*_nDimension+iDimension]=event[iDimension];
    for(int iClass=0;iClass<_nClass;iClass++)
        _testF[_testCurrentEvent * _nClass + iClass] = 0.;
    _testClass[_testCurrentEvent]=iclass;
    _testCurrentEvent++;
}

dataManager::~dataManager() {
    if (_nTrainEvents > 0) {
        delete[] _trainDescendingDirection;
        delete[] _trainX;
        delete[] _trainF;
        delete[] _trainClass;
        
        delete[] _loss;
        delete[] _lossGradient;
        delete[] _lossHessian;

        delete[] _projectedX;
        delete[] _dataIndexTemp;

        for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
            delete[] _dataIndex[iDimension];
            delete[] _dataIndex0[iDimension];
            delete[] _dataReverseIndex[iDimension];
            delete[] _dataReverseIndex0[iDimension];
        }
        delete[] _dataIndex;
        delete[] _dataIndex0;
        delete[] _dataReverseIndex;
        delete[] _dataReverseIndex0;
    }
    if (_nTestEvents > 0) {
        delete[] _testDescendingDirection;
        delete[] _testX;
        delete[] _testF;
        delete[] _testClass;
    }
    delete[] _correctOld;
    delete[] _correctNew;
    delete[] _wrongOld;
    delete[] _wrongNew;

    delete[] _correctOldTest;
    delete[] _correctNewTest;
    delete[] _wrongOldTest;
    delete[] _wrongNewTest;
};
void dataManager::finishAddingEvent(){
    if(_trainCurrentEvent<_nTrainEvents){
        cout<<"You have added "<<_trainCurrentEvent<<" training events, which is less than "<<_nTrainEvents<<endl;
        _nTrainEvents=_trainCurrentEvent;
    }
    if(_testCurrentEvent<_nTestEvents){
        cout<<"You have added "<<_testCurrentEvent<<" validating events, which is less than "<<_nTestEvents<<endl;
        _nTestEvents=_testCurrentEvent;
    }
    for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
        for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
            _projectedX[iEvent] = _trainX[iEvent * _nDimension + iDimension];
            _dataIndex0[iDimension][iEvent] = iEvent;
        }
        sort(0, _nTrainEvents - 1, iDimension, true);
        //store reverse data index table
        for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++)
            _dataReverseIndex0[iDimension][_dataIndex0[iDimension][iEvent]] = iEvent;
    }
//    //print sort result
//    for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
//        for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
//            cout << _trainX[_dataIndex0[iDimension][iEvent] * _nDimension + iDimension] << ", ";
//        }
//        cout << endl;
//    }
    
    double g, h;
    _trainLoss=0.;
    _testLoss=0.;
    for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
        for (int iG = 0; iG < _nG; iG++) {
            _loss[iEvent]=_lossFunction->loss(_trainF + iEvent*_nClass, _trainClass[iEvent], g, h,iG);
            _lossGradient[iEvent * _nG + iG] = g;
            _lossHessian[iEvent * _nG + iG] = h;
        }
        _trainLoss+=_loss[iEvent];
    }
    //exit(0);
    for (int iEvent = 0; iEvent < _nTestEvents; iEvent++)
        _testLoss+=_lossFunction->loss(_testF + iEvent*_nClass, _testClass[iEvent], g, h,1);
    
}
void  dataManager::increment(double shrinkage,int iRound) {
    //test the direction
    for(int ix=0;ix<_nTrainEvents;ix++){
        for(int ic=0;ic<_nClass;ic++)
            cout<<_trainDescendingDirection[ix*_nClass+ic]<<", ";
        cout<<endl;
    }
    exit(0);
    
    const int OUTPUT_INTERVAL = 1000000;
    for (int iClass = 0; iClass < _nClass; iClass++) {
        _correctNew[iClass] = 0;
        _wrongNew[iClass] = 0;
        _correctNewTest[iClass] = 0;
        _wrongNewTest[iClass] = 0;
    }
    _trainCorrectClassification=0.;
    _testCorrectClassification=0.;
    _trainLoss=0.;
    _testLoss=0.;
    for(int iEvent=0;iEvent<_nTrainEvents;iEvent++){
       double maxF=-1.e300;
       int maxI=-1;
       for(int iClass=0;iClass<_nClass;iClass++){
           _trainF[iEvent*_nClass+iClass]+=shrinkage*_trainDescendingDirection[iEvent*_nClass+iClass];
           if(maxF<_trainF[iEvent*_nClass+iClass]){
               maxF=_trainF[iEvent*_nClass+iClass];
               maxI=iClass;
           }
       }
       for (int iClass = 0; iClass < _nClass; iClass++) {
            if (_trainClass[iEvent] == iClass) {
                if (maxI == iClass){
                    _correctNew[iClass]++;
                    _trainCorrectClassification++;
                }
                else
                    _wrongNew[iClass]++;
            }
        }
       //update the gradients and hessian informations
        double g, h;
        for (int iG = 0; iG < _nG; iG++) {
            _loss[iEvent] = _lossFunction->loss(_trainF + iEvent*_nClass, _trainClass[iEvent],  g, h,iG);
            _lossGradient[iEvent * _nG + iG] = g;
            _lossHessian[iEvent * _nG + iG] = h;
        }
        _trainLoss+=_loss[iEvent];
        //_trainLoss+=_lossFunction->_costMatrix[_trainClass[iEvent]*_nClass+_maxI];
    }
    //_trainLoss/=_nTrainEvents;
    for (int iEvent = 0; iEvent < _nTestEvents; iEvent++) {
        double maxF = -1.e300;
        double maxI = -1;
        for (int iClass = 0; iClass < _nClass; iClass++){
            _testF[iEvent * _nClass + iClass] += shrinkage * _testDescendingDirection[iEvent * _nClass + iClass];
            if(maxF<_testF[iEvent * _nClass + iClass]){
                maxF=_testF[iEvent * _nClass + iClass];
                maxI=iClass;
            }
        }
        for (int iClass = 0; iClass < _nClass; iClass++) {
            if (_testClass[iEvent] == iClass) {
                if (maxI == iClass){
                    _correctNewTest[iClass]++;
                    _testCorrectClassification++;
                }
                else
                    _wrongNewTest[iClass]++;
            }
        }
        double g,h;
        _testLoss+=_lossFunction->loss(_testF + iEvent*_nClass, _testClass[iEvent], g, h,0);
        //_testLoss+=_lossFunction->_costMatrix[_testClass[iEvent]*_nClass+_maxI];
    }
    //_testLoss/=_nTestEvents;
    if(_nTrainEvents>0)
        _trainAccuracy=double(_trainCorrectClassification)/_nTrainEvents;
    else
        _trainAccuracy=0.;
    if(_nTestEvents>0)
        _testAccuracy=double(_testCorrectClassification)/_nTestEvents;
    else
        _nTestEvents=0.;
    if (iRound % OUTPUT_INTERVAL == OUTPUT_INTERVAL-1) {
        double totalOld=0.;
        double totalNew=0.;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        for(int iClass = 0; iClass < _nClass; iClass++){
            totalOld+=_correctOld[iClass];
            totalNew+=_correctNew[iClass];
        }
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~ Round= "<<iRound<<" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "Training: " <<totalOld/_nTrainEvents<<"-->" <<totalNew/_nTrainEvents<<" Correct "<<totalNew<<" Train Loss "<<_trainLoss<<", Test Loss "<<_testLoss<<endl;
        for (int iClass = 0; iClass < _nClass; iClass++) {
            cout << "Class: " << iClass << " [+" << _correctOld[iClass] << ", -" << _wrongOld[iClass] << "]--->[+" << _correctNew[iClass] << ",-" << _wrongNew[iClass] << "] total= " << _correctNew[iClass] + _wrongNew[iClass] << endl;
        }
        totalOld=0.;
        totalNew=0.;
        for(int iClass = 0; iClass < _nClass; iClass++){
            totalOld+=_correctOldTest[iClass];
            totalNew+=_correctNewTest[iClass];
        }
        cout << "Testing: " <<totalOld/_nTestEvents<<"-->" <<totalNew/_nTestEvents<<" Correct "<<totalNew<<endl;
        for (int iClass = 0; iClass < _nClass; iClass++) {
            cout << "Class: " << iClass << " [+" << _correctOldTest[iClass] << ", -" << _wrongOldTest[iClass] << "]--->[+" << _correctNewTest[iClass] << ",-" << _wrongNewTest[iClass] << "] total= " << _correctNewTest[iClass] + _wrongNewTest[iClass] << endl;
        }
        for (int iClass = 0; iClass < _nClass; iClass++) {
            _correctOld[iClass] = _correctNew[iClass];
            _wrongOld[iClass] = _wrongNew[iClass];
            _correctOldTest[iClass] = _correctNewTest[iClass];
            _wrongOldTest[iClass] = _wrongNewTest[iClass];
        }
    }
}
void dataManager::sort(int low, int high, int iDimension, bool atInit) {
    if (low >= high)
        return;
    //copy the low,high values
    int _low = low;
    int _high = high;
    int currentPoint;
    int middlePoint = (_low + _high) / 2;
    while (_low < _high) {
        swap((_low + _high) / 2, _low, iDimension, atInit);
        currentPoint = _low;
        for (int i = _low + 1; i <= _high; i++) {
            if (_projectedX[i] < _projectedX[_low]) {
                currentPoint++;
                swap(currentPoint, i, iDimension, atInit);
            }
        }
        swap(_low, currentPoint, iDimension, atInit);
        if (currentPoint <= middlePoint) _low = currentPoint + 1;
        if (currentPoint >= middlePoint) _high = currentPoint - 1;
    }
    sort(low, middlePoint - 1, iDimension, atInit);
    sort(middlePoint + 1, high, iDimension, atInit);
}


void dataManager::swap(int i, int j, int iDimension, bool atInit) {
    if (i == j)
        return;
    double temp = _projectedX[i];
    _projectedX[i] = _projectedX[j];
    _projectedX[j] = temp;
    if (atInit) {
        int tempIndex = _dataIndex0[iDimension][i];
        _dataIndex0[iDimension][i] = _dataIndex0[iDimension][j];
        _dataIndex0[iDimension][j] = tempIndex;
    } else {
        int tempIndex = _dataIndex[iDimension][i];
        _dataIndex[iDimension][i] = _dataIndex[iDimension][j];
        _dataIndex[iDimension][j] = tempIndex;
    }
}