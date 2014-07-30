/* 
 * File:   dataManager.cpp
 * Author: kaiwu
 * 
 * Created on December 4, 2013, 9:40 AM
 */

#include <string.h>
#include "dataManager.h"

dataManager::dataManager(int nDimension,int nClass,LossFunction::_LOSSTYPE lossType, LossFunction::_PHIFUNCTION phiFunction, int nTrainEvent,int nTestEvent){
    _nTrainEvents=nTrainEvent;
    _nTestEvents=nTestEvent;
    _nDimension=nDimension;
    _nClass=nClass;
    _trainAccuracy=0.;
    _testAccuracy=0.;
    _trainLoss=0.;
    _testLoss=0.;
    allocateDataSpace();
    
    _lossFunction=new LossFunction(lossType,phiFunction,_nDimension,_nClass);
    //_lossFunction=new costSensitiveLossFunction(phiFunction,_nDimension,_nClass);
}
void dataManager::allocateDataSpace() {
    if (_nTrainEvents > 0) {
        _trainDescendingDirection=new double[_nTrainEvents*_nClass];
        _trainX=new double[_nTrainEvents*_nDimension];
        _trainF=new double[_nTrainEvents*_nClass];
        _loss=new double[_nTrainEvents];
        _lossGradient=new double[_nTrainEvents*_nClass];
        _lossHessian=new double[_nTrainEvents*_nClass];
        _trainClass=new int[_nTrainEvents];
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
    for(int iDimension=0;iDimension<_nDimension;iDimension++){
        _trainX[_trainCurrentEvent*_nDimension+iDimension]=event[iDimension];
    }
    for(int iClass=0;iClass<_nClass;iClass++) {
        //if (_nClass != 2) {
            _trainF[_trainCurrentEvent * _nClass + iClass] = 0.;
        //   continue;
        //} 
        
//        if(iClass==0)
//            _trainF[_trainCurrentEvent*_nClass+iClass]=1;
//        else
//            _trainF[_trainCurrentEvent*_nClass+iClass]=-1;
    }
    _trainClass[_trainCurrentEvent]=iclass;
    _trainCurrentEvent++;
}
void dataManager::addValidateEvent(double* event,int iclass){
    for(int iDimension=0;iDimension<_nDimension;iDimension++)
        _testX[_testCurrentEvent*_nDimension+iDimension]=event[iDimension];
    for(int iClass=0;iClass<_nClass;iClass++) {
        //if (_nClass != 2) {
            _testF[_testCurrentEvent * _nClass + iClass] = 0.;
//            continue;
//        }
//        if(iClass==0){
//            _testF[_testCurrentEvent*_nClass+iClass]=1;
//        }
//        else{
//            _testF[_testCurrentEvent*_nClass+iClass]=-1.;
//        }
    }
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
    double g, h;
    _trainLoss=0.;
    _testLoss=0.;
    for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
        for (int iClass = 0; iClass < _nClass; iClass++) {
            _loss[iEvent]=_lossFunction->loss(_trainF + iEvent*_nClass, _trainClass[iEvent], iClass, g, h);
            _lossGradient[iEvent * _nClass + iClass] = g;
            _lossHessian[iEvent * _nClass + iClass] = h;
            //cout<<"["<<_loss[iEvent]<<", "<<g<<", "<<h<<"], ";
        }
        //cout<<endl;
        _trainLoss+=_loss[iEvent];
    }
    //exit(0);
    for (int iEvent = 0; iEvent < _nTestEvents; iEvent++)
        _testLoss+=_lossFunction->loss(_testF + iEvent*_nClass, _testClass[iEvent], 0, g, h);
    
}
void  dataManager::increment(double shrinkage,directionFunction* df,int iRound) {
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
       _maxF=-1.e300;
       _maxI=-1;
       for(int iClass=0;iClass<_nClass;iClass++){
           _trainF[iEvent*_nClass+iClass]+=shrinkage*_trainDescendingDirection[iEvent*_nClass+iClass];
           if(_maxF<_trainF[iEvent*_nClass+iClass]){
               _maxF=_trainF[iEvent*_nClass+iClass];
               _maxI=iClass;
           }
       }
       for (int iClass = 0; iClass < _nClass; iClass++) {
            if (_trainClass[iEvent] == iClass) {
                if (_maxI == iClass){
                    _correctNew[iClass]++;
                    _trainCorrectClassification++;
                }
                else
                    _wrongNew[iClass]++;
            }
        }
        double g, h;
        for (int iClass = 0; iClass < _nClass; iClass++) {
            _loss[iEvent] = _lossFunction->loss(_trainF + iEvent*_nClass, _trainClass[iEvent], iClass, g, h);
            _lossGradient[iEvent * _nClass + iClass] = g;
            if(h<1.0e-20)
                h=1.0e-20;
            _lossHessian[iEvent * _nClass + iClass] = h;
        }
        _trainLoss+=_loss[iEvent];
        //_trainLoss+=_lossFunction->_costMatrix[_trainClass[iEvent]*_nClass+_maxI];
    }
    //_trainLoss/=_nTrainEvents;
    for (int iEvent = 0; iEvent < _nTestEvents; iEvent++) {
        _maxF = -1.e300;
        _maxI = -1;
        for (int iClass = 0; iClass < _nClass; iClass++){
            _testF[iEvent * _nClass + iClass] += shrinkage * _testDescendingDirection[iEvent * _nClass + iClass];
            if(_maxF<_testF[iEvent * _nClass + iClass]){
                _maxF=_testF[iEvent * _nClass + iClass];
                _maxI=iClass;
            }
        }
        for (int iClass = 0; iClass < _nClass; iClass++) {
            if (_testClass[iEvent] == iClass) {
                if (_maxI == iClass){
                    _correctNewTest[iClass]++;
                    _testCorrectClassification++;
                }
                else
                    _wrongNewTest[iClass]++;
            }
        }
        double g,h;
        _testLoss+=_lossFunction->loss(_testF + iEvent*_nClass, _testClass[iEvent], 0, g, h);
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