    /* 
 * File:   train.cpp
 * Author: kaiwu
 * 
 * Created on July 25, 2014, 11:05 AM
 */

#include "train.h"
#include <string>
#include <sstream>
#include <map>

train::train(char* fTrain, char* fTest, char* fOut, int nTrainEvents,int nTestEvents,int nClass,int nVariables,
            directionFunction::_TREE_TYPE_ treeType, double shrinkage,int nLeaves,int minimumNodeSize,int nMaxIteration){
    //dataset parameters
    _fTrain=fTrain;
    _fTest=fTest;
    _fOut=fOut;
    
    _nTrainEvents=nTrainEvents;
    _nTestEvents=nTestEvents;
    _nClass=nClass;
    _nVariables=nVariables;
    
    //decision tree parameters
    _treeType=treeType;
    _shrinkage=shrinkage;
    _nLeaves=nLeaves;
    _minimumNodeSize=minimumNodeSize;
    _nMaxIteration=nMaxIteration;
}
train::train(char* fTrain, 
            directionFunction::_TREE_TYPE_ treeType, double shrinkage,int nLeaves,int minimumNodeSize,int nMaxIteration){
    _fTrain=fTrain;
    char paramPrefix[1024];
    switch(treeType){
        case directionFunction::_ABC_LOGITBOOST_:
            sprintf(paramPrefix,"abcLogit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
            sprintf(paramPrefix,"aosoLogit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_LOGITBOOST_:
            sprintf(paramPrefix,"logit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_MART_:
            sprintf(paramPrefix,"mart_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_SLOGITBOOST_:
            sprintf(paramPrefix,"slogit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        default:
            cout<<"tree type= "<<int(_treeType)<<" has not been implemented!"<<endl;
            exit(-1);
            break;
    }
    _fOut=_fTrain+paramPrefix;
    _fOut=_fOut+".out";
    _fParam=_fTrain+paramPrefix;
    _fParam=_fParam+".model";
    
    _nTestEvents=0;
    getDataInformation(fTrain,_nTrainEvents,_nClass,_nVariables);
    
    //decision tree parameters
    _treeType=treeType;
    _shrinkage=shrinkage;
    _nLeaves=nLeaves;
    _minimumNodeSize=minimumNodeSize;
    _nMaxIteration=nMaxIteration;
}

train::train(char* fTrain, char* fOldOut,
            directionFunction::_TREE_TYPE_ treeType, double shrinkage,int nLeaves,int minimumNodeSize,int nMaxIteration){
    _fTrain=fTrain;
    char paramPrefix[1024];
    switch(treeType){
        case directionFunction::_ABC_LOGITBOOST_:
            sprintf(paramPrefix,"abcLogit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
            sprintf(paramPrefix,"aosoLogit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_LOGITBOOST_:
            sprintf(paramPrefix,"logit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_MART_:
            sprintf(paramPrefix,"mart_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        case directionFunction::_SLOGITBOOST_:
            sprintf(paramPrefix,"slogit_shrinkage%f_nLeave%d_minimumNodeSize%d_nMaxIteration%d",shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
            break;
        default:
            cout<<"tree type= "<<int(_treeType)<<" has not been implemented!"<<endl;
            exit(-1);
            break;
    }
    _fOut=_fTrain+paramPrefix;
    _fOut=_fOut+".out";
    _fParam=_fTrain+paramPrefix;
    _fParam=_fParam+".model";
    
    _fOldOut=fOldOut;
    _fOldOut+=paramPrefix;
    _fOldOut=_fOldOut+".out";
    _fOldParam=fOldOut;
    _fOldParam+=paramPrefix;
    _fOldParam=_fOldParam+".model";
    
    _nTestEvents=0;
    getDataInformation(fTrain,_nTrainEvents,_nClass,_nVariables);
    
    //decision tree parameters
    _treeType=treeType;
    _shrinkage=shrinkage;
    _nLeaves=nLeaves;
    _minimumNodeSize=minimumNodeSize;
    _nMaxIteration=nMaxIteration;
}

void train::init(){
    double* X=new double[_nVariables];
    int     label;
    
    _accuracyTrainArray=new double[_nMaxIteration];
    _accuracyTestArray=new double[_nMaxIteration];
    _lossTestArray=new double[_nMaxIteration];
    _lossTrainArray=new double[_nMaxIteration];
    
    memset(_accuracyTrainArray,0,sizeof(double)*_nMaxIteration);
    memset(_accuracyTestArray,0,sizeof(double)*_nMaxIteration);
    memset(_lossTestArray,0,sizeof(double)*_nMaxIteration);
    memset(_lossTrainArray,0,sizeof(double)*_nMaxIteration);
    
    _data = new dataManager(_nVariables, _nClass,_treeType,_nTrainEvents, _nTestEvents);
    ifstream infTrain(_fTrain.c_str(),ifstream::in);
    if(!infTrain.good()){
        cout<<"Can not open "<<_fTrain<<endl;
        exit(-1);
    }
    for(int iEvent=0;iEvent<_nTrainEvents;iEvent++){
        for(int iDimension=0;iDimension<_nVariables;iDimension++){
            infTrain>>X[iDimension];
        }
        infTrain>>label;
        _data->addEvent(X,label);
    }
    
    infTrain.close();
    if (_nTestEvents > 0) {
        ifstream infTest(_fTest.c_str(), ifstream::in);
        if (!infTest.good()) {
            cout << "Can not open " << _fTest << endl;
            exit(-1);
        }
        for (int iEvent = 0; iEvent < _nTestEvents; iEvent++) {
            for (int iDimension = 0; iDimension < _nVariables; iDimension++)
                infTest >> X[iDimension];
            infTest>>label;
            _data->addValidateEvent(X, label);
        }
        infTest.close();
    }
    _data->finishAddingEvent();
    cout<<"Finish Reading Data!"<<endl;
    _outf=new ofstream(_fOut.c_str(),ofstream::app);
    if(!_outf){
        cout<<"Can not open "<<_fOut<<endl;
        exit(-1);
    }
    _outf->setf(ios::scientific);
    _paramf=new ofstream(_fParam.c_str(),ofstream::app);
    if(!_paramf){
        cout<<"Can not open "<<_fParam<<endl;
        exit(-1);
    }
    if(_fOldParam.size()==0)
        (*_paramf)<<_treeType<<" "<<_nClass<<" "<<_data->_nDimension<<" "<<_nMaxIteration<<" "<<_shrinkage<<endl;
    _linearSearchMinimizer = new linearSearch(_data,_nLeaves,_shrinkage,_minimumNodeSize,_treeType);
}
int train::load(){
    loadModel loader(_fOldParam.c_str(), _fOldOut.c_str(),_fParam.c_str(), _fOut.c_str(), _data);
    loader.rebuild();
    return loader._availableIterations;
}
void train::getDataInformation(char* fileInName,int& nEvent,int& nClass,int& nVariable){
    map<int,int> classMap;
    ifstream fin1(fileInName,ifstream::in);
    //get the number of variables
    int nGuess=16;
    char* line;
    while(true){
        line=new char[nGuess];
        fin1.getline(line,nGuess);
        if(fin1.fail()){
            nGuess*=2;
            delete line;
            fin1.clear();
            fin1.seekg(0);
        }
        else
            break;
    }
    nVariable=0;
    stringstream ss;
    ss.str(line);
    double t;int l;
    ss>>t;
    while(ss.good()){
        nVariable++;
        ss>>t;
    }
    
    fin1.clear();
    fin1.seekg(0);
    nEvent=0;
    for(int iVar=0;iVar<nVariable;iVar++){
        fin1>>t;
    }
    fin1>>l;
    classMap[l]=1;
    while (fin1.good()) {
        nEvent++;
        for (int iVar = 0; iVar < nVariable; iVar++) {
            fin1>>t;
        }
        fin1>>l;
        classMap[l] = 1;
    }
    nClass=classMap.size();
    fin1.close();
    delete[] line;
}
void train::start(){
    int iIteration;
    //start the training iteration
    iIteration=load();
    for (; iIteration < _nMaxIteration; iIteration++) {
        //call the weak learner for each fold
        _linearSearchMinimizer->minimization(iIteration);
        _linearSearchMinimizer->saveDirection(*_paramf);
        _paramf->flush();
//        cout << "Gradient: " << endl;
//        for (int ix = 0; ix < _data->_nTrainEvents; ix++) {
//            cout << "[";
//            for (int ic = 0; ic < 10; ic++) {
//                cout << _data->_trainX[ix * _data->_nDimension + ic] << ", ";
//            }
//            cout << "] ";
//            for (int ic = 0; ic < _data->_nClass; ic++) {
//                cout << _data->_lossGradient[ix * _data->_nClass + ic] << ", ";
//            }
//            cout << endl;
//        }
//        exit(0);
        _accuracyTestArray[iIteration] = _data->_testAccuracy ;
        _accuracyTrainArray[iIteration]= _data->_trainAccuracy;
        _lossTrainArray[iIteration] = _data->_trainLoss/_nTrainEvents;
        _lossTestArray[iIteration] = _data->_testLoss/_nTestEvents;
        
        if(_accuracyTestArray[iIteration]>_bestAccuracy){
            _bestAccuracy=_accuracyTestArray[iIteration];
            _bestIteration=iIteration;
        }
//        if(iIteration%1==0){
//            cout<<"+++++++++++++ Round "<<iIteration<<" +++++++++++++++++"<<endl;
//            cout<<"Train accuracy: "<<_accuracyTrainArray[iIteration]<<" loss "<<_lossTrainArray[iIteration]<<endl;
//            cout<<"Test accuracy: "<<_accuracyTestArray[iIteration]<<" loss "<<_lossTestArray[iIteration]<<" best "<<_bestAccuracy<<" at "<<_bestIteration<<endl;
//        }
        (*_outf)<<_accuracyTrainArray[iIteration]<<"\t"<<_accuracyTestArray[iIteration]<<"\t"<<_lossTrainArray[iIteration]<<"\t"<<_lossTestArray[iIteration]<<endl;
        _outf->flush();
    }
}
void train::saveResult(){
//    _data->saveF();
    if(_outf->is_open())
        _outf->close();
    if(_paramf){
        if(_paramf->is_open())
            _paramf->close();
    }
}
train::~train() {
    delete _data;
    delete _linearSearchMinimizer;
    delete[] _accuracyTrainArray;
    delete[] _accuracyTestArray;
    delete[] _lossTestArray;
    delete[] _lossTrainArray;
}



