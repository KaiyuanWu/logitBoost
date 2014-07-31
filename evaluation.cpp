/* 
 * File:   evaluation.cpp
 * Author: kaiwu
 * 
 * Created on July 25, 2014, 11:05 AM
 */

#include "evaluation.h"

evaluation::evaluation(char* fTrain, char* fTest, char* fOut, int nTrainEvents,int nTestEvents,int nClass,int nVariables,
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
void evaluation::init(){
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

    ifstream infTest(_fTest.c_str(),ifstream::in);
    if(!infTest.good()){
        cout<<"Can not open "<<_fTest<<endl;
        exit(-1);
    }
    for(int iEvent=0;iEvent<_nTestEvents;iEvent++){
        for(int iDimension=0;iDimension<_nVariables;iDimension++)
            infTest>>X[iDimension];
        infTest>>label;
        _data->addValidateEvent(X,label);
    }
    infTest.close();
    _data->finishAddingEvent();
    cout<<"Finish Reading Data!"<<endl;
    _outf=new ofstream(_fOut.c_str(),ofstream::out);
    if(!_outf){
        cout<<"Can not open "<<_fOut<<endl;
        exit(-1);
    }
    _outf->setf(ios::scientific);
    _linearSearchMinimizer = new linearSearch(_data,_nLeaves,_shrinkage,_minimumNodeSize);
}
void evaluation::start(){
    int iIteration;
    //start the training iteration
    for (iIteration = 0; iIteration < _nMaxIteration; iIteration++) {
        //call the weak learner for each fold
        _linearSearchMinimizer->minimization(iIteration);
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
void evaluation::saveResult(){
    _outf->close();
}
evaluation::~evaluation() {
    delete _data;
    delete _linearSearchMinimizer;
    delete[] _accuracyTrainArray;
    delete[] _accuracyTestArray;
    delete[] _lossTestArray;
    delete[] _lossTrainArray;
}



