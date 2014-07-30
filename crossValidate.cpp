/* 
 * File:   crossValidate.cpp
 * Author: kaiwu
 * 
 * Created on January 21, 2014, 5:21 PM
 */
#include "crossValidate.h"
#include "string.h"

crossValidate::crossValidate(int jobID,int nFold, LossFunction::_LOSSTYPE lossType, LossFunction::_PHIFUNCTION phiFunction,double shrinkage,int nLeaves,int minimumNodeSize,int nMaxIteration){
    _nFold=nFold;
    _jobID=jobID;
    _shrinkage=shrinkage;
    _nLeaves=nLeaves;
    if(minimumNodeSize<1)
        minimumNodeSize=1;
    _minimumNodeSize=minimumNodeSize;
    _lossType=lossType;
    _phiFunction=phiFunction;
    _nMaxIteration=nMaxIteration;
    //print job info
    cout<<"jobID=          "<<jobID<<endl;
    cout<<"nFold=          "<<nFold<<endl;
    cout<<"shrinkage=      "<<shrinkage<<endl;
    cout<<"nLeaves=        "<<nLeaves<<endl;
    cout<<"minimumNodeSize="<<minimumNodeSize<<endl;
    cout<<"nMaxIteration=  "<<nMaxIteration<<endl;
    switch(lossType){
        case LossFunction::_ONE_VS_ALL:
            cout<<"Loss Type _ONE_VS_ALL"<<endl;
            break;
        case LossFunction::_ONE_VS_ONE:
            cout<<"Loss Type _ONE_VS_ONE"<<endl;
            break;
        case LossFunction::_CONSTRAIN_COMPARE_:
            cout<<"Loss Type _CONSTRAIN_COMPARE_"<<endl;
            break;
        case LossFunction::_COUPLEDLOGISTIC_:
            cout<<"Loss Type _COUPLEDLOGISTIC_"<<endl;
            break;
        default:
            cout<<"This loss type has not been implemented! Please Check!"<<endl;
            return;
            break;
    }
    switch(phiFunction){
        case LossFunction::_EXP_:
            cout<<"Phi Type Exp"<<endl;
            break;
        case LossFunction::_L2_:
            cout<<"Phi Type _L2_"<<endl;
            break;
        case LossFunction::_LIKELIHOOD_:
            cout<<"Phi Type _LIKELIHOOD"<<endl;
            break;
        default:
            cout<<"This phi type has not been implemented! Please Check!"<<endl;
            return;
            break;
    }
    
    _bestAccuracy=0. ;
    _bestIteration=-1;
}
void crossValidate::start(){
    int iIteration;
    //start the training iteration
    for (iIteration = 0; iIteration < _nMaxIteration; iIteration++) {
        //call the weak learner for each fold
        for (int iFold = 0; iFold < _nFold; iFold++) {
            _linearSearchMinimizer[iFold]->minimization(iIteration);
            _accuracyTestArray[iIteration]  += _data[iFold]->_testAccuracy * _data[iFold]->_nTestEvents;
            _accuracyTrainArray[iIteration] += _data[iFold]->_trainAccuracy * _data[iFold]->_nTrainEvents;
            _lossTrainArray[iIteration]+=_data[iFold]->_trainLoss;
            _lossTestArray[iIteration]+=_data[iFold]->_testLoss;
        }
        _accuracyTestArray[iIteration]/=_nEvents;
        if(_accuracyTestArray[iIteration]>_bestAccuracy){
            _bestAccuracy=_accuracyTestArray[iIteration];
            _bestIteration=iIteration;
        }
        _accuracyTrainArray[iIteration]/=_totalNumberOfTrainEvents;
        _lossTestArray[iIteration]/=_nEvents;
        _lossTrainArray[iIteration]/=_totalNumberOfTrainEvents;

        (*_outf)<<_accuracyTrainArray[iIteration]<<"\t"<<_accuracyTestArray[iIteration]<<"\t"<<_lossTrainArray[iIteration]<<"\t"<<_lossTestArray[iIteration]<<endl;

    }
}
//init procedure will allocate the space to store the data
void crossValidate::init(){
    const char* prefix="/afs/cern.ch/work/k/kaiwu/files/datasets/cleanData/";
    const char* outputPrefix="/afs/cern.ch/work/k/kaiwu/files/output/";
    const char* fileNames[]={"BreastTissue.dat","ionosphere.dat","letter.dat","liver.dat,","optdigits.dat",
    "pendigits.dat","pima.dat","satImage.dat","shuttle.dat","tic-tac-toe.dat","iris.dat","wine.dat","glass.dat","dna.dat",
    "vowel.dat","vehicle.dat"};
    const int nClass[]={6,2,26,2,10,10,2,6,7,2,3,3,6,3,11,4};
    const int nVariables[]={9,34,16,6,64,16,8,36,9,9,4,13,9,180,10,18};
    const int nEvents[]={106,351,20000,345,5620,10992,768,6435,58000,958,150,178,214,3186,528,846};
    _fileInName="";
    _fileInName+=prefix;
    _fileInName+=fileNames[_jobID];
    
    _fileOutName="";
    _fileOutName+=outputPrefix;
    char NN[256];
    sprintf(NN,"logitBoostS_nMaxIteration%dlossType%dphiFunction%dshrinkage%fnLeaves%d",_nMaxIteration,_lossType,_phiFunction,_shrinkage,_nLeaves);

    _fileOutName+=NN;
    _fileOutName+=fileNames[_jobID];
    
    _nEvents=nEvents[_jobID];
    _nVariables=nVariables[_jobID];
    _nClass=nClass[_jobID];
    cout<<"Dataset Name: "<<fileNames[_jobID]<<", nEvents= "<<_nEvents<<", nVariables= "<<_nVariables<<", nClass= "<<_nClass<<endl;
    
    _dataX=new double[_nEvents*_nVariables];
    _classX=new int[_nEvents];
    _foldIndex=new int[_nEvents];
    _numberPerFold=new int[_nFold];
    _accuracyTrainArray=new double[_nMaxIteration];
    _accuracyTestArray=new double[_nMaxIteration];
    _lossTestArray=new double[_nMaxIteration];
    _lossTrainArray=new double[_nMaxIteration];
    
    memset(_numberPerFold,0,sizeof(int)*_nFold);
    memset(_accuracyTrainArray,0,sizeof(double)*_nMaxIteration);
    memset(_accuracyTestArray,0,sizeof(double)*_nMaxIteration);
    memset(_lossTestArray,0,sizeof(double)*_nMaxIteration);
    memset(_lossTrainArray,0,sizeof(double)*_nMaxIteration);
    
    ifstream infs(_fileInName.c_str(),ifstream::in);
    _outf=new ofstream(_fileOutName.c_str(),ofstream::out);
    _outf->setf(ios::scientific);
    if(!infs.good()){
        cout<<"Can not open "<<_fileInName<<endl;
        exit(-1);
    }
    for(int iEvent=0;iEvent<_nEvents;iEvent++){
        for(int iDimension=0;iDimension<_nVariables;iDimension++)
            infs>>_dataX[iEvent*_nVariables+iDimension];
        infs>>_classX[iEvent];
        //because the label starts from 1 in the data file
        _classX[iEvent]=_classX[iEvent]-1;
    }
    //print reading data
//    cout<<"*********************************"<<endl;
//    for(int iEvent=0;iEvent<_nEvents;iEvent++){
//        for(int iDimension=0;iDimension<_nVariables;iDimension++)
//            cout<<_dataX[iEvent*_nVariables+iDimension]<<" ";
//        cout<<_classX[iEvent]<<endl;
//    }
//    cout<<"*********************************"<<endl;
    infs.close();
    
    splitData();
    //print fold info
//    cout<<"************************************"<<endl;
//    for(int iFold=0;iFold<_nFold;iFold++)
//        cout<<"Fold "<<iFold<<" : "<<_numberPerFold[iFold]<<endl;
//    for(int iEvent=0;iEvent<_nEvents;iEvent++){
//        cout<<_foldIndex[iEvent]<<" ";
//    }
//    cout<<endl;
//    cout<<"************************************"<<endl;
////    exit(0);
//    for(int iFold=0;iFold<_nFold;iFold++){
//        int* nc=new int[_nClass];
//        for(int iClass=0;iClass<_nClass;iClass++)
//            nc[iClass]=0;
//        for(int iEvent=0;iEvent<_nEvents;iEvent++){
//            if(_foldIndex[iEvent]==iFold)
//                nc[_classX[iEvent]]++;
//        }
//        cout<<"Fold "<<iFold<<" : "<<endl;
//        for(int iClass=0;iClass<_nClass;iClass++)
//            cout<<nc[iClass]<<" ";
//        cout<<endl;
//    }
//    exit(0);
    
    
    _data=new dataManager*[_nFold];
    _linearSearchMinimizer=new linearSearch*[_nFold];
    for(int iFold=0;iFold<_nFold;iFold++) {
        _data[iFold] = new dataManager(_nVariables, _nClass,_lossType,_phiFunction, _nEvents - _numberPerFold[iFold], _numberPerFold[iFold]);
        for (int iEvent = 0; iEvent < _nEvents; iEvent++) {
            if (_foldIndex[iEvent] != iFold)
                _data[iFold]->addEvent(_dataX + iEvent * _nVariables, _classX[iEvent]);
            else
                _data[iFold]->addValidateEvent(_dataX + iEvent*_nVariables, _classX[iEvent]);
        }
        _data[iFold]->finishAddingEvent();
        _linearSearchMinimizer[iFold] = new linearSearch(_data[iFold],_nLeaves,_shrinkage,_minimumNodeSize);
    }
}
//init procedure will allocate the space to store the data
void crossValidate::init2(){
    const char* prefix="/home/kaiwu/Documents/MulticlassLossFunctionComparision/datasets/dataToNewProgram/";
    const char* outputPrefix="/home/kaiwu/Documents/MulticlassLossFunctionComparision/output/";
    const char* fileNames[]={"iris.scale.root","wine.scale.root","glass.scale.root",
                             "vowel.scale.root","vehicle.scale.root","segment.scale.root"};
    const int nClass[]={3,3,6,11,4,7};
    const int nVariables[]={4,13,9,10,18,19};
    const int nEvents[]={150,178,214,528,846,2310};
    _fileInName="";
    _fileInName+=prefix;
    _fileInName+=fileNames[_jobID];
    
    _fileOutName="";
    _fileOutName+=outputPrefix;
    char NN[256];
    sprintf(NN,"logitBoostS_nMaxIteration%dshrinkage%fnLeaves%dminimumNodeSize%d",_nMaxIteration,_shrinkage,_nLeaves,_minimumNodeSize);
    _fileOutName+=NN;
    _fileOutName+=fileNames[_jobID];
    
    _nEvents=nEvents[_jobID];
    _nVariables=nVariables[_jobID];
    _nClass=nClass[_jobID];
    cout<<"Dataset Name: "<<fileNames[_jobID]<<", nEvents= "<<_nEvents<<", nVariables= "<<_nVariables<<", nClass= "<<_nClass<<endl;
    
    _dataX=new double[_nEvents*_nVariables];
    _classX=new int[_nEvents];
    _foldIndex=new int[_nEvents];
    _numberPerFold=new int[_nFold];
    _accuracyTrainArray=new double[_nMaxIteration];
    _accuracyTestArray=new double[_nMaxIteration];
    _lossTestArray=new double[_nMaxIteration];
    _lossTrainArray=new double[_nMaxIteration];
    
    memset(_numberPerFold,0,sizeof(int)*_nFold);
    memset(_accuracyTrainArray,0,sizeof(double)*_nMaxIteration);
    memset(_accuracyTestArray,0,sizeof(double)*_nMaxIteration);
    memset(_lossTestArray,0,sizeof(double)*_nMaxIteration);
    memset(_lossTrainArray,0,sizeof(double)*_nMaxIteration);
    
    ifstream infs(_fileInName.c_str(),ifstream::in);
    _outf=new ofstream(_fileOutName.c_str(),ofstream::out);
    _outf->setf(ios::scientific);
    for(int iEvent=0;iEvent<_nEvents;iEvent++){
        for(int iDimension=0;iDimension<_nVariables;iDimension++)
            infs>>_dataX[iEvent*_nVariables+iDimension];
        infs>>_classX[iEvent];
        _classX[iEvent]=_classX[iEvent]-1;
    }
    infs.close();
    splitData();
    
    _data=new dataManager*[_nFold];
    _linearSearchMinimizer=new linearSearch*[_nFold];
    for(int iFold=0;iFold<_nFold;iFold++) {
        _data[iFold] = new dataManager(_nVariables, _nClass,_lossType,_phiFunction, _nEvents - _numberPerFold[iFold], _numberPerFold[iFold]);
        for (int iEvent = 0; iEvent < _nEvents; iEvent++) {
            if (_foldIndex[iEvent] != iFold)
                _data[iFold]->addEvent(_dataX + iEvent * _nVariables, _classX[iEvent]);
            else
                _data[iFold]->addValidateEvent(_dataX + iEvent*_nVariables, _classX[iEvent]);
        }
        _data[iFold]->finishAddingEvent();
        _linearSearchMinimizer[iFold] = new linearSearch(_data[iFold],_nLeaves,_shrinkage,_minimumNodeSize);
    }
}
void crossValidate::saveResult(){
    cout<<_fileOutName<<endl;
    _outf->close();
}
void crossValidate::splitData(){
    int* classStart=new int[_nClass];
    int* classIndex=new int[_nClass];
    int* classCount=new int[_nClass];
    int* foldIndex=new int[_nFold];
    int* perm=new int[_nEvents];
    for(int i1=0;i1<_nClass;i1++){
        classStart[i1]=0;
        classIndex[i1]=0;
        classCount[i1]=0;
    }
    for(int i1=0;i1<_nFold;i1++){
        foldIndex[i1]=0;
        _numberPerFold[i1]=0;
    }
    //get number of samples for each class
    for(int i1=0;i1<_nEvents;i1++)
        classStart[_classX[i1]]++;
    int tot=0;
    for(int i1=0;i1<_nClass;i1++){
        classCount[i1]=classStart[i1];
        classStart[i1]=tot;
        classIndex[i1]=tot;
        tot+=classCount[i1];
    }
    for(int i1=0;i1<_nEvents;i1++){
        perm[classIndex[_classX[i1]]]=i1;
//        cout<<_classX[i1]<<" "<<classIndex[_classX[i1]]<<" "<<i1<<endl;
        classIndex[_classX[i1]]++;
    }
    //randomly shuffle
    for(int iClass=0;iClass<_nClass;iClass++){
        int iStart=classStart[iClass];
        int iEnd=classStart[iClass]+classCount[iClass];
        for(int i1=iStart;i1<iEnd;i1++){
            int i2 = iStart+rand()%classCount[iClass];
            int temp=perm[i1];
            perm[i1]=perm[i2];
            perm[i2]=temp;
        }
    }
    //allocate fold id
    int iFold=0;
    for(int iClass=0;iClass<_nClass;iClass++){
        int iStart=classStart[iClass];
        int iEnd=classStart[iClass]+classCount[iClass];
        for(int i1=iStart;i1<iEnd;i1++){
            _foldIndex[perm[i1]]=iFold;
            _numberPerFold[iFold]++;
            iFold++;
            iFold=(iFold%_nFold);
        }
    }
    _totalNumberOfTrainEvents=0;
    for(int iFold=0;iFold<_nFold;iFold++)
        _totalNumberOfTrainEvents+=(_nEvents-_numberPerFold[iFold]);
    delete[] classStart;
    delete[] classIndex;
    delete[] classCount;
    delete[] foldIndex;
    delete[] perm;
}
crossValidate::~crossValidate() {
    delete[] _dataX;
    delete[] _classX;
    delete[] _foldIndex;
    delete[] _numberPerFold;
    delete[] _accuracyTrainArray;
    delete[] _accuracyTestArray;
    delete[] _lossTestArray;
    delete[] _lossTrainArray;
}