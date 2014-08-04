/* 
 * File:   crossValidate.cpp
 * Author: kaiwu
 * 
 * Created on January 21, 2014, 5:21 PM
 */
#include "crossValidate.h"
#include <string>
#include <sstream>
#include <map>

crossValidate::crossValidate(int jobID,int nFold, directionFunction::_TREE_TYPE_ treeType, 
        double shrinkage,int nLeaves,int minimumNodeSize,int nMaxIteration){
    _nFold=nFold;
    _jobID=jobID;
    _shrinkage=shrinkage;
    _nLeaves=nLeaves;
    if(minimumNodeSize<1)
        minimumNodeSize=1;
    _minimumNodeSize=minimumNodeSize;
    _treeType=treeType;
    _nMaxIteration=nMaxIteration;
    //print job info
    cout<<"jobID=          "<<jobID<<endl;
    cout<<"nFold=          "<<nFold<<endl;
    cout<<"shrinkage=      "<<shrinkage<<endl;
    cout<<"nLeaves=        "<<nLeaves<<endl;
    cout<<"minimumNodeSize="<<minimumNodeSize<<endl;
    cout<<"nMaxIteration=  "<<nMaxIteration<<endl;
    switch(_treeType){
        case directionFunction::_ABC_LOGITBOOST_:
            cout<<"direction Type _ABC_LOGITBOOST_"<<endl;
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
            cout<<"direction Type _AOSO_LOGITBOOST_"<<endl;
            break;
        case directionFunction::_LOGITBOOST_:
            cout<<"direction Type _LOGITBOOST_"<<endl;
            break;
        case directionFunction::_MART_:
            cout<<"direction Type _MART_"<<endl;
            break;
        case directionFunction::_SLOGITBOOST_:
            cout<<"direction Type _SLOGITBOOST_"<<endl;
            break;
        default:
            cout<<"This loss type has not been implemented! Please Check!"<<endl;
            exit(-1);
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
void crossValidate::init(char* prefix, char* outputPrefix){
    //const char* prefix="/home/kaiwu/Documents/MulticlassLossFunctionComparision/datasets/dataToNewProgram/";
    //const char* outputPrefix="/home/kaiwu/Documents/MulticlassLossFunctionComparision/output/";
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
    sprintf(NN,"logitBoost_nMaxIteration%dshrinkage%fnLeaves%dminimumNodeSize%d",_nMaxIteration,_shrinkage,_nLeaves,_minimumNodeSize);
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
        _data[iFold] = new dataManager(_nVariables, _nClass,_treeType, _nEvents - _numberPerFold[iFold], _numberPerFold[iFold]);
        for (int iEvent = 0; iEvent < _nEvents; iEvent++) {
            if (_foldIndex[iEvent] != iFold)
                _data[iFold]->addEvent(_dataX + iEvent * _nVariables, _classX[iEvent]);
            else
                _data[iFold]->addValidateEvent(_dataX + iEvent*_nVariables, _classX[iEvent]);
        }
        _data[iFold]->finishAddingEvent();
        _linearSearchMinimizer[iFold] = new linearSearch(_data[iFold],_nLeaves,_shrinkage,_minimumNodeSize,_treeType);
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
void crossValidate::getDataInformation(char* fileInName,int& nEvent,int& nClass,int& nVariable){
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