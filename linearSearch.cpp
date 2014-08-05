/* 
 * File:   linearSearch.cpp
 * Author: kaiwu
 * 
 * Created on December 4, 2013, 9:43 AM
 */

#include "linearSearch.h"
#include "treeVectorDirection.h"
#include "treeScalarDirection.h"


linearSearch::linearSearch(dataManager*  data,int nLeaves,double shrinkage,int minimumNodeSize,directionFunction::_TREE_TYPE_ treeType){
    _data=data;
    _nClass=_data->_nClass;
    _nDimension=_data->_nDimension;
    _nTrainEvents=_data->_nTrainEvents;
    _nTestEvents=_data->_nTestEvents;
    
    _treeType=treeType;
    _nLeaves=nLeaves;
    _minimumNodeSize=minimumNodeSize;
    _shrinkage=shrinkage;
    switch(_treeType){
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            _nDirection=_nClass;
            _df=new directionFunction*[_nDirection];
            for(int id=0;id<_nDirection;id++)
                _df[id]=new treeScalarDirection(data,_nLeaves,_minimumNodeSize,_treeType,id);
            break;
        case directionFunction::_ABC_LOGITBOOST_:
            _nDirection=_nClass*_nClass;
            _df=new directionFunction*[_nDirection];
            for(int id=0;id<_nDirection;id++)
                _df[id]=new treeScalarDirection(data,_nLeaves,_minimumNodeSize,_treeType,id);
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            _nDirection=1;
            _df=new directionFunction*[_nDirection];
            _df[0]=new treeVectorDirection(data,_nLeaves,_minimumNodeSize,_treeType);
            //tt=new treeVectorDirection(data,_nLeaves,_minimumNodeSize,_treeType);
            break;
        default:
            cout<<"Has not been implemented!"<<endl;
            exit(-1);
            break;
    }
    //variables for the "FAST abcLogitBoost"
    _g=20;
    _G=20;
    _baseClass=0;
    _F=new double[_nClass];
}
linearSearch::~linearSearch() {
    for(int id=0;id<_nDirection;id++)
        delete _df[id];
    delete[] _df;
    delete _F;
}
double linearSearch::minimization(int iRound){
    double ret=0.;
    buildDirection();
    updateDirection(iRound);
    _data->increment(_shrinkage,  iRound);
    ret = _data->_trainAccuracy;
    return ret;
}
void linearSearch::updateDirection1() {
    for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
        for (int iClass = 0; iClass < _nClass; iClass++) {
            double d ;
            _df[iClass]->eval(_data->_trainX + iEvent * _nDimension,&d);
            _data->_trainDescendingDirection[iEvent * _nClass + iClass] = d;
        }
    }
    for (int iEvent = 0; iEvent < _nTestEvents; iEvent++) {
        for (int iClass = 0; iClass < _nClass; iClass++) {
            double d;
            _df[iClass]->eval(_data->_testX + iEvent * _nDimension,&d);
            _data->_testDescendingDirection[iEvent * _nClass + iClass] = d;
        }
    }
}

void linearSearch::updateDirection2() {
    double maxL = -1;
    //reset baseClass and search the best base class
    if (_g == _G) {
        _baseClass = 0;
        double maxF = -1.e300, sumF, sumExpF, py;
        //search the best base classifier
        //loop of base classifier
        for (int iClass1 = 0; iClass1 < _nClass; iClass1++) {
            double L = 0.;
            for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
                maxF = -1.e300;
                memset(_F, 0, sizeof (double)*_nClass);
                sumF = 0.;
                for (int iClass2 = 0; iClass2 < _nClass; iClass2++) {
                    if (iClass1 == iClass2)
                        continue;
                    double d;
                    _df[iClass1 * _nClass + iClass2]->eval(_data->_trainX + iEvent * _nDimension,&d);
//                    if(iClass1<iClass2)
                        _F[iClass2] = _data->_trainF[iEvent * _nClass + iClass2] + _shrinkage * d;
//                    else
//                        _F[iClass2] = _data->_trainF[iEvent * _nClass + iClass2] - 
//                                _shrinkage * d;
                    sumF += _F[iClass2];
                    if (_F[iClass2] < maxF)
                        maxF = _F[iClass2];
                }
                _F[iClass1] = -sumF;
                if (_F[iClass1] > maxF) maxF = _F[iClass1];
                sumExpF = 0.;
                for (int iClass2 = 0; iClass2 < _nClass; iClass2++)
                    sumExpF += exp(_F[iClass2] - maxF);
                py = exp(_F[_data->_trainClass[iEvent]] - maxF) / sumExpF;
                if (py > 0)
                    L -= log(py);
                else
                    L += 100;
            }
            if (L > maxL) {
                maxL = L;
                _baseClass = iClass1;
            }
        }
        _g = -1;
    }
    //cout<<"Base Class= "<<_baseClass<<endl;
    for (int iEvent = 0; iEvent < _nTrainEvents; iEvent++) {
        double sumD = 0.;
        for (int iClass = 0; iClass < _nClass; iClass++) {
            if (iClass == _baseClass)
                continue;
            double d;
            _df[_baseClass * _nClass + iClass]->eval(_data->_trainX + iEvent * _nDimension,&d);
            _data->_trainDescendingDirection[iEvent * _nClass + iClass] = d;
            sumD += d;
        }
        _data->_trainDescendingDirection[iEvent * _nClass + _baseClass] = -1. * sumD;
    }
    for (int iEvent = 0; iEvent < _nTestEvents; iEvent++) {
        double sumD = 0.;
        for (int iClass = 0; iClass < _nClass; iClass++) {
            if (iClass == _baseClass)
                continue;
            double d;
            _df[_baseClass * _nClass + iClass]->eval(_data->_testX + iEvent * _nDimension,&d);
//            if(iClass<_baseClass)
//                d*= -1.;
            _data->_testDescendingDirection[iEvent * _nClass + iClass] = d;
            sumD += d;
        }
        _data->_testDescendingDirection[iEvent * _nClass + _baseClass] = -1. * sumD;
    }
    _g++;
    for(int iClass=0;iClass<_nClass;iClass++){
        if(iClass==_baseClass)
            continue;
        ((treeScalarDirection*)_df[_baseClass * _nClass + iClass])->printInfo();
    }
}
void linearSearch::updateDirection(int iRound){
    switch(_treeType){
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            updateDirection1();
            break;
        case directionFunction::_ABC_LOGITBOOST_:
            updateDirection2();
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            for (int iEvent = 0; iEvent < _data->_nTrainEvents; iEvent++)
                _df[0]->eval(_data->_trainX + iEvent * _data->_nDimension, _data->_trainDescendingDirection + iEvent * _data->_nClass);
            for (int iEvent = 0; iEvent < _data->_nTestEvents; iEvent++)
                _df[0]->eval(_data->_testX + iEvent * _data->_nDimension, _data->_testDescendingDirection + iEvent * _data->_nClass);
            break;
        default:
            cout<<"Has not been implemented!"<<endl;
            exit(-1);
            break;
    }
//    if (iRound == 1) {
//        cout << "Direction: " << endl;
//        for (int ix = 0; ix < _data->_nTrainEvents; ix++) {
//            for (int ic = 0; ic < _data->_nClass; ic++) {
//                cout << _data->_trainDescendingDirection[ix * _data->_nClass + ic] << ", ";
//            }
//            cout << endl;
//        }
//        exit(0);
//    }
}
void linearSearch::buildDirection(){
    switch(_treeType){
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            for(int id=0;id<_nDirection;id++)
                _df[id]->buildDirection();
            break;
        case directionFunction::_ABC_LOGITBOOST_:
            for(int iClass1=0;iClass1<_data->_nClass;iClass1++){
                if(_g<_G&&iClass1!=_baseClass){
                    continue;
                }
                for(int iClass2=0;iClass2<_data->_nClass;iClass2++){
                    if(iClass2==iClass1)
                        continue;
                    _df[iClass1*_data->_nClass+iClass2]->buildDirection();
                }
            }
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            _df[0]->buildDirection();
            break;
        default:
            cout<<"Has not been implemented!"<<endl;
            exit(-1);
            break;
    }
}
void linearSearch::saveDirection(ofstream& fileDB){
    switch(_treeType){
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            for(int id=0;id<_nDirection;id++)
                _df[id]->saveTree(fileDB);
            break;
        case directionFunction::_ABC_LOGITBOOST_:
            fileDB<<_baseClass<<" "<<endl;
            for(int iClass1=0;iClass1<_data->_nClass;iClass1++){
                if(iClass1==_baseClass)
                    continue;
                _df[_baseClass*_data->_nClass+iClass1]->saveTree(fileDB);
            }
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            _df[0]->saveTree(fileDB);
            break;
        default:
            cout<<"Has not been implemented!"<<endl;
            exit(-1);
            break;
    }
}