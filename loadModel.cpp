/*
 * File:   loadModel.cpp
 * Author: kaiwu
 *
 * Created on September 14, 2014, 2:24 PM
 */

#include "loadModel.h"
#include <sstream>

loadModel::loadModel(const char* oldModelFileName, const char* oldOutputName,
        const char* newModelFileName, const char* newOutputName, dataManager* data):_direction(NULL),_trees(NULL),
_oldModelFile(oldModelFileName),_oldOutput(oldOutputName),_newModelFile(newModelFileName),_newOutput(newOutputName)
{
    _fileOK = true;
    if(!_oldModelFile.open(QIODevice::ReadOnly)){
        cout << "Can not open " << oldModelFileName << endl;
        _fileOK = false;
    }
    if(!_oldOutput.open( QIODevice::ReadOnly)){
        cout << "Can not open " << oldOutputName << endl;
        _fileOK = false;
    }
    if(!_newModelFile.open( QIODevice::WriteOnly)){
        cout << "Can not open " << newModelFileName << endl;
        _fileOK = false;
    }
    if(!_newOutput.open( QIODevice::WriteOnly)){
         cout << "Can not open " << newOutputName << endl;
        _fileOK = false;
    }
    //try to close all open files
    if(!_fileOK){
        _oldModelFile.close();
        _oldOutput.close();
        _newModelFile.close();
        _newOutput.close();
    }
    else{
        _oldModelFileReader.setDevice(&_oldModelFile);
        _oldOutputReader.setDevice(&_oldOutput);
        _newModelFileReader.setDevice(&_newModelFile);
        _newOutputReader.setDevice(&_newOutput);
    }

    _data = data;
    _availableIterations = 0;

}

loadModel::~loadModel() {
    _oldModelFile.close();
    _oldOutput.close();
    _newModelFile.close();
    _newOutput.close();
    if (_direction) delete[] _direction;
    if (_trees){
        for(int iTree=0;iTree<_nTrees;iTree++)
            delete _trees[iTree];
        delete[] _trees;
    }
}
//update the dataManager
void loadModel::updateDirection() {
    for (int iEvent = 0; iEvent < _data->_nTrainEvents; iEvent++) {
        switch (_treeType) {
            case directionFunction::_ABC_LOGITBOOST_:
                evalS(_data->_trainX+iEvent*_nVariable);
                for (int iClass = 0; iClass < _nClass; iClass++)
                    _direction[iClass] *= -1;
                break;
            case directionFunction::_MART_:
            case directionFunction::_LOGITBOOST_:
                evalS(_data->_trainX+iEvent*_nVariable);
                break;
            case directionFunction::_AOSO_LOGITBOOST_:
            case directionFunction::_SLOGITBOOST_:
                evalV(_data->_trainX+iEvent*_nVariable);
                break;
            default:
                cout << "This tree type " << _treeType << " has not been implemented!" << endl;
                break;
        }
        for (int iClass = 0; iClass < _nClass; iClass++)
            _data->_trainDescendingDirection[iEvent*_nClass+iClass]=_direction[iClass];
    }
    _data->increment(_shrinkage,_availableIterations);
}
void loadModel::evalS(float * pnt){
    for (int iClass = 0; iClass < _nClass; iClass++)
        _direction[iClass] = 0.;
    for(int iTree = 0; iTree < _nTrees; iTree++) {
        struct _NODE_ *n = _trees[iTree];
        while (n->_isInternal) {
            if (pnt[n->_iDimension] <= n->_cut) {
                n = n->_leftChildNode;
            } else {
                n = n->_rightChildNode;
            }
            if(!n){
                cout<<_modelDescription<<endl;
                for(int iT=0;iT<_nTrees;iT++){
                    cout<<"========================== Tree "<<iT<<"=========================="<<endl;
                    _trees[iT]->printInfo("",false);
                }
                exit(-1);
            }
        }
        _direction[n->_class%_nClass]=n->_f;
    }
    if(_treeType==directionFunction::_ABC_LOGITBOOST_){
        for(int iClass=0;iClass<_nClass;iClass++){
            if(iClass!=_baseClass){
                _direction[_baseClass]-=_direction[iClass];
            }
        }
    }
}
void loadModel::evalV(float * pnt) {
    for (int iClass = 0; iClass < _nClass; iClass++)
        _direction[iClass] = 0.;
    struct _NODE_ *n = _trees[0];
//    _bootedTrees[iIteration]->printInfo("",true);
    while (n->_isInternal) {
        if (pnt[n->_iDimension] <= n->_cut) {
            n = n->_leftChildNode;
        } else {
            n = n->_rightChildNode;
        }
    }
    float  f;
    int workingClass, workingClass1, workingClass2;
    f = n->_f;
    workingClass = n->_class;
    workingClass1 = workingClass / _nClass;
    workingClass2 = workingClass % _nClass;
    switch (_treeType) {
        case directionFunction::_SLOGITBOOST_:
            for (int iClass = 0.; iClass < _nClass; iClass++) {
                if (iClass == workingClass)
                    _direction[iClass] = (_nClass - 1.) * f;
                else
                    _direction[iClass] = -f;
            }
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
            for (int iClass = 0; iClass < _nClass; iClass++) {
                if (iClass == workingClass1)
                    _direction[iClass] = f;
                else if (iClass == workingClass2)
                    _direction[iClass] = -f;
                else
                    _direction[iClass] = 0;
            }
            break;
        default:
            cout << "This tree type " << _treeType << " has not been implemented!" << endl;
            break;
    }
}
//try to rebuild the dataManager from previous train result
void loadModel::rebuild() {
    if(!_fileOK)
        return;
    //variables list
    float  oldTrainAccuracy, oldTestAccuracy, oldTrainLoss, oldTestLoss;
    float  newTrainAccuracy, newTrainLoss;
    
    _availableIterations=0;
    _oldOutputReader>>oldTrainAccuracy>>oldTestAccuracy>>oldTrainLoss>>oldTestLoss;
    
    if(!loadTree(true))
        return;

    while(!_oldOutputReader.atEnd()&&!_oldModelFileReader.atEnd()){
        updateDirection();
        for(int iTree=0;iTree<_nTrees;iTree++)
            _trees[iTree]->clear();
        newTrainAccuracy=_data->_trainAccuracy;
        newTrainLoss=_data->_trainLoss;
        if(fabs(newTrainLoss-oldTrainLoss)<0.00001*oldTrainLoss){
            _availableIterations++;
            for(int iTree=0;iTree<_nTrees;iTree++)
                _trees[iTree]->saveNode(_newModelFileReader,_baseClass);
            _newModelFileReader<<qint8(';');
            _newOutputReader<<newTrainAccuracy<<newTrainLoss;
        }
        else
            break;
        _oldOutputReader>>oldTrainAccuracy>>oldTestAccuracy>>oldTrainLoss>>oldTestLoss;
        if (oldTrainAccuracy == 0)
            break;
        if(!loadTree())
            return;
    }
}

bool loadModel::loadTree(bool isFirstIteration) {
    bool ret = true;
    _modelDescription="";
    //if this is the first iteration, we will load informations about the tree
    if (isFirstIteration) {
        int k;
        _oldModelFileReader >>k;
        _treeType = directionFunction::_TREE_TYPE_(k);
        _oldModelFileReader >> _nClass >> _nVariable >> _nMaximumIteration>>_shrinkage;
        
        if(_nClass==0||_nVariable==0||_nMaximumIteration==0)
            return false;
        _direction = new float [_nClass];
        switch (_treeType) {
            case directionFunction::_ABC_LOGITBOOST_:
                _nTrees = _nClass - 1;
                break;
            case directionFunction::_MART_:
            case directionFunction::_LOGITBOOST_:
                _nTrees = _nClass;
                break;
            case directionFunction::_AOSO_LOGITBOOST_:
            case directionFunction::_SLOGITBOOST_:
                _nTrees = 1;
                break;
            default:
                cout << "This tree type " << _treeType << " has not been implemented!" << endl;
                break;
        }
        _trees = new struct _NODE_*[_nTrees];
        for(int iT=0;iT<_nTrees;iT++)
            _trees[iT]=new struct _NODE_;
    }
    if (_treeType == directionFunction::_ABC_LOGITBOOST_)
        _oldModelFileReader >> _baseClass;
    for (int iTree = 0; iTree < _nTrees; iTree++) {
        ret=buildTree(_oldModelFileReader, _trees[iTree]);
        if(!ret)
            break;
    }
    return ret;
}
bool loadModel::buildTree(QDataStream& fileDBReader, struct _NODE_* root){
    qint8 op;
    int  iDimension,iclass;
    float cut,f;
    bool internal;
    struct _NODE_* curNode;
    root->_leftChildNode=NULL;
    root->_rightChildNode=NULL;
    root->_parentNode=NULL;
    curNode=root;
    fileDBReader >> iDimension >> cut>>f>>internal>>iclass;
    
    root->_cut = cut;
    root->_iDimension = iDimension;
    root->_f = f;
    root->_isInternal=internal;
    root->_class=iclass;
    while(1){
        fileDBReader>>op;
        switch(op){
            case '(':
                curNode->_leftChildNode=new struct _NODE_;
                curNode->_leftChildNode->_leftChildNode=NULL;
                curNode->_leftChildNode->_rightChildNode=NULL;
                curNode->_leftChildNode->_parentNode=curNode;
                curNode->_rightChildNode=new struct _NODE_;
                curNode->_rightChildNode->_leftChildNode=NULL;
                curNode->_rightChildNode->_rightChildNode=NULL;
                curNode->_rightChildNode->_parentNode=curNode;
                curNode=curNode->_leftChildNode;
                break;
                
            case ')':
                curNode=curNode->_parentNode;
                break;
                
            case '+':
                curNode=curNode->_rightChildNode;
                fileDBReader>>op;
                break;
            case ';':
                break;      
            default:
                cout<<"Error: parse the database!";
                return false;
                break;
        }
        if(op==';')
            break;
        if(op=='(') {
            fileDBReader >> iDimension >> cut>>f>>internal>>iclass;
            curNode->_cut = cut;
            curNode->_iDimension = iDimension;
            curNode->_f = f;
            curNode->_isInternal=internal;
            curNode->_class=iclass;
        }
    }
    return true;
}
void loadModel::_NODE_::saveNode(QDataStream& fileDBReader,int baseClass){
    fileDBReader<<_iDimension<<_cut<<_f<<_isInternal<<baseClass;
    if(_isInternal){
        fileDBReader<<qint8('(');
        _leftChildNode->saveNode(fileDBReader);
        fileDBReader<<qint8(')')<<qint8('+')<<qint8('()');
        _rightChildNode->saveNode(fileDBReader);
        fileDBReader<<qint8(')');
    }
}