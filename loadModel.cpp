/* 
 * File:   loadModel.cpp
 * Author: kaiwu
 * 
 * Created on September 14, 2014, 2:24 PM
 */

#include "loadModel.h"
#include <sstream>

loadModel::loadModel(const char* oldModelFileName, const char* oldOutputName,
        const char* newModelFileName, const char* newOutputName, dataManager* data):_direction(NULL),_trees(NULL) {
    _oldModelFile.open(oldModelFileName, ifstream::in);
    _oldOutput.open(oldOutputName, ifstream::in);
    _newModelFile.open(newModelFileName, ofstream::out);
    _newOutput.open(newOutputName, ofstream::out);
    //check the files
    _fileOK = true;
    
    if (!_oldModelFile.good()) {
        cout << "Can not open " << oldModelFileName << endl;
        _fileOK = false;
    }
    if (!_oldOutput.good()) {
        cout << "Can not open " << oldOutputName << endl;
        _fileOK = false;
    }
    if (!_newModelFile.good()) {
        cout << "Can not open " << newModelFileName << endl;
        _fileOK = false;
    }
    if (!_newOutput.good()) {
        cout << "Can not open " << newOutputName << endl;
        _fileOK = false;
    }


    _data = data;
    _availableIterations = 0;

}

loadModel::~loadModel() {
    if (_oldModelFile.is_open()) {
        _oldModelFile.close();
    }
    if (_newModelFile.is_open()) {
        _newModelFile.close();
    }
    if (_oldOutput.is_open()) {
        _oldOutput.close();
    }
    if (_newOutput.is_open()) {
        _newOutput.close();
    }
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
void loadModel::evalS(double* pnt){
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
void loadModel::evalV(double* pnt) {
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
    double f;
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
    char oldTrainAccuracyStr[1024], oldTestAccuracyStr[1024], oldTrainLossStr[1024], oldTestLossStr[1024];
    
    double oldTrainAccuracy, oldTestAccuracy, oldTrainLoss, oldTestLoss;
    double newTrainAccuracy, newTrainLoss;
    
    _availableIterations=0;
    _oldOutput>>oldTrainAccuracyStr>>oldTestAccuracyStr>>oldTrainLossStr>>oldTestLossStr;
    oldTrainAccuracy=atof(oldTrainAccuracyStr);
    oldTestAccuracy=atof(oldTestAccuracyStr);
    oldTrainLoss=atof(oldTrainLossStr);
    oldTestLoss=atof(oldTestLossStr);
    if(oldTrainAccuracy==0.0)
        return;
    if(!loadTree(true))
        return;
    
    while(_oldOutput.good()&&_oldModelFile.good()){
        updateDirection();
        for(int iTree=0;iTree<_nTrees;iTree++)
            _trees[iTree]->clear();
        newTrainAccuracy=_data->_trainAccuracy;
        newTrainLoss=_data->_trainLoss;
        if(fabs(newTrainLoss-oldTrainLoss)<0.00001*oldTrainLoss){
            _availableIterations++;
            _newModelFile<<_modelDescription;
            _newOutput<<newTrainAccuracy<<" "<<newTrainLoss<<endl;
        }
        else
            break;
        _oldOutput>>oldTrainAccuracyStr>>oldTestAccuracyStr>>oldTrainLossStr>>oldTestLossStr;
        oldTrainAccuracy = atof(oldTrainAccuracyStr);
        oldTestAccuracy = atof(oldTestAccuracyStr);
        oldTrainLoss = atof(oldTrainLossStr);
        oldTestLoss = atof(oldTestLossStr);
        if (oldTrainAccuracy == 0.0)
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
        _oldModelFile>>k;
        _treeType = directionFunction::_TREE_TYPE_(k);
        _oldModelFile >> _nClass >> _nVariable >> _nMaximumIteration>>_shrinkage;
        //_newModelFile <<k<<" "<<_nClass<<" "<<_nVariable<<" "<<_nMaximumIteration<<" "<<_shrinkage<<endl;
        _direction = new double[_nClass];
        
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
        //skip the first endl
        getline(_oldModelFile, _treeDescription);
    }
    if (_treeType == directionFunction::_ABC_LOGITBOOST_) {
        getline(_oldModelFile, _treeDescription);
        _modelDescription += _treeDescription;
        _modelDescription += "\n";
        _baseClass = atoi(_treeDescription.c_str());
    }
    for (int iTree = 0; iTree < _nTrees; iTree++) {
        getline(_oldModelFile, _treeDescription);
        _modelDescription += _treeDescription;
        _modelDescription += "\n";
        ret=buildTree(_treeDescription.c_str(), _trees[iTree]);
        if(!ret){
            break;
        }
    }
    return ret;
}

bool loadModel::buildTree(const char* tree, struct _NODE_* root) {
    bool ret=true;
    char op;
    int iDimension, iclass;
    float cut, f;
    bool internal;
    struct _NODE_* curNode;
    root->_leftChildNode = NULL;
    root->_rightChildNode = NULL;
    root->_parentNode = NULL;
    curNode = root;
    stringstream ss;
    ss.str(tree);
    ss >> iDimension >> cut >> f >> internal>>iclass;

    root->_cut = cut;
    root->_iDimension = iDimension;
    root->_f = f;
    root->_isInternal = internal;
    root->_class = iclass;
    while (ss.good()) {
        ss>>op;
        switch (op) {
            case '(':
                curNode->_leftChildNode = new struct _NODE_;
                curNode->_leftChildNode->_leftChildNode = NULL;
                curNode->_leftChildNode->_rightChildNode = NULL;
                curNode->_leftChildNode->_parentNode = curNode;
                curNode->_rightChildNode = new struct _NODE_;
                curNode->_rightChildNode->_leftChildNode = NULL;
                curNode->_rightChildNode->_rightChildNode = NULL;
                curNode->_rightChildNode->_parentNode = curNode;
                curNode = curNode->_leftChildNode;
                break;

            case ')':
                curNode = curNode->_parentNode;
                break;

            case '+':
                curNode = curNode->_rightChildNode;
                ss>>op;
                break;
            default:
                ret=false;
                break;
        }
        if (op == '(') {
            ss >> iDimension >> cut >> f >> internal>>iclass;
            curNode->_cut = cut;
            curNode->_iDimension = iDimension;
            curNode->_f = f;
            curNode->_isInternal = internal;
            curNode->_class = iclass;
        }
    }
    
    return ret;
}