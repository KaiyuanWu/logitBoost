/* 
 * File:   loadModel.cpp
 * Author: kaiwu
 * 
 * Created on September 14, 2014, 2:24 PM
 */

#include "loadModel.h"
#include <sstream>

loadModel::loadModel(const char* oldModelFileName, const char* oldOutputName,
        const char* newModelFileName, const char* newOutputName, dataManager* data) {
    _oldModelFile.open(oldModelFileName, ifstream::in);
    _oldOutput.open(oldOutputName, ifstream::in);
    _newModelFile.open(newModelFileName, ofstream::out);
    _newOutput.open(newOutputName, ofstream::out);
    //check the files
    bool fileOK = true;
    if (!_oldModelFile.good()) {
        cout << "Can not open " << oldModelFileName << endl;
        fileOK = false;
    }
    if (!_oldOutput.good()) {
        cout << "Can not open " << oldOutputName << endl;
        fileOK = false;
    }
    if (!_newModelFile.good()) {
        cout << "Can not open " << newModelFileName << endl;
        fileOK = false;
    }
    if (!_newOutput.good()) {
        cout << "Can not open " << newOutputName << endl;
        fileOK = false;
    }
    if (!fileOK) {
        exit(-1);
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
    if (_trees) delete[] _trees;
}

//try to rebuild the dataManager from previous train result

void loadModel::rebuild() {
    //variables list
    double oldTrainAccuracy, oldTestAccuracy, oldTrainLoss, oldTestLoss;
    double newTrainAccuracy, newTrainLoss;


}

bool loadModel::loadTree(bool isFirstIteration) {
    bool ret = true;
    //if this is the first iteration, we will load informations about the tree
    if (isFirstIteration) {
        int k;
        _oldModelFile>>k;
        _treeType = directionFunction::_TREE_TYPE_(k);
        _oldModelFile >> _nClass >> _nVariable >> _nMaximumIteration>>_shrinkage;
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
        _trees = new struct _NODE_[_nTrees];
    }
    string tree;
    //skip the first endl
    getline(_oldModelFile, tree);
    if (_treeType == directionFunction::_ABC_LOGITBOOST_) {
        getline(_oldModelFile, tree);
        _baseClass = atoi(tree.c_str());
    }
    for (int iTree = 0; iTree < _nTrees; iTree++) {
        getline(_oldModelFile, tree);
        buildTree(tree.c_str(), _trees[iTree]);
    }
    return ret;
}

void loadModel::buildTree(const char* tree, struct _NODE_* root) {
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
}