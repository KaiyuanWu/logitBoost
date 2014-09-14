/* 
 * File:   loadModel.h
 * Author: kaiwu
 *
 * Created on September 14, 2014, 2:24 PM
 */

#ifndef LOADMODEL_H
#define	LOADMODEL_H
#include <fstream>
#include <iostream>
#include "stdlib.h"
#include "string.h"
#include "dataManager.h"
#include "directionFunction.h"

using namespace std;
class loadModel {
    struct _NODE_ {
        int _iDimension;
        float _cut;
        float _f;
        bool _isInternal;
        int _class;
        struct _NODE_* _leftChildNode;
        struct _NODE_* _rightChildNode;
        struct _NODE_* _parentNode;
    };
public:
    loadModel(const char* oldModelFileName, const char* oldOutputName, 
                     const char* newModelFileName, const char* newOutputName, dataManager* data);
    void rebuild();
    virtual ~loadModel();
private:
    ifstream _oldModelFile;
    ifstream _oldOutput;
    ofstream _newModelFile;
    ofstream _newOutput;
    
    dataManager* _data;
    int _availableIterations;
    
    
    directionFunction::_TREE_TYPE_ _treeType;
    int _nClass;
    int _nVariable;
    int _nMaximumIteration;
    double _shrinkage;
    double *_direction=NULL;
    //for abc-LogitBoost
    int _baseClass;
    int  _nTrees;
    struct _NODE_* _trees=NULL;
    
    
    //load decision tree from the db file
    bool loadTree(bool isFirstIteration=false);
    bool buildTree(const char* tree,struct _NODE_* root);
};

#endif	/* LOADMODEL_H */

