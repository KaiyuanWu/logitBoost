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
        void clear(){
          if(_leftChildNode){
              _leftChildNode->clear();
              delete _leftChildNode;
          }  
          if(_rightChildNode){
              _rightChildNode->clear();
              delete _rightChildNode;
          }
        };
         void printInfo(const char* indent, bool last) {
            bool ret;
            char leftS[1024];
            char rightS[1024];
            if (_isInternal) {
                if (last) {
                    sprintf(rightS, "%s", indent);
                    printf("%s|-x%.2d<=%.6f f= %f\n", rightS, _iDimension, _cut, _f);
                    sprintf(leftS, "%s|  ", indent);
                    _leftChildNode->printInfo(leftS, true);
                    _rightChildNode->printInfo(leftS, false);
                } else {
                    sprintf(rightS, "%s", indent);
                    printf("%s+-x%.2d<=%.6f f= %f\n", rightS, _iDimension, _cut, _f);
                    sprintf(leftS, "%s  ", indent);
                    _leftChildNode->printInfo(leftS, true);
                    _rightChildNode->printInfo(leftS, false);
                }
            } else {
                if (!last) {
                    sprintf(rightS, "%s", indent);
                    printf("%s+-f= %f\n", rightS, _f);
                } else {
                    sprintf(rightS, "%s", indent);
                    printf("%s|-f= %f\n", rightS, _f);
                }
            }
        };
    };
public:
    loadModel(const char* oldModelFileName, const char* oldOutputName, 
                     const char* newModelFileName, const char* newOutputName, dataManager* data);
    void rebuild();
    virtual ~loadModel();
    int _availableIterations;
private:
    ifstream _oldModelFile;
    ifstream _oldOutput;
    ofstream _newModelFile;
    ofstream _newOutput;
    
    dataManager* _data;

    
    
    directionFunction::_TREE_TYPE_ _treeType;
    int _nClass;
    int _nVariable;
    int _nMaximumIteration;
    double _shrinkage;
    double *_direction;
    //for abc-LogitBoost
    int _baseClass;
    int  _nTrees;
    struct _NODE_** _trees;
    string _treeDescription;
    string _modelDescription;
    
    
    //load decision tree from the db file
    bool loadTree(bool isFirstIteration=false);
    bool buildTree(const char* tree,struct _NODE_* root);
    void updateDirection();
    void evalS(double* pnt);
    void evalV(double* pnt);
    
};

#endif	/* LOADMODEL_H */

