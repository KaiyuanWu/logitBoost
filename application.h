/* 
 * File:   application.h
 * Author: kaiwu
 *
 * Created on August 4, 2014, 7:42 PM
 */

#ifndef APPLICATION_H
#define	APPLICATION_H
#include<fstream>
#include<iostream>
#include<sstream>
#include<string>
#include"directionFunction.h"

using namespace std;
class application {
public:
    struct _NODE_ {
        int _iDimension;
        float _cut;
        float _f;
        bool _isInternal;
        int _class;
        struct _NODE_* _leftChildNode;
        struct _NODE_* _rightChildNode;
        struct _NODE_* _parentNode;

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
    application(const char* fileDBName);
    application(const application& orig);
    void eval(double* pnt, double* f,int nIteration=0);
    void eval(double* pnt,int iIteration);
    double* _direction;
    void evalV(double* pnt,int iIteration);
    void evalS(double* pnt,int iIteration);
    virtual ~application();
    int _nClass;
    int _nVariable;
    int _nMaximumIteration;
    double _shrinkage;
private:
    string _fileDBName;
    ifstream* _fileDB;
    directionFunction::_TREE_TYPE_ _treeType;
    bool init();
    void buildTree(const char* tree,struct _NODE_* root);
    struct _NODE_** _bootedTrees;
    int _nTrees;
    int* _baseClass;        
};

#endif	/* APPLICATION_H */

