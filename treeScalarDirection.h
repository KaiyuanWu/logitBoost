/* 
 * File:   treeScalarDirection.h
 * Author: kaiwu
 *
 * Created on March 16, 2014, 11:22 PM
 */
#ifndef TREESCALARDIRECTION_H
#define	TREESCALARDIRECTION_H
#include <string.h>
#include <QFile>
#include <QDataStream>
#include "directionFunction.h"
#include "dataManager.h"
#include "LossFunction.h"
#include "bitArray.h"

class treeScalarDirection:public directionFunction {
public:
    treeScalarDirection(dataManager* data,int nLeaves, int minimumNodeSize, _TREE_TYPE_ treeType,int treeClass);
    virtual ~treeScalarDirection();
    void buildDirection();
    //Data Manager
    dataManager* _data;
    int _treeClass;
    void printInfo(){
        _rootNode->printInfo("",true);
    }
    //tree train
    void eval(float * pnt, float * direction);
    void   buildIndex(int** dataIndex=NULL,int** dataReverseIndex0=NULL);
    void saveTree(QDataStream& fileDBReader);
private:
    class NODE{
    public:
        NODE(dataManager* data,treeScalarDirection* tree,int leftPoint,int rightPoint);
        ~NODE();
        int _leftPoint;
        int _rightPoint;
        NODE* _leftChildNode;
        NODE* _rightChildNode;
        //gain of current node
        float  _nodeGain;
        //raw _nodeSumH   --> sum of Hessian elements
        //    _nodeSumG   --> sum of Gradient elements
        float  _nodeSumH;
        float  _nodeSumG;
        //best split gain with this node
        float  _additiveGain;
        //whether this point is an internal point
        bool _isInternal;
        int _iDimension;
        float  _cut;
        //regression value of current node
        float  _f;
        bool _ableSplit;
        dataManager* _data;
        treeScalarDirection* _tree;
        void bestNode(NODE*& n,float & gain);
        void splitNode();
        bool printInfo(const char* indent,bool last);
        void saveNode(QDataStream& fileDBReader);
        //select best working class
        void calculateF();
        float  leftSumG, leftSumH;
        float  rightSumG, rightSumH;
        float  leftSumG1, leftSumH1;
        float  rightSumG1, rightSumH1;
    };
    bitArray* _indexMask;
    //recursively sort the projected data
    void reArrange(NODE* node,int splitPoint);
    
    //recursively reset the root node;
    void resetRootNode();
    //tree array for different classes
    NODE* _rootNode;
    //initialize root nodes
    void initNode();
    
    int _nG;
};


#endif	/* TREESCALARDIRECTION_H */

