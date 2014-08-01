/* 
 * File:   treeScalarDirection.h
 * Author: kaiwu
 *
 * Created on March 16, 2014, 11:22 PM
 */
#ifndef TREESCALARDIRECTION_H
#define	TREESCALARDIRECTION_H
#include <string.h>
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
        //_rootNode->printInfo("",true);
    }
    //tree evaluation
    void eval(double* pnt, double* direction);
    void   buildIndex(int** dataIndex=NULL,int** dataReverseIndex0=NULL);

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
        double _nodeGain;
        //raw _nodeSumH   --> sum of Hessian elements
        //    _nodeSumG   --> sum of Gradient elements
        double _nodeSumH;
        double _nodeSumG;
        //best split gain with this node
        double _additiveGain;
        //whether this point is an internal point
        bool _isInternal;
        int _iDimension;
        double _cut;
        //regression value of current node
        double _f;
        bool _ableSplit;
        dataManager* _data;
        treeScalarDirection* _tree;
        void bestNode(NODE*& n,double& gain);
        void splitNode();
        bool printInfo(const char* indent,bool last);
        //select best working class
        void calculateF();
        double leftSumG, leftSumH;
        double rightSumG, rightSumH;
        double leftSumG1, leftSumH1;
        double rightSumG1, rightSumH1;
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

