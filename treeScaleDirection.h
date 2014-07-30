/* 
 * File:   treeScaleDirection.h
 * Author: kaiwu
 *
 * Created on March 16, 2014, 11:22 PM
 */
#ifndef TREESINGLEDIRECTION_H
#define	TREESINGLEDIRECTION_H
#include <string.h>
#include "directionFunction.h"
#include "dataManager.h"
#include "LossFunction.h"
#include "bitArray.h"

class treeScaleDirection:public directionFunction {
public:
    treeScaleDirection(dataManager* data,int nLeaves,int treeClass);
    virtual ~treeScaleDirection();
    void buildDirection();
    //Data Manager
    dataManager* _data;
    int _treeClass;
    void printInfo(){
        //_rootNode->printInfo("",true);
    }
    //tree evaluation
    double eval(double* s);
    void   buildIndex(int** dataIndex=NULL,int** dataReverseIndex0=NULL);

private:
    class NODE{
    public:
        NODE(dataManager* data,treeScaleDirection* tree,int leftPoint,int rightPoint);
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
        treeScaleDirection* _tree;
        void bestNode(NODE*& n,double& gain);
        void splitNode();
        bool printInfo(const char* indent,bool last);
        //select best working class
        void calculateF();
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
};


#endif	/* TREESINGLEDIRECTION_H */

