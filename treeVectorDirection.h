/* 
 * File:   treeVectorDirection.h
 * Author: kaiwu
 *
 * Created on March 16, 2014, 11:22 PM
 */
#ifndef TREEVECTORDIRECTION_H
#define	TREEVECTORDIRECTION_H
#include <string.h>
#include "directionFunction.h"
#include "dataManager.h"
#include "LossFunction.h"
#include "bitArray.h"

class treeVectorDirection:public directionFunction {
public:
    treeVectorDirection(dataManager* data,int nLeaves, int minimumNodeSize, _TREE_TYPE_ treeType);
    virtual ~treeVectorDirection();
    void eval(double* pnt, double* direction);
    void buildDirection();
    //Data Manager
    dataManager* _data;
    void printInfo(){
        //_rootNode->printInfo("",true);
    }
private:
    class NODE{
    public:
        NODE(dataManager* data,treeVectorDirection* tree,int leftPoint,int rightPoint, double loss);
        ~NODE();
        int _leftPoint;
        int _rightPoint;
        NODE* _leftChildNode;
        NODE* _rightChildNode;
        //loss
        double _nodeLoss;
        //gain of current node
        double _nodeGain;
        //raw _nodeSumH   --> sum of Hessian elements
        //    _nodeSumG   --> sum of Gradient elements
        double* _nodeSumH;
        double* _nodeSumG;
        //best split gain with this node
        double _additiveGain;
        //whether this point is an internal point
        bool _isInternal;
        int _iDimension;
        double _cut;
        //current working class
        int _class;
        //regression value of current node
        double _f;
        bool _ableSplit;
        dataManager* _data;
        treeVectorDirection* _tree;
        void bestNode(NODE*& n,double& gain);
        void splitNode();
        bool printInfo(const char* indent,bool last);
        //select best working class
        void selectBestClass();

        //variables for node split
        int _nG;
        double *leftSumG, *leftSumH;
        double *rightSumG, *rightSumH;
        double *leftSumG1, *leftSumH1;
        double *rightSumG1, *rightSumH1;
        double leftLoss, rightLoss;
        double leftLoss1, rightLoss1;
        
    };
    
    bitArray* _indexMask;
    //recursively reset the root node;
    void resetRootNode();
    //tree array for different classes
    NODE* _rootNode;
    //tree evaluation
    double evalp(double* s,int& iClass);
    //initialize root nodes
    void initNode();
    void reArrange(NODE* node, int splitPoint);
    int _nG;
};

#endif	/* TREEVECTORDIRECTION_H */

