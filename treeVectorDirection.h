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
    void eval(float * pnt, float * direction);
    void buildDirection();
    void saveTree(ofstream& fileDB);
    //Data Manager
    dataManager* _data;
    void printInfo(){
        //_rootNode->printInfo("",true);
    }
private:
    class NODE{
    public:
        NODE(dataManager* data,treeVectorDirection* tree,int leftPoint,int rightPoint, float  loss);
        ~NODE();
        int _leftPoint;
        int _rightPoint;
        NODE* _leftChildNode;
        NODE* _rightChildNode;
        //loss
        float  _nodeLoss;
        //gain of current node
        float  _nodeGain;
        //raw _nodeSumH   --> sum of Hessian elements
        //    _nodeSumG   --> sum of Gradient elements
        float * _nodeSumH;
        float * _nodeSumG;
        //best split gain with this node
        float  _additiveGain;
        //whether this point is an internal point
        bool _isInternal;
        int _iDimension;
        float  _cut;
        //current working class
        int _class;
        //regression value of current node
        float  _f;
        bool _ableSplit;
        dataManager* _data;
        treeVectorDirection* _tree;
        void bestNode(NODE*& n,float & gain);
        void splitNode();
        void saveNode(ofstream& fileDB);
        bool printInfo(const char* indent,bool last);
        //select best working class
        void selectBestClass();

        //variables for node split
        int _nG;
        float  *leftSumG, *leftSumH;
        float  *rightSumG, *rightSumH;
        float  *leftSumG1, *leftSumH1;
        float  *rightSumG1, *rightSumH1;
        float  leftLoss, rightLoss;
        float  leftLoss1, rightLoss1;
        
    };
    
    bitArray* _indexMask;
    //recursively reset the root node;
    void resetRootNode();
    //tree array for different classes
    NODE* _rootNode;
    //tree evaluation
    float  evalp(float * s,int& iClass);
    //initialize root nodes
    void initNode();
    void reArrange(NODE* node, int splitPoint);
    int _nG;
};

#endif	/* TREEVECTORDIRECTION_H */

