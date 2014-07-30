/* 
 * File:   treeVectorDiretion.h
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

class treeVectorDiretion:public directionFunction {
public:
    treeVectorDiretion(dataManager* data,int nLeaves);
    virtual ~treeVectorDiretion();
    void eval(double* pnt, double* direction,int iEvent=-1,bool isTrain=true);
    void buildDirection();
    void updateDirection();
    //Data Manager
    dataManager* _data;
    void printInfo(){
        //_rootNode->printInfo("",true);
    }
private:
    class NODE{
    public:
        NODE(dataManager* data,treeVectorDiretion* tree,int leftPoint,int rightPoint, double loss);
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
        double _leftV;
        double _rightV;
        double _purity;
        //current working class
        int _class;
        //regression value of current node
        double _f;
        bool _ableSplit;
        dataManager* _data;
        treeVectorDiretion* _tree;
        void bestNode(NODE*& n,double& gain);
        void splitNode();
        bool printInfo(const char* indent,bool last);
        //select best working class
        void selectBestClass();
    };
    
    bitArray* _indexMask;
    //recursively reset the root node;
    void resetRootNode();
    //tree array for different classes
    NODE* _rootNode;
    //tree evaluation
    double evalp(double* s,int &iClass,bool printPurity=false);
    //initialize root nodes
    void initNode();
    void reArrange(NODE* node, int splitPoint);
    
};


#endif	/* TREESINGLEDIRECTION_H */

