/*
 * File:   treeScaleDirection.cpp
 * Author: kaiwu
 *
 * Created on March 16, 2014, 11:22 PM
 */

#include <bitset>
#include <iostream>
#include "treeScaleDirection.h"

using namespace std;

double leftSumG, leftSumH;
double rightSumG,rightSumH;
double leftSumG1,leftSumH1;
double rightSumG1,rightSumH1;

treeScaleDirection::treeScaleDirection(dataManager* data, int nLeaves,int treeClass) {
    _data = data;
    _nLeaves = nLeaves;
    if (_nLeaves < 2) {
        cout << "Number of terminate nodes is " << _nLeaves << ", change it to 2!" << endl;
        _nLeaves = 2;
    }
    _treeClass=treeClass;
    _nClass     = _data->_nClass    ;
    _nDimension = _data->_nDimension;
    _nEvents = _data->_nTrainEvents ;
    _round = 0;

    _minimumNodeSize=1;
    _rootNode      = new NODE(_data, this,  0, _nEvents - 1);
    _rootNode->_isInternal = true;
    _indexMask     = new bitArray(_nEvents)   ;

    //print sort result
//    for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
//        for (int iEvent = 0; iEvent < _nEvents; iEvent++) {
//            cout<<_data->_trainX[_dataIndex0[iDimension][iEvent] * _nDimension + iDimension]<<", ";
//        }
//        cout<<endl;
//    }
    _zMax = 4.   ;
}
void treeScaleDirection::resetRootNode() {
    if (_rootNode) {
        if (_rootNode->_leftChildNode)
            delete _rootNode->_leftChildNode;
        if (_rootNode->_rightChildNode)
            delete _rootNode->_rightChildNode;
        _rootNode->_leftChildNode = NULL ;
        _rootNode->_rightChildNode = NULL;
        _rootNode->_isInternal = false;
        _rootNode->_ableSplit = true  ;
    }
}

treeScaleDirection::NODE::NODE(dataManager* data, treeScaleDirection* tree, int leftPoint, int rightPoint) {
    _additiveGain = 0.;
    _data = data;
    _tree = tree;
    _f    = 0.;
    _leftPoint  = leftPoint;
    _rightPoint = rightPoint;
    if (_rightPoint - _leftPoint >=2*_tree->_minimumNodeSize)
         _ableSplit = true;
    else
        _ableSplit = false;
    _leftChildNode  = NULL;
    _rightChildNode = NULL;
    _isInternal = false;
}

treeScaleDirection::NODE::~NODE() {
    if (_leftChildNode)
        delete _leftChildNode;
    _leftChildNode = NULL;
    if (_rightChildNode)
        delete _rightChildNode;
    _rightChildNode = NULL;
}

double treeScaleDirection::eval(double* s) {
    NODE* n = _rootNode;
    while (n->_isInternal) {
        if (s[n->_iDimension] <= n->_cut) {
            n = n->_leftChildNode;
        } else{
            n = n->_rightChildNode;
        }
    }
    return n->_f    ;
}

//initialization of a given node

void treeScaleDirection::initNode() {
    NODE* n = _rootNode;
    //calculate the gain
    n->_isInternal = false;
    n->_ableSplit  = true ;
    n->_additiveGain = 0. ;

    n->_nodeSumG=0;
    n->_nodeSumH=0;

    for (int iPoint = 0; iPoint < _nEvents; iPoint++) {
        n->_nodeSumG += _data->_lossGradient[iPoint * _nClass + _treeClass];
        n->_nodeSumH += _data->_lossHessian[iPoint * _nClass + _treeClass];
    }

    n->_iDimension=0;
    n->calculateF();
}

void treeScaleDirection::NODE::splitNode() {
    int splitPoint;
    //initialization
    double maxGain   = _nodeGain;
    int maxDimension = 0 ;
    int maxI         = -1;
    int    iDimension;
    double bestC = 0.;
    double bestLeftV,bestRightV;


    int shift=_tree->_minimumNodeSize;

    for (iDimension = 0; iDimension < _tree->_nDimension; iDimension++) {
        leftSumG= 0.;
        leftSumH= 0.;
        rightSumG= _nodeSumG;
        rightSumH= _nodeSumH;

        splitPoint = _leftPoint;
        //get rid of the same value elements
        double postX = _data->_trainX[_data->_dataIndex[iDimension][_leftPoint + 1] * _tree->_nDimension + iDimension];
        double cVal  = 1.e300   ;
        double leftV,rightV     ;
        for (splitPoint = _leftPoint; splitPoint < _leftPoint + shift; splitPoint++) {
            double g, h;
            g = _data->_lossGradient[_data->_dataIndex[iDimension][splitPoint] * _tree->_nClass + _tree->_treeClass];
            h = _data->_lossHessian[_data->_dataIndex[iDimension][splitPoint] * _tree->_nClass + _tree->_treeClass];

            leftSumG += g;
            leftSumH += h;
            rightSumG -= g;
            rightSumH -= h;
        }
        for (splitPoint = _leftPoint + shift; splitPoint <=_rightPoint-shift; splitPoint++) {
            double x = _data->_trainX[_data->_dataIndex[iDimension][splitPoint] * _tree->_nDimension + iDimension];
            double g, h;
            g = _data->_lossGradient[_data->_dataIndex[iDimension][splitPoint] * _tree->_nClass + _tree->_treeClass];
            h = _data->_lossHessian[_data->_dataIndex[iDimension][splitPoint] * _tree->_nClass + _tree->_treeClass];
            leftSumG += g;
            leftSumH += h;
            rightSumG -= g;
            rightSumH -= h;
            postX = _data->_trainX[_data->_dataIndex[iDimension][splitPoint + 1] * _tree->_nDimension + iDimension];
            if (x == postX)
                continue;
            leftV=x;rightV=postX;
            postX = x                   ;
            cVal  = 0.5 * (x + postX)   ;
            double gain=-1.;

            double newgain=0.;
            if(leftSumH>0)
                newgain+= leftSumG * leftSumG/leftSumH;
            if(rightSumH>0)
                newgain+= rightSumG*rightSumG/rightSumH;
            if (newgain > gain)
                gain = newgain;
            if (gain >= maxGain) {
                bestC     = cVal ;
                bestLeftV = leftV;
                bestRightV=rightV;
                maxGain   = gain;
                maxI      = splitPoint;
                maxDimension = iDimension;

                leftSumG1 =leftSumG;
                leftSumH1 =leftSumH;
                rightSumG1=rightSumG;
                rightSumH1=rightSumH;
            }
        }
    }
    // a better split has been found
    if (maxI != -1) {
        _iDimension = maxDimension;
        _cut        = bestC;
        _additiveGain =maxGain-_nodeGain;
        _tree->reArrange(this, maxI);

        NODE* t = new NODE(_data, _tree, _leftPoint, maxI);
        _leftChildNode = t;
        t->_nodeSumG= leftSumG1;
        t->_nodeSumH= leftSumH1;
        t->_iDimension=_iDimension;
        t->calculateF();

        t = new NODE(_data, _tree, maxI + 1, _rightPoint);
        _rightChildNode = t;
        t->_nodeSumG= rightSumG1;
        t->_nodeSumH= rightSumH1;
        t->_iDimension=_iDimension;
        t->calculateF();
    }
    _ableSplit = false;
}

void treeScaleDirection::reArrange(NODE* node, int splitPoint) {
    int iDimension = node->_iDimension;
    for (int id = 0; id < _nDimension; id++) {
        _indexMask->reset(node->_leftPoint, node->_rightPoint);
        //there is no need to rearrange current split dimension
        if (id == iDimension)
            continue;
        for (int ip = splitPoint + 1; ip <= node->_rightPoint; ip++) {
            if (!_indexMask->set(_data->_dataReverseIndex[id][_data->_dataIndex[iDimension][ip]])) {
                cout << "here here " << _shrinkage << ", " << _nLeaves << endl;
                exit(0);
            }
        }
        int left = node->_leftPoint;
        int right = splitPoint + 1;
        for (int ip = node->_leftPoint; ip <= node->_rightPoint; ip++) {
            //belong to right child node
            if (_indexMask->test(ip)) {
                _data->_dataIndexTemp[right] = _data->_dataIndex[id][ip];
                _data->_dataReverseIndex[id][_data->_dataIndexTemp[right]] = right;
                if (right > node->_rightPoint) {
                    cout << "here here!" << endl;
                }
                right++;
            } else {
                _data->_dataIndexTemp[left] = _data->_dataIndex[id][ip];
                _data->_dataReverseIndex[id][_data->_dataIndexTemp[left]] = left;
                left++;
            }
        }
        memcpy(_data->_dataIndex[id] + node->_leftPoint, _data->_dataIndexTemp + node->_leftPoint, (node->_rightPoint - node->_leftPoint + 1) * sizeof (int));
    }
}

treeScaleDirection::~treeScaleDirection() {
    delete _indexMask;
    delete  _rootNode;
}

bool treeScaleDirection::NODE::printInfo(const char* indent, bool last) {
    bool ret;
    char leftS[1024];
    char rightS[1024];
    if (_isInternal) {
        if(last){
            sprintf(rightS,"%s",indent);
            printf("%s|-x%.2d<=%.6f\n",rightS,_iDimension,_cut);
            sprintf(leftS,"%s|  ",indent);
            ret = _leftChildNode->printInfo(leftS, true);
            ret = _rightChildNode->printInfo(leftS, false);
        }
        else{
            sprintf(rightS,"%s",indent);
            printf("%s+-x%.2d<=%.6f\n",rightS,_iDimension,_cut);
            sprintf(leftS,"%s  ",indent);
            ret = _leftChildNode->printInfo(leftS, true);
            ret = _rightChildNode->printInfo(leftS, false);
        }
    } else{
        if(!last){
            sprintf(rightS,"%s",indent);
            printf("%s+-f= %f (%d,%d) \n",rightS, _f, _leftPoint, _rightPoint);
        }
        else{
            sprintf(rightS,"%s",indent);
            printf("%s|-f= %f (%d,%d) \n",rightS, _f, _leftPoint, _rightPoint);
        }
        return true;
    }
    return ret;
}
void treeScaleDirection::NODE::calculateF(){
    if(_nodeSumH>0)
        _nodeGain=_nodeSumG * _nodeSumG/_nodeSumH;
    else
        _nodeGain=0.;
    if(_nodeSumH<=0){
        _ableSplit=false;
        _f=0.;
    }
    else {
        _f = _nodeSumG/_nodeSumH;
        if(!(_f==_f)){
            _f=0;
        }
        if (!(_f <= _tree->_zMax))
            _f = _tree->_zMax;
        if (!(_f >= -_tree->_zMax))
            _f = -_tree->_zMax;
    }
}

void treeScaleDirection::buildDirection() {
    _round++;
    for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
        memcpy(_data->_dataIndex[iDimension], _data->_dataIndex0[iDimension], _nEvents * sizeof (int));
        memcpy(_data->_dataReverseIndex[iDimension], _data->_dataReverseIndex0[iDimension], _nEvents * sizeof (int));
    }
    initNode();
    for (int ileaf = 0; ileaf < _nLeaves - 1; ileaf++){
        NODE* node = NULL    ;
        double bestGain = -1.;
        _rootNode->bestNode(node, bestGain);
        if (bestGain == -1.)
            break;
        if (node->_leftChildNode)
            node->_isInternal = true;
    }
}

void treeScaleDirection::NODE::bestNode(NODE*& n, double& gain) {
    //if this node has not been split
    if (_ableSplit)
        splitNode();
    //if this node has been used as an internal node, then check the child nodes
    if (_isInternal) {
        if (_leftChildNode)
            _leftChildNode->bestNode(n, gain);
        if (_rightChildNode)
            _rightChildNode->bestNode(n, gain);
    } else {
        if (_additiveGain > gain) {
            gain = _additiveGain;
            n = this;
        }
    }
}