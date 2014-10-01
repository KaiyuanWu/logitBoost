/*
 * File:   treeScalarDirection.cpp
 * Author: kaiwu
 *
 * Created on March 16, 2014, 11:22 PM
 */

#include <bitset>
#include <iostream>
#include "treeScalarDirection.h"

using namespace std;

treeScalarDirection::treeScalarDirection(dataManager* data,int nLeaves, int minimumNodeSize, _TREE_TYPE_ treeType,int treeClass) {
    _data = data;
    _nLeaves = nLeaves;
    _minimumNodeSize=minimumNodeSize;
    if(_minimumNodeSize<1){
        _minimumNodeSize=1;
    }
    _treeType=treeType;
    _treeClass=treeClass;
    
    if (_nLeaves < 2) {
        cout << "Number of terminate nodes is " << _nLeaves << ", change it to 2!" << endl;
        _nLeaves = 2;
    }
    _nClass     = _data->_nClass    ;
    _nDimension = _data->_nDimension;
    _nEvents = _data->_nTrainEvents ;
    _round = 0;

    _rootNode      = new NODE(_data, this,  0, _nEvents - 1);
    _rootNode->_leftChildNode=NULL;
    _rootNode->_rightChildNode=NULL;
    _rootNode->_isInternal = true;
    _indexMask     = new bitArray(_nEvents)   ;
    _zMax = 4.   ;

    switch (_treeType) {
        case _MART_:
        case _LOGITBOOST_:
            _nG = _nClass;
            break;
        case _ABC_LOGITBOOST_:
            _nG = _nClass*_nClass;
            break;
        default:
            cout << "This type of tree: " << _treeType << " has not been implmented!" << endl;
            exit(-1);
    }
}
void treeScalarDirection::resetRootNode() {
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

treeScalarDirection::NODE::NODE(dataManager* data, treeScalarDirection* tree, int leftPoint, int rightPoint) {
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
void treeScalarDirection::NODE::saveNode(QDataStream& fileDBReader){
    fileDBReader<<_iDimension<<_cut<<_f<<_isInternal<<_tree->_treeClass;
    if(_isInternal){
        fileDBReader<<qint8('(');
        _leftChildNode->saveNode(fileDBReader);
        fileDBReader<<qint8(')')<<qint8('+')<<qint8('(');
        _rightChildNode->saveNode(fileDBReader);
        fileDBReader<<qint8(')');
    }
}

void treeScalarDirection::saveTree(QDataStream& fileDBReader){
    _rootNode->saveNode(fileDBReader);
    fileDBReader<<qint8(';');
}
treeScalarDirection::NODE::~NODE() {
    if (_leftChildNode)
        delete _leftChildNode;
    _leftChildNode = NULL;
    if (_rightChildNode)
        delete _rightChildNode;
    _rightChildNode = NULL;
}

void treeScalarDirection::eval(float * pnt, float * direction) {
    NODE* n = _rootNode;
    while (n->_isInternal) {
        if (pnt[n->_iDimension] <= n->_cut) {
            n = n->_leftChildNode;
        } else{
            n = n->_rightChildNode;
        }
    }
    (*direction)=n->_f;
}

//initialization of a given node

void treeScalarDirection::initNode() {
    NODE* n = _rootNode;
    //calculate the gain
    n->_isInternal = false;
    n->_ableSplit  = true ;
    n->_additiveGain = 0. ;

    n->_nodeSumG=0;
    n->_nodeSumH=0;

    for (int iPoint = 0; iPoint < _nEvents; iPoint++) {
        n->_nodeSumG += _data->_lossGradient[iPoint * _nG + _treeClass];
        n->_nodeSumH += _data->_lossHessian[iPoint * _nG + _treeClass];
    }

    n->_iDimension=0;
    n->calculateF();
}

void treeScalarDirection::NODE::splitNode() {
    int splitPoint;
    //initialization
    float  maxGain   = _nodeGain;
    int maxDimension = 0 ;
    int maxI         = -1;
    int    iDimension;
    float  bestC = 0.;
    int shift=_tree->_minimumNodeSize;
    for (iDimension = 0; iDimension < _tree->_nDimension; iDimension++) {
        leftSumG= 0.;
        leftSumH= 0.;
        rightSumG= _nodeSumG;
        rightSumH= _nodeSumH;

        splitPoint = _leftPoint;
        //get rid of the same value elements
        float  postX = _data->_trainX[_data->_dataIndex[iDimension][_leftPoint + 1] * _tree->_nDimension + iDimension];
        float  cVal  = 1.e300   ;
        for (splitPoint = _leftPoint; splitPoint < _leftPoint + shift; splitPoint++) {
            float  g, h;
            g = _data->_lossGradient[_data->_dataIndex[iDimension][splitPoint]  * _tree->_nG + _tree->_treeClass];
            h = _data->_lossHessian[_data->_dataIndex[iDimension][splitPoint]  * _tree->_nG + _tree->_treeClass];
            leftSumG += g;
            leftSumH += h;
            rightSumG -= g;
            rightSumH -= h;
        }
        for (splitPoint = _leftPoint + shift; splitPoint <=_rightPoint-shift; splitPoint++) {
            float  x = _data->_trainX[_data->_dataIndex[iDimension][splitPoint] * _tree->_nDimension + iDimension];
            float  g, h;
            g = _data->_lossGradient[_data->_dataIndex[iDimension][splitPoint]  * _tree->_nG + _tree->_treeClass];
            h = _data->_lossHessian[_data->_dataIndex[iDimension][splitPoint]  * _tree->_nG + _tree->_treeClass];
            leftSumG += g;
            leftSumH += h;
            rightSumG -= g;
            rightSumH -= h;
            postX = _data->_trainX[_data->_dataIndex[iDimension][splitPoint + 1] * _tree->_nDimension + iDimension];
            if (x == postX)
                continue;
            postX = x                   ;
            cVal  = 0.5 * (x + postX)   ;
            float  gain=-1.;

            float  newgain=0.;
            switch (_tree->_treeType) {
                case _LOGITBOOST_:
                case _ABC_LOGITBOOST_:
                    if (leftSumH > 0)
                        newgain += leftSumG * leftSumG / leftSumH;
                    if (rightSumH > 0)
                        newgain += rightSumG * rightSumG / rightSumH;
                    break;
                case _MART_:
                    newgain+= leftSumG * leftSumG+rightSumG * rightSumG;
                    break;
                default:
                    cout << "Tree Type has not been implemented for this case: " << _tree->_treeType << endl;
                    exit(-1);
            }
            if (newgain > gain)
                gain = newgain;
            if (gain >= maxGain) {
                bestC     = cVal ;
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
//        if (t->_nodeSumH < 0.00001) {
//            float  g = 0., h = 0.;
//            for (int ip = t->_leftPoint; ip <= t->_rightPoint; ip++) {
//                cout << _data->_lossGradient[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass] << ", " << _data->_lossHessian[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass] << endl;
//                g += _data->_lossGradient[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass];
//                h += _data->_lossHessian[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass];
//            }
//            cout << "g= " << g << ", h= " << h <<", nodeG= "<<t->_nodeSumG<<", nodeH= "<<t->_nodeSumH<< endl;
//            exit(0);
//        }
        
        t = new NODE(_data, _tree, maxI + 1, _rightPoint);
        _rightChildNode = t;
        t->_nodeSumG= rightSumG1;
        t->_nodeSumH= rightSumH1;
        t->_iDimension=_iDimension;
        t->calculateF();
        
//        if (t->_nodeSumH < 0.00001) {
//            float  g = 0., h = 0.;
//            for (int ip = t->_leftPoint; ip <= t->_rightPoint; ip++) {
//                cout << _data->_lossGradient[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass] << ", " << _data->_lossHessian[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass] << endl;
//                g += _data->_lossGradient[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass];
//                h += _data->_lossHessian[_data->_dataIndex[_iDimension][ip]* _tree->_nG + _tree->_treeClass];
//            }
//            cout << "g= " << g << ", h= " << h <<", nodeG= "<<t->_nodeSumG<<", nodeH= "<<t->_nodeSumH<< endl;
//            exit(0);
//        }
    }
    _ableSplit = false;
}

void treeScalarDirection::reArrange(NODE* node, int splitPoint) {
    int iDimension = node->_iDimension;
    for (int id = 0; id < _nDimension; id++) {
        _indexMask->reset(node->_leftPoint, node->_rightPoint);
        //there is no need to rearrange current split dimension
        if (id == iDimension)
            continue;
        for (int ip = splitPoint + 1; ip <= node->_rightPoint; ip++) {
            if (!_indexMask->set(_data->_dataReverseIndex[id][_data->_dataIndex[iDimension][ip]])) {
                cout << "Error: reArrange fails to set index Mask! "<< endl;
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
                    cout << "right>node->_rightPoint" << endl;
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

treeScalarDirection::~treeScalarDirection() {
    delete _indexMask;
    delete  _rootNode;
}

bool treeScalarDirection::NODE::printInfo(const char* indent, bool last) {
    bool ret;
    char leftS[1024];
    char rightS[1024];  
    if (_isInternal) {
        if(last){
            sprintf(rightS,"%s",indent);
            printf("%s|-x%.2d<=%.6f f= %f (%d,%d) nodeGain=%f additiveGain=%f G=%f H=%f \n",rightS,_iDimension,_cut,_f, _leftPoint, _rightPoint,_nodeGain,_additiveGain,_nodeSumG,_nodeSumH);
//            if(_nodeSumH<0.00001){
//                float  g=0.,h=0.;
//                for(int ip=_leftPoint;ip<=_rightPoint;ip++){
//                    cout<<_data->_lossGradient[_data->_dataReverseIndex[_iDimension][ip]]<<", "<<_data->_lossHessian[_data->_dataReverseIndex[_iDimension][ip]]<<endl;
//                    g+=_data->_lossGradient[_data->_dataReverseIndex[_iDimension][ip]];
//                    h+=_data->_lossHessian[_data->_dataReverseIndex[_iDimension][ip]];
//                }
//                cout<<"g= "<<g<<", h= "<<h<<endl;
//                exit(0);
//            }    
            sprintf(leftS,"%s|  ",indent);
            ret = _leftChildNode->printInfo(leftS, true);
            ret = _rightChildNode->printInfo(leftS, false);
        }
        else{
            sprintf(rightS,"%s",indent);
            printf("%s+-x%.2d<=%.6f f= %f (%d,%d) nodeGain=%f additiveGain= %f G=%f H=%f \n",rightS,_iDimension,_cut,_f, _leftPoint, _rightPoint,_nodeGain,_additiveGain,_nodeSumG,_nodeSumH);
            sprintf(leftS,"%s  ",indent);
            ret = _leftChildNode->printInfo(leftS, true);
            ret = _rightChildNode->printInfo(leftS, false);
        }
    } else{
        if(!last){
            sprintf(rightS,"%s",indent);
            printf("%s+-f= %f (%d,%d) nodeGain=%f additiveGain=%f G=%f H=%f \n",rightS, _f, _leftPoint, _rightPoint,_nodeGain,_additiveGain,_nodeSumG,_nodeSumH);
        }
        else{
            sprintf(rightS,"%s",indent);
            printf("%s|-f= %f (%d,%d) nodeGain=%f additiveGain=%f G=%f H=%f \n",rightS, _f, _leftPoint, _rightPoint,_nodeGain,_additiveGain,_nodeSumG,_nodeSumH);
        }
        return true;
    }
    return ret;
}
void treeScalarDirection::NODE::calculateF(){
    switch(_tree->_treeType) {
        case _LOGITBOOST_:
        case _ABC_LOGITBOOST_:
            if (_nodeSumH > 0)
                _nodeGain = _nodeSumG * _nodeSumG / _nodeSumH;
            else
                _nodeGain = 0.;
            break;
        case _MART_:
            _nodeGain=_nodeSumG*_nodeSumG;
            break;
        default:
            cout << "Tree Type has not been implemented for this case: " << _tree->_treeType << endl;
            exit(-1);
    }
    
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

void treeScalarDirection::buildDirection() {
//    if (_treeClass == 5) {
//        for (int ix = 0; ix < _data->_nTrainEvents; ix++) {
//            cout << _data->_lossGradient[ix * _nG + _treeClass] << ", " << _data->_lossHessian[ix * _nG + _treeClass] << endl;
//        }
//    }
    _round++;
    for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
        memcpy(_data->_dataIndex[iDimension], _data->_dataIndex0[iDimension], _nEvents * sizeof (int));
        memcpy(_data->_dataReverseIndex[iDimension], _data->_dataReverseIndex0[iDimension], _nEvents * sizeof (int));
    }
    resetRootNode();
    initNode();
    for (int ileaf = 0; ileaf < _nLeaves - 1; ileaf++){
        NODE* node = NULL    ;
        float  bestGain = -1.;
        _rootNode->bestNode(node, bestGain);
        if (bestGain == -1.)
            break;
        if (node->_leftChildNode)
            node->_isInternal = true;
    }
//    _rootNode->printInfo("",true);
//    saveTree("test.dat");
//    cout<<endl;
//    cout<<"++++++++++ "<<_round<<" treeClass= "<<_treeClass<<" +++++++++++++++"<<endl;
//
//    exit(0);
//    if (_treeClass == 5) {
//        cout<<_rootNode->_nodeSumG<<", "<<_rootNode->_nodeSumH<<endl;
//        exit(0);
//    }
//    exit(0);
}

void treeScalarDirection::NODE::bestNode(NODE*& n, float & gain) {
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