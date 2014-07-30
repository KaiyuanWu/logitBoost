/* 
 * File:   treeVectorDiretion.cpp
 * Author: kaiwu
 * 
 * Created on March 16, 2014, 11:22 PM
 */

#include <bitset>
#include "treeVectorDirection.h"
#include "treeScalarDirection.h"

//arrays for node splitting
treeVectorDiretion::treeVectorDiretion(dataManager* data,int nLeaves, int minimumNodeSize, _TREE_TYPE_ treeType) {
    _data = data;
    _nLeaves = nLeaves;
    _minimumNodeSize=minimumNodeSize;
    _treeType=treeType;
    if (_nLeaves < 2) {
        cout << "Number of terminate nodes is " << _nLeaves << ", change it to 2!" << endl;
        _nLeaves = 2;
    }
    _nClass     = _data->_nClass    ;
    _nDimension = _data->_nDimension;
    _nEvents = _data->_nTrainEvents ;
    _round = 0;

    _rootNode      = new NODE(_data, this,  0, _nEvents - 1,1.0e300);
    _rootNode->_isInternal = true;
    _indexMask     = new bitArray(_nEvents)   ;
    
    _zMax = 4.   ;
}

void treeVectorDiretion::resetRootNode() {
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

treeVectorDiretion::NODE::NODE(dataManager* data, treeVectorDiretion* tree, int leftPoint, int rightPoint,double loss) {
    _additiveGain = 0.;
    _data = data;
    _tree = tree;
    _f    = 0.;
    _leftPoint  = leftPoint;
    _rightPoint = rightPoint;
    if (_rightPoint - _leftPoint >= 2*_tree->_minimumNodeSize)
         _ableSplit = true;
    else
        _ableSplit = false;
    _leftChildNode  = NULL;
    _rightChildNode = NULL;
    _isInternal = false;
    //initialize the sumH, sumH array
    _nodeLoss=loss;
    
    switch(_tree->_treeType){
        case _AOSO_LOGITBOOST_:
            _nG=_nClass*_nClass;
            break;
        case _SLOGITBOOST_:
            _nG=_nClass;
            break;
        default:
            cout<<"This type of tree: "<<_treeType<<" has not been implmented!"<<endl;
            exit(-1);
    }
    _nodeSumG=new double[_nG];
    _nodeSumH=new double[_nG];
    leftSumG =new double[_nG];
    leftSumH =new double[_nG];
    rightSumG=new double[_nG];
    rightSumH=new double[_nG];
    
    leftSumG1 =new double[_nG];
    leftSumH1 =new double[_nG];
    rightSumG1=new double[_nG];
    rightSumH1=new double[_nG];
}

treeVectorDiretion::NODE::~NODE() {
    if (_leftChildNode)
        delete _leftChildNode;
    _leftChildNode = NULL;
    if (_rightChildNode)
        delete _rightChildNode;
    _rightChildNode = NULL;
    delete[] _nodeSumG;
    delete[] _nodeSumH;
    
    delete[] leftSumG ;
    delete[] leftSumG1;
    delete[] leftSumH ;
    delete[] leftSumH1;
    
    delete[] rightSumG;
    delete[] rightSumG1;
    delete[] rightSumH ;
    delete[] rightSumH1;
}

double treeVectorDiretion::evalp(double* s,int& iClass) {
    NODE* n = _rootNode;
    while (n->_isInternal) {
        if (s[n->_iDimension] <= n->_cut) {
            n = n->_leftChildNode;
        } else{
            n = n->_rightChildNode;
        }
    }
    iClass=n->_class;
    return n->_f    ;
}

void treeVectorDiretion::initNode() {
    NODE* n = _rootNode;
    //calculate the gain
    n->_isInternal = false;
    n->_ableSplit  = true ;
    n->_additiveGain = 0. ;
    n->_nodeLoss=0;
    memset(n->_nodeSumG,0,_nG);
    memset(n->_nodeSumH,0,_nG);

    for (int iPoint = 0; iPoint < _nEvents; iPoint++) {
        n->_nodeLoss+=_data->_loss[iPoint];
        for(int iG=0;iG<_nG;iG++){
            n->_nodeSumG[iG]+=_data->_lossGradient[iPoint*_nG+iG];
            n->_nodeSumH[iG]+=_data->_lossHessian[iPoint*_nG+iG];
        }
    }
    n->_iDimension=0;
    n->selectBestClass();
}

void treeVectorDiretion::NODE::splitNode() {
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
        memset(leftSumG, 0,sizeof(double)*_nG);
        memset(leftSumH, 0,sizeof(double)*_nG);
        leftLoss=0;
        rightLoss= _nodeLoss;
        memcpy(rightSumG , _nodeSumG,sizeof(double)*_nG);
        memcpy(rightSumH , _nodeSumH,sizeof(double)*_nG);
        
        splitPoint = _leftPoint;
        //get rid of the same value elements
        double postX = _data->_trainX[_data->_dataIndex[iDimension][_leftPoint + 1] * _tree->_nDimension + iDimension];
        double cVal  = 1.e300   ;
        double leftV,rightV     ;
        for (splitPoint = _leftPoint; splitPoint < _leftPoint + shift; splitPoint++) {
            for (int iG = 0; iG < _nG; iG++) {
                double g,h;
                g=_data->_lossGradient[_data->_dataIndex[iDimension][splitPoint]*_nG+iG];
                h=_data->_lossHessian[_data->_dataIndex[iDimension][splitPoint]*_nG+iG];
                
                leftSumG[iG]  += g; leftSumH[iG]  += h;
                rightSumG[iG] -= g; rightSumH[iG] -= h;
            }
            double l=_data->_loss[_data->_dataIndex[iDimension][splitPoint]];
            leftLoss  +=l; 
            rightLoss -=l;
        }
        for (splitPoint = _leftPoint + shift; splitPoint <=_rightPoint-shift; splitPoint++) {
            double x = _data->_trainX[_data->_dataIndex[iDimension][splitPoint] * _tree->_nDimension + iDimension];
//            cout<<"Cut at "<<splitPoint<<": ";
            for (int iG = 0; iG < _nG; iG++) {
                double  g, h;
                g=_data->_lossGradient[_data->_dataIndex[iDimension][splitPoint]*_nG+iG];
                h=_data->_lossHessian[_data->_dataIndex[iDimension][splitPoint]*_nG+iG];
                leftSumG[iG]  += g; leftSumH[iG]  += h;
                rightSumG[iG] -= g; rightSumH[iG] -= h;
            }
//            cout<<endl;
            double l=_data->_loss[_data->_dataIndex[iDimension][splitPoint]];
            leftLoss +=l;
            rightLoss -=l;
//            cout<<endl;
            postX = _data->_trainX[_data->_dataIndex[iDimension][splitPoint + 1] * _tree->_nDimension + iDimension];
            if (x == postX)
                continue;
            leftV=x;rightV=postX;
            postX = x                   ;
            cVal  = 0.5 * (x + postX)   ;
            double gain=-1.;
            for(int iG=0;iG<_nG;iG++) {
                if (leftSumH[iG] == 0.||rightSumH[iG]==0.) {
                    cout << "Something is wrong! Hessian is 0 in [treeVectorDiretion::NODE::splitNode]" << endl;
                    exit(0);
                }
                double newgain=(leftSumG[iG] *leftSumG[iG]*rightSumH[iG]+rightSumG[iG]*rightSumG[iG]*leftSumH[iG])/(leftSumH[iG]*rightSumH[iG]);
                if(newgain>gain)
                    gain=newgain;
            }
            if (gain >= maxGain) {
                bestC     = cVal ;
                bestLeftV = leftV;
                bestRightV=rightV;
                maxGain   = gain;
                maxI      = splitPoint;
                maxDimension = iDimension;
                
                leftLoss1=leftLoss;
                rightLoss1=rightLoss;
                memcpy(leftSumG1 ,leftSumG, sizeof(double)*_tree->_nClass);
                memcpy(leftSumH1 ,leftSumH, sizeof(double)*_tree->_nClass);                
                memcpy(rightSumG1 ,rightSumG, sizeof(double)*_tree->_nClass);
                memcpy(rightSumH1 ,rightSumH, sizeof(double)*_tree->_nClass);
            }
        }
    }
    // a better split has been found
    if (maxI != -1) {
        _iDimension = maxDimension;
        _cut        = bestC;
        _additiveGain =maxGain-_nodeGain;
        _tree->reArrange(this, maxI);
        
        NODE* t = new NODE(_data, _tree, _leftPoint, maxI,leftLoss1);
        _leftChildNode = t;
        memcpy(t->_nodeSumG ,leftSumG1, sizeof(double)*_tree->_nClass);
        memcpy(t->_nodeSumH ,leftSumH1, sizeof(double)*_tree->_nClass);
        t->_iDimension=_iDimension;
        t->selectBestClass();

        t = new NODE(_data, _tree, maxI + 1, _rightPoint,rightLoss1);
        _rightChildNode = t;
        memcpy(t->_nodeSumG ,rightSumG1, sizeof(double)*_tree->_nClass);
        memcpy(t->_nodeSumH ,rightSumH1, sizeof(double)*_tree->_nClass);
        t->_iDimension=_iDimension;
        t->selectBestClass();
        //check the node is correctly split
//        cout<<"Box("<<_leftPoint<<", "<<_rightPoint<<"): "<<endl;
//        for(int iClass=0;iClass<_tree->_nClass;iClass++){
//            double lg=_leftChildNode->_nodeSumG[iClass];
//            double lh=_leftChildNode->_nodeSumH[iClass];
//            double rg=_rightChildNode->_nodeSumG[iClass];
//            double rh=_rightChildNode->_nodeSumH[iClass];
//            cout<<_nodeSumG[iClass]<<" "<<lg<<"+"<<rg<<"= "<<lg+rg<<"; "<<_nodeSumH[iClass]<<" "<<lh<<"+"<<rh<<"= "<<lh+rh<<endl;          
//        }
    }
//    cout<<"_additiveGain= "<<_additiveGain<<endl;
    _ableSplit = false;
}

void treeVectorDiretion::reArrange(NODE* node, int splitPoint) {
    int iDimension = node->_iDimension;
    for (int id = 0; id < _nDimension; id++) {
        _indexMask->reset(node->_leftPoint, node->_rightPoint);
        //there is no need to rearrange current split dimension 
        if (id == iDimension)
            continue;
        for (int ip = splitPoint + 1; ip <= node->_rightPoint; ip++) {
            if (!_indexMask->set(_data->_dataReverseIndex[id][_data->_dataIndex[iDimension][ip]])) {
                cout << "Fail to set the index mask!"<< endl;
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

treeVectorDiretion::~treeVectorDiretion() {
    delete _indexMask;
    delete  _rootNode;
}

bool treeVectorDiretion::NODE::printInfo(const char* indent, bool last) {
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
            printf("%s+-f= %f (%d,%d)\n",rightS, _f, _leftPoint, _rightPoint);
        }
        else{
            sprintf(rightS,"%s",indent);
            printf("%s|-f= %f (%d,%d)\n",rightS, _f, _leftPoint, _rightPoint);
        }
        return true;
    }
    return ret;
}
void treeVectorDiretion::NODE::selectBestClass(){
    double maxG=0.;
    int maxIndex=-1;
    for(int iG=0;iG<_nG;iG++){
        if(_nodeSumH[iG]<=0.){
            cout<<"Something is wrong! [treeVectorDiretion::NODE::selectBestClass]"<<endl;
            continue;
        }
        double gain=_nodeSumG[iG]*_nodeSumG[iG]/_nodeSumH[iG];
        if(gain>=maxG){
            maxG=gain      ;
            maxIndex=iG;
        }
    }
    if(maxIndex!=-1)
        _class=maxIndex;
    else
        _class=0;
    _nodeGain=maxG;
    
    if(maxIndex==-1){
        cout<<"All probabilities of this node are either 1 or 0!"<<endl;
        _ableSplit=false;
        _f=0.;
    }
    else {
        _f = _nodeSumG[_class] /_nodeSumH[_class];
        if(!(_f==_f)){
            _f=0                ;
        }
        if (!(_f <= _tree->_zMax))
            _f = _tree->_zMax;
        if (!(_f >= -_tree->_zMax))
            _f = -_tree->_zMax;
    }
    //cout<<"Node("<<_leftPoint<<", "<<_rightPoint<<") Loss= "<<_nodeLoss<<endl;
}

void treeVectorDiretion::eval(double* pnt, double* direction) {
    double f;
    int workingClass, workingClass1, workingClass2;
    f = evalp(pnt,workingClass);
    workingClass1=workingClass/_nClass;
    workingClass2=workingClass%_nClass;
    switch(_treeType){
        case _AOSO_LOGITBOOST_:
            for (int iClass = 0.; iClass < _nClass; iClass++) {
                if (iClass == workingClass)
                    direction[iClass] = (_nClass - 1.) * f;
                else
                    direction[iClass] = -f;
            }    
            break;
        case _SLOGITBOOST_:
            for (int iClass = 0.; iClass < _nClass; iClass++) {
                if (iClass == workingClass1)
                    direction[iClass] =  f;
                else if(iClass == workingClass2)
                    direction[iClass] = -f;
                else
                    direction[iClass] = 0;
            } 
            break;
        default:
            cout<<"This tree type "<<_treeType<<" has not been implemented!"<<endl;
            break;
    }
}

void treeVectorDiretion::buildDirection() {
    _round++;
    for (int iDimension = 0; iDimension < _nDimension; iDimension++) {
        memcpy(_data->_dataIndex[iDimension], _data->_dataIndex0[iDimension], _nEvents * sizeof (int));
        memcpy(_data->_dataReverseIndex[iDimension], _data->_dataReverseIndex0[iDimension], _nEvents * sizeof (int));
    }
    initNode();
    for (int ileaf = 0; ileaf < _nLeaves - 1; ileaf++) {
        NODE* node = NULL    ;
        double bestGain = -1.;
        _rootNode->bestNode(node, bestGain);
        if (bestGain == -1.)
            break;
        if (node->_leftChildNode)
            node->_isInternal = true;
    }
//    _rootNode->printInfo("",true);
//    cout<<"+++++++++++++++++++++++++"<<endl;
//    exit(0);
}

void treeVectorDiretion::NODE::bestNode(NODE*& n, double& gain) {
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