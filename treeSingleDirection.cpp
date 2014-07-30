/* 
 * File:   treeSingleDirection.cpp
 * Author: kaiwu
 * 
 * Created on March 16, 2014, 11:22 PM
 */

#include <bitset>
#include "treeSingleDirection.h"

//arrays for node splitting
double *leftSumG, *leftSumH;
double *rightSumG, *rightSumH;
double *leftSumG1, *leftSumH1;
double *rightSumG1, *rightSumH1;
double leftLoss,rightLoss;
double leftLoss1,rightLoss1;
int *classN;
treeSingleDirection::treeSingleDirection(dataManager* data, int nLeaves) {
    _data = data;
    _nLeaves = nLeaves;
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
    
    leftSumG =new double[_nClass];
    leftSumH =new double[_nClass];
    rightSumG=new double[_nClass];
    rightSumH=new double[_nClass];
    leftSumG1 =new double[_nClass];
    leftSumH1 =new double[_nClass];
    rightSumG1=new double[_nClass];
    rightSumH1=new double[_nClass];
}

void treeSingleDirection::resetRootNode() {
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

treeSingleDirection::NODE::NODE(dataManager* data, treeSingleDirection* tree, int leftPoint, int rightPoint,double loss) {
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
    _nodeSumG=new double[_tree->_nClass];
    _nodeSumH=new double[_tree->_nClass];
    _purity=0.;
}

treeSingleDirection::NODE::~NODE() {
    if (_leftChildNode)
        delete _leftChildNode;
    _leftChildNode = NULL;
    if (_rightChildNode)
        delete _rightChildNode;
    _rightChildNode = NULL;
    delete[] _nodeSumG;
    delete[] _nodeSumH;
}

double treeSingleDirection::evalp(double* s,int& iClass,bool printPurity) {
    NODE* n = _rootNode;
    while (n->_isInternal) {
        if (s[n->_iDimension] <= n->_cut) {
            if (printPurity&&!n->_leftChildNode->_isInternal) {
                cout << "Left " << n->_leftV << ", Right " << n->_rightV << " cut " << n->_cut << " x " << s[n->_iDimension] << "(" << n->_leftPoint << ", " << n->_rightPoint << ")" << endl;
                cout << "Purity= " << n->_leftChildNode->_purity << ", base class= " << n->_leftChildNode->_class << ", c= " << iClass << endl;
            }
            n = n->_leftChildNode;
        } else{
            if (printPurity&&!n->_rightChildNode->_isInternal) {
                cout << "Left " << n->_leftV << ", Right " << n->_rightV << " cut " << n->_cut << " x " << s[n->_iDimension] << "(" << n->_leftPoint << ", " << n->_rightPoint << ")" << endl;
                cout << "Purity= " << n->_rightChildNode->_purity << ", base class= " << n->_rightChildNode->_class << ", c= " << iClass << endl;
            }
            n = n->_rightChildNode;
        }
    }
    iClass=n->_class;
    return n->_f    ;
}

void treeSingleDirection::initNode() {
    NODE* n = _rootNode;
    //calculate the gain
    n->_isInternal = false;
    n->_ableSplit  = true ;
    n->_additiveGain = 0. ;
    n->_nodeLoss=0;
    memset(n->_nodeSumG,0,sizeof(double)*_nClass);
    memset(n->_nodeSumH,0,sizeof(double)*_nClass);

    for (int iPoint = 0; iPoint < _nEvents; iPoint++) {
        n->_nodeLoss+=_data->_loss[iPoint];
        for(int iClass=0;iClass<_nClass;iClass++){
            n->_nodeSumG[iClass]+=_data->_lossGradient[iPoint*_nClass+iClass];
            n->_nodeSumH[iClass]+=_data->_lossHessian[iPoint*_nClass+iClass];
        }
    }
    n->_iDimension=0;
    n->selectBestClass();
}

void treeSingleDirection::NODE::splitNode() {
    int splitPoint;
    //initialization
    double maxGain   = _nodeGain;
    int maxDimension = 0 ;
    int maxI         = -1;
    int    iDimension;
    double bestC = 0.;
    double bestLeftV,bestRightV;
//////////////////////////////////////////////////
//    cout<<"++++++++++++ Node("<<_leftPoint<<", "<<_rightPoint<<") Loss= "<<_nodeLoss<<" +++++++++++++++"<<endl;
//    for(int iClass=0;iClass<_tree->_nClass;iClass++){
//        cout<<"["<<_nodeSumG[iClass]<<", "<<_nodeSumH[iClass]<<"]"<<endl;
//    }
//    double* nodeSumG=new double[_tree->_nClass];
//    double* nodeSumH=new double[_tree->_nClass];
//    memset(nodeSumG,0,sizeof(double)*_tree->_nClass);
//    memset(nodeSumH,0,sizeof(double)*_tree->_nClass);
//    for(int iPoint=_leftPoint;iPoint<=_rightPoint;iPoint++){
//        for(int iClass=0;iClass<_tree->_nClass;iClass++){
//            nodeSumG[iClass]+=_data->_lossGradient[_tree->_dataIndex[0][iPoint]*_tree->_nClass+iClass];
//            nodeSumH[iClass]+=_data->_lossHessian[_tree->_dataIndex[0][iPoint]*_tree->_nClass+iClass];
//        }
//    }
//    cout<<"-----------------------"<<endl;
//    for(int iClass=0;iClass<_tree->_nClass;iClass++){
//        cout<<"["<<nodeSumG[iClass]<<", "<<nodeSumH[iClass]<<"]"<<endl;
//    }
//    
//////////////////////////////////////////////////    
    
    int shift=_tree->_minimumNodeSize;
    
    for (iDimension = 0; iDimension < _tree->_nDimension; iDimension++) {
        memset(leftSumG, 0,sizeof(double)*_tree->_nClass);
        memset(leftSumH, 0,sizeof(double)*_tree->_nClass);
        leftLoss=0;
        rightLoss= _nodeLoss;
        memcpy(rightSumG , _nodeSumG,sizeof(double)*_tree->_nClass);
        memcpy(rightSumH , _nodeSumH,sizeof(double)*_tree->_nClass);
        
        splitPoint = _leftPoint;
        //get rid of the same value elements
        double postX = _data->_trainX[_data->_dataIndex[iDimension][_leftPoint + 1] * _tree->_nDimension + iDimension];
        double cVal  = 1.e300   ;
        double leftV,rightV     ;
        for (splitPoint = _leftPoint; splitPoint < _leftPoint + shift; splitPoint++) {
            for (int iClass = 0; iClass < _tree->_nClass; iClass++) {
                double g,h;
                g=_data->_lossGradient[_data->_dataIndex[iDimension][splitPoint]*_tree->_nClass+iClass];
                h=_data->_lossHessian[_data->_dataIndex[iDimension][splitPoint]*_tree->_nClass+iClass];
                
                leftSumG[iClass]  += g; leftSumH[iClass]  += h;
                rightSumG[iClass] -= g; rightSumH[iClass] -= h;
            }
            double l=_data->_loss[_data->_dataIndex[iDimension][splitPoint]];
            leftLoss  +=l; 
            rightLoss -=l;
        }
        for (splitPoint = _leftPoint + shift; splitPoint <=_rightPoint-shift; splitPoint++) {
            double x = _data->_trainX[_data->_dataIndex[iDimension][splitPoint] * _tree->_nDimension + iDimension];
//            cout<<"Cut at "<<splitPoint<<": ";
            for (int iClass = 0; iClass < _tree->_nClass; iClass++) {
                double  g, h;
                g=_data->_lossGradient[_data->_dataIndex[iDimension][splitPoint]*_tree->_nClass+iClass];
                h=_data->_lossHessian[_data->_dataIndex[iDimension][splitPoint]*_tree->_nClass+iClass];
                leftSumG[iClass]  += g; leftSumH[iClass]  += h;
                rightSumG[iClass] -= g; rightSumH[iClass] -= h;
//                cout<<"["<<g<<", "<<h<<"] ("<<leftSumH[iClass]<<", "<<rightSumH[iClass]<<"), ";
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
            for(int ic=0;ic<_tree->_nClass;ic++) {
                if (leftSumH[ic] == 0.||rightSumH[ic]==0.) {
                    cout << "Something is wrong! Hessian is 0 in [treeSingleDirection::NODE::splitNode]" << endl;
                    exit(0);
                }
                double newgain=(leftSumG[ic] *leftSumG[ic]*rightSumH[ic]+rightSumG[ic]*rightSumG[ic]*leftSumH[ic])/(leftSumH[ic]*rightSumH[ic]);
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
        _leftV      = bestLeftV;
        _rightV     = bestRightV;
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

void treeSingleDirection::reArrange(NODE* node, int splitPoint) {
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

treeSingleDirection::~treeSingleDirection() {
    delete _indexMask;
    delete  _rootNode;
    
    delete[] leftSumG ;
    delete[] leftSumG1;
    delete[] leftSumH ;
    delete[] leftSumH1;
    delete[] rightSumG;
    delete[] rightSumG1;
    delete[] rightSumH ;
    delete[] rightSumH1;
}

void treeSingleDirection::updateDirection() {
    for (int iEvent = 0; iEvent < _data->_nTrainEvents; iEvent++)
        eval(_data->_trainX + iEvent * _data->_nDimension, _data->_trainDescendingDirection + iEvent * _data->_nClass, iEvent,true);
    for (int iEvent = 0; iEvent < _data->_nTestEvents; iEvent++)
        eval(_data->_testX + iEvent * _data->_nDimension, _data->_testDescendingDirection + iEvent * _data->_nClass, iEvent,false);
    resetRootNode();
}

bool treeSingleDirection::NODE::printInfo(const char* indent, bool last) {
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
            printf("%s+-f= %f (%d,%d) Base Class= %d Purity=%f\n",rightS, _f, _leftPoint, _rightPoint,_class,_purity);
        }
        else{
            sprintf(rightS,"%s",indent);
            printf("%s|-f= %f (%d,%d) Base Class= %d Purity=%f\n",rightS, _f, _leftPoint, _rightPoint,_class,_purity);
        }
        return true;
    }
    return ret;
}
void treeSingleDirection::NODE::selectBestClass(){
    double maxG=0.;
    int maxIndex=-1;
    for(int iClass=0;iClass<_tree->_nClass;iClass++){
        if(_nodeSumH[iClass]==0.){
            cout<<"Something is wrong! [treeSingleDirection::NODE::selectBestClass]"<<endl;
            exit(0);
        }
        double gain=_nodeSumG[iClass]*_nodeSumG[iClass]/_nodeSumH[iClass];
        if(gain>=maxG){
            maxG=gain      ;
            maxIndex=iClass;
        }
    }
    if(maxIndex!=-1)
        _class=maxIndex;
    else
        _class=0;
    _nodeGain=maxG;
    
    for(int ip=_leftPoint;ip<=_rightPoint;ip++)
        if(_data->_trainClass[_data->_dataIndex[_iDimension][ip]]==_class)
            _purity++;
    _purity/=(_rightPoint-_leftPoint+1.);
    if(maxIndex==-1){
        cout<<"All probabilities of this node are either 1 or 0!"<<endl;
        _ableSplit=false;
        //possible guess for the node values
        //original default value is 0
//        if(_purity==1)
//            _f=1.;
//        else
//            _f=-1.;
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

void treeSingleDirection::eval(double* pnt, double* direction, int iEvent,bool isTrain) {
    double f;
    int workingClass;
    f = evalp(pnt,workingClass);
    
    for (int iClass = 0.; iClass < _nClass; iClass++) {
        if(iClass==workingClass)
            direction[iClass]=(_nClass-1.)*f;
        else
            direction[iClass]=-f;
    }
}

void treeSingleDirection::buildDirection() {
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

void treeSingleDirection::NODE::bestNode(NODE*& n, double& gain) {
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