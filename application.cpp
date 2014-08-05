/* 
 * File:   application.cpp
 * Author: kaiwu
 * 
 * Created on August 4, 2014, 7:42 PM
 */

#include "application.h"

application::application(const char* fileDBName) {
    _fileDBName=fileDBName;
    bool ret=init();
    if(!ret){
        cout<<"Can not initialize the classifier!"<<endl;
        exit(0);
    }
}

application::application(const application& orig) {
}

application::~application() {
    for(int iTree=0;iTree<_nTrees;iTree++){
        delete _bootedTrees[iTree];
    }
    delete[] _bootedTrees;
    delete[] _direction;
    if(_treeType==directionFunction::_ABC_LOGITBOOST_)
        delete[] _baseClass;
}
void application::eval(double* pnt, double* f){
    for(int iClass=0;iClass<_nClass;iClass++)
        f[iClass]=0.;
    switch(_treeType) {
        case directionFunction::_ABC_LOGITBOOST_:
            for(int iIteration=0;iIteration<_nMaximumIteration;iIteration++){
                evalS(pnt, iIteration);
                for(int iClass=0;iClass<_nClass;iClass++)
                    f[iClass]-=_shrinkage*_direction[iClass];
            }
            break;
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            for(int iIteration=0;iIteration<_nMaximumIteration;iIteration++){
                evalS(pnt, iIteration);
                for(int iClass=0;iClass<_nClass;iClass++)
                    f[iClass]+=_shrinkage*_direction[iClass];
            }
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            for(int iIteration=0;iIteration<_nMaximumIteration;iIteration++){
                evalV(pnt, iIteration);
                for(int iClass=0;iClass<_nClass;iClass++)
                    f[iClass]+=_shrinkage*_direction[iClass];
            }
            break;
        default:
            cout << "This tree type " << _treeType << " has not been implemented!" << endl;
            break;
    }
}
void application::evalS(double* pnt,int iIteration){
    for (int iClass = 0; iClass < _nClass; iClass++)
        _direction[iClass] = 0.;
    struct _NODE_ *n;
    for(int iTree = 0; iTree < _nTrees; iTree++) {
        n = _bootedTrees[iIteration * _nTrees + iTree];
        while (n->_isInternal) {
            if (pnt[n->_iDimension] <= n->_cut) {
                n = n->_leftChildNode;
            } else {
                n = n->_rightChildNode;
            }
        }
        _direction[n->_class%_nClass]=n->_f;
    }
    if(_treeType==directionFunction::_ABC_LOGITBOOST_){
        for(int iClass=0;iClass<_nClass;iClass++){
            if(iClass!=_baseClass[iIteration]){
                _direction[_baseClass[iIteration]]-=_direction[iClass];
            }
        }
    }
}
void application::evalV(double* pnt,int iIteration) {
    for (int iClass = 0; iClass < _nClass; iClass++)
        _direction[iClass] = 0.;
    struct _NODE_ *n = _bootedTrees[iIteration];
//    _bootedTrees[iIteration]->printInfo("",true);
    while (n->_isInternal) {
        if (pnt[n->_iDimension] <= n->_cut) {
            n = n->_leftChildNode;
        } else {
            n = n->_rightChildNode;
        }
    }
    double f;
    int workingClass, workingClass1, workingClass2;
    f = n->_f;
    workingClass = n->_class;
    workingClass1 = workingClass / _nClass;
    workingClass2 = workingClass % _nClass;
    switch (_treeType) {
        case directionFunction::_SLOGITBOOST_:
            for (int iClass = 0.; iClass < _nClass; iClass++) {
                if (iClass == workingClass)
                    _direction[iClass] = (_nClass - 1.) * f;
                else
                    _direction[iClass] = -f;
            }
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
            for (int iClass = 0.; iClass < _nClass; iClass++) {
                if (iClass == workingClass1)
                    _direction[iClass] = f;
                else if (iClass == workingClass2)
                    _direction[iClass] = -f;
                else
                    _direction[iClass] = 0;
            }
            break;
        default:
            cout << "This tree type " << _treeType << " has not been implemented!" << endl;
            break;
    }
}
void application::buildTree(const char* tree, struct _NODE_* root){
    char op;
    int  iDimension,iclass;
    float cut,f;
    bool internal;
    struct _NODE_* curNode;
    root->_leftChildNode=NULL;
    root->_rightChildNode=NULL;
    root->_parentNode=NULL;
    curNode=root;
    stringstream ss;
    ss.str(tree);
    ss >> iDimension >> cut>>f>>internal>>iclass;
    
    root->_cut = cut;
    root->_iDimension = iDimension;
    root->_f = f;
    root->_isInternal=internal;
    root->_class=iclass;
    while(ss.good()){
        ss>>op;
        switch(op){
            case '(':
                curNode->_leftChildNode=new struct _NODE_;
                curNode->_leftChildNode->_leftChildNode=NULL;
                curNode->_leftChildNode->_rightChildNode=NULL;
                curNode->_leftChildNode->_parentNode=curNode;
                curNode->_rightChildNode=new struct _NODE_;
                curNode->_rightChildNode->_leftChildNode=NULL;
                curNode->_rightChildNode->_rightChildNode=NULL;
                curNode->_rightChildNode->_parentNode=curNode;
                curNode=curNode->_leftChildNode;
                break;
                
            case ')':
                curNode=curNode->_parentNode;
                break;
                
            case '+':
                curNode=curNode->_rightChildNode;
                ss>>op;
                break;
        }
        if(op=='(') {
            ss >> iDimension >> cut>>f>>internal>>iclass;
            curNode->_cut = cut;
            curNode->_iDimension = iDimension;
            curNode->_f = f;
            curNode->_isInternal=internal;
            curNode->_class=iclass;
        }
    }
}
bool application::init(){
    bool ret=true;
    _fileDB=new ifstream(_fileDBName.c_str(),ifstream::in);
    if(!_fileDB->good()){
        cout<<"Can not open "<<_fileDBName<<endl;
        return false;
    }
    int k;
    (*_fileDB)>>k;
    _treeType=directionFunction::_TREE_TYPE_(k);
    (*_fileDB)>>_nClass>>_nVariable>>_nMaximumIteration>>_shrinkage;
    _direction=new double[_nClass];
    if(_treeType==directionFunction::_ABC_LOGITBOOST_){
        _baseClass=new int[_nMaximumIteration];
    }
    
    switch(_treeType) {
        case directionFunction::_ABC_LOGITBOOST_:
            _nTrees=_nClass-1;
            break;
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            _nTrees=_nClass;
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            _nTrees=1;
            break;
        default:
            cout << "This tree type " << _treeType << " has not been implemented!" << endl;
            break;
    }
    _bootedTrees=new struct _NODE_*[_nTrees*_nMaximumIteration];
    for(int iT=0;iT<_nTrees*_nMaximumIteration;iT++){
        _bootedTrees[iT]=new struct _NODE_;
    }
    string tree;
    //skip the first endl
    getline(*_fileDB,tree);
    for(int iIteration=0;iIteration<_nMaximumIteration;iIteration++) {
        if (_treeType == directionFunction::_ABC_LOGITBOOST_) {
            getline(*_fileDB, tree);
            _baseClass[iIteration] = atoi(tree.c_str());
        }
//        cout<<"------------- iIteration= "<<iIteration<<" baseClass= "<<_baseClass[iIteration]<<" -------------"<<endl;
        for(int iTree=0;iTree<_nTrees;iTree++){
            getline(*_fileDB,tree);
            buildTree(tree.c_str(),_bootedTrees[iIteration*_nTrees+iTree]);
//            cout<<"["<<iIteration*_nTrees+iTree<<"]"<<endl;
//            _bootedTrees[iIteration*_nTrees+iTree]->printInfo("",true);
        }
//        cout<<endl;
    }
    return ret;
}