/* 
 * File:   application.cpp
 * Author: kaiwu
 * 
 * Created on August 4, 2014, 7:42 PM
 */

#include <Qt/qdatastream.h>

#include "application.h"
#include "stdlib.h"

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
void application::eval(float * pnt,int iIteration){
    switch(_treeType) {
        case directionFunction::_ABC_LOGITBOOST_:
            evalS(pnt, iIteration);
            for(int iClass=0;iClass<_nClass;iClass++)
                _direction[iClass]*=-1.;
            break;
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            evalS(pnt, iIteration);
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            evalV(pnt, iIteration);
            break;
        default:
            cout << "This tree type " << _treeType << " has not been implemented!" << endl;
            break;
    }
}
void application::eval(float * pnt, float * f,int nIteration){
    for(int iClass=0;iClass<_nClass;iClass++)
        f[iClass]=0.;
    if(nIteration<=0||nIteration>=_nMaximumIteration)
        nIteration=_nMaximumIteration;
    
    switch(_treeType) {
        case directionFunction::_ABC_LOGITBOOST_:
            for(int iIteration=0;iIteration<nIteration;iIteration++){
                evalS(pnt, iIteration);
                for(int iClass=0;iClass<_nClass;iClass++)
                    f[iClass]-=_shrinkage*_direction[iClass];
            }
            break;
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            for(int iIteration=0;iIteration<nIteration;iIteration++){
                evalS(pnt, iIteration);
                for(int iClass=0;iClass<_nClass;iClass++)
                    f[iClass]+=_shrinkage*_direction[iClass];
            }
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            for(int iIteration=0;iIteration<nIteration;iIteration++){
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
void application::evalS(float * pnt,int iIteration){
    for (int iClass = 0; iClass < _nClass; iClass++)
        _direction[iClass] = 0.;
    struct _NODE_ *n;
    for(int iTree = 0; iTree < _nTrees; iTree++) {
        n = _bootedTrees[iIteration * _nTrees + iTree];
        while (n->_isInternal) {
            if (pnt[n->_iDimension] <= 1.0000005*n->_cut) {
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
void application::evalV(float * pnt,int iIteration) {
    for (int iClass = 0; iClass < _nClass; iClass++)
        _direction[iClass] = 0.;
    struct _NODE_ *n = _bootedTrees[iIteration];
//    _bootedTrees[iIteration]->printInfo("",true);
    while (n->_isInternal) {
        if (pnt[n->_iDimension] <= 1.0000005*n->_cut) {
            n = n->_leftChildNode;
        } else {
            n = n->_rightChildNode;
        }
    }
    float  f;
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
void application::buildTree(QDataStream& fileDBReader, struct _NODE_* root){
    qint8 op;
    int  iDimension,iclass;
    float cut,f;
    bool internal;
    struct _NODE_* curNode;
    root->_leftChildNode=NULL;
    root->_rightChildNode=NULL;
    root->_parentNode=NULL;
    curNode=root;
    fileDBReader >> iDimension >> cut>>f>>internal>>iclass;
    
    root->_cut = cut;
    root->_iDimension = iDimension;
    root->_f = f;
    root->_isInternal=internal;
    root->_class=iclass;
    while(1){
        fileDBReader>>op;
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
                fileDBReader>>op;
                break;
            case ';':
                break;      
            default:
                cout<<"Error: parse the database!";
                exit(0);
                break;
        }
        if(op==';')
            break;
        if(op=='(') {
            fileDBReader >> iDimension >> cut>>f>>internal>>iclass;
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
    QFile fileDB(_fileDBName.c_str());
    fileDB.open(QIODevice::ReadOnly);
    if(!fileDB.good()){
        cout<<"Can not open "<<_fileDBName<<endl;
        return false;
    }
    QDataStream fileDBReader(&fileDB);

    int treeType,baseClass;
    fileDBReader>>treeType;
    _treeType=directionFunction::_TREE_TYPE_(treeType);
    fileDBReader>>_nClass>>_nVariable>>_nMaximumIteration>>_shrinkage;
    _direction=new float [_nClass];
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
    for(int iT=0;iT<_nTrees*_nMaximumIteration;iT++)
        _bootedTrees[iT]=new struct _NODE_;
    //skip the first endl
    for(int iIteration=0;iIteration<_nMaximumIteration;iIteration++) {
        if (_treeType == directionFunction::_ABC_LOGITBOOST_) {
            fileDBReader>>baseClass;
            _baseClass[iIteration] = baseClass;
        }
        for(int iTree=0;iTree<_nTrees;iTree++)
            buildTree(fileDBReader,_bootedTrees[iIteration*_nTrees+iTree]);
    }
    return ret;
}