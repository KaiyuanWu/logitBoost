/* 
 * File:   application.cpp
 * Author: kaiwu
 * 
 * Created on August 4, 2014, 7:42 PM
 */

#include "application.h"

application::application(char* fileDBName) {
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
}

void application::eval(double* pnt, double* f){
    for(int i)
}
void application::buildTree(char* tree, struct _NODE_* root){
    char op;
    int  iDimension;
    float cut,f;
    bool internal;
    struct _NODE_* curNode;
    root->_leftChildNode=NULL;
    root->_rightChildNode=NULL;
    root->_parentNode=NULL;
    curNode=root;
    stringstream ss;
    ss.str(tree);
    ss >> iDimension >> cut>>f>>internal;
    root->_cut = cut;
    root->_iDimension = iDimension;
    root->_f = f;
    root->_isInternal=internal;
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
            ss >> iDimension >> cut>>f>>internal;
            curNode->_cut = cut;
            curNode->_iDimension = iDimension;
            curNode->_f = f;
            curNode->_isInternal=internal;
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
    (*_fileDB)>>_nClass;
    (*_fileDB)>>_nVariable;
    (*_fileDB)>>_nMaximumIteration;
    (*_fileDB)>>_ZMAX;
    _bootedTrees=new struct _NODE_*[_nMaximumIteration];
    
    return ret;
}

