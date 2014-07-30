/* 
 * File:   linearSearch.cpp
 * Author: kaiwu
 * 
 * Created on December 4, 2013, 9:43 AM
 */

#include "linearSearch.h"
#include "treeScalarDirection.h"
#include "treeVectorDirection.h"
linearSearch::linearSearch(dataManager*  data,int nLeaves,double shrinkage,int minimumNodeSize,directionFunction::_TREE_TYPE_ treeType){
    _data=data;
    _treeType=treeType;
    _nLeaves=nLeaves;
    _minimumNodeSize=minimumNodeSize;
    _shrinkage=shrinkage;
    switch(_treeType){
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            _nDirection=_data->_nClass;
            _df=new treeScalarDiretion*[_nDirection];
            for(int id=0;id<_nDirection;id++)
                _df[id]=new treeScalarDiretion(data,_nLeaves,_minimumNodeSize,_treeType,id);
            break;
        case directionFunction::_ABC_LOGITBOOST_:
            _nDirection=_data->_nClass*_data->_nClass;
            _df=new treeScalarDiretion*[_nDirection];
            for(int id=0;id<_nDirection;id++)
                _df[id]=new treeScalarDiretion(data,_nLeaves,_minimumNodeSize,_treeType,id);
            break;
            break;
        case directionFunction::_AOSO_LOGITBOOST_:
        case directionFunction::_SLOGITBOOST_:
            
            break;
        default:
            cout<<"Has not been implemented!"<<endl;
            exit(-1);
            break;
    }
 

}
linearSearch::~linearSearch() {
    if(_df)
        delete _df;
}
double linearSearch::minimization(int iRound){
    double ret=0.;
    _df->buildDirection();
    updateDirection();
    _data->increment(_shrinkage, _df, iRound);
    ret = _data->_trainAccuracy;
    return ret;
}

void linearSearch::updateDirection(){
    
}