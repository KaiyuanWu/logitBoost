/* 
 * File:   linearSearch.cpp
 * Author: kaiwu
 * 
 * Created on December 4, 2013, 9:43 AM
 */

#include "linearSearch.h"
#include "treeScalarDirection.h"
#include "treeVectorDirection.h"
linearSearch::linearSearch(dataManager*  data,int nLeaves,double shrinkage,int minimumNodeSize,_TREE_TYPE_ treeType,directionFunction::_GAIN_TYPE_ gainType){
    _data=data;
    _nLeaves=nLeaves;
    _treeType=treeType;
    _gainType=gainType;
    switch(_treeType){
        case _SCALE_TREE_:
            _df=new treeScaleDirection(data,_nLeaves);
            break;
        case _VECTOR_TREE_:
            _df=new treeVectorDiretion(data,_nLeaves);
        default:
            cout<<"Has not been implemented!"<<endl;
            exit(-1);
            break;
    }
    if(minimumNodeSize==0)
        _df->_minimumNodeSize=1;
    else 
        _df->_minimumNodeSize=_data->_nTrainEvents*0.01*minimumNodeSize;     
    _df->_shrinkage=_shrinkage;
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