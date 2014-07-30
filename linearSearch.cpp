/* 
 * File:   linearSearch.cpp
 * Author: kaiwu
 * 
 * Created on December 4, 2013, 9:43 AM
 */

#include "linearSearch.h"
#include "treeScaleDirection.h"
#include "treeVectorDirection.h"
linearSearch::linearSearch(dataManager*  data,int nLeaves, double shrinkage, int minimumNodeSize){
    _data=data  ;
    _nLeaves=nLeaves;
    _shrinkage=shrinkage;
    _df=new treeScaleDirection(data,_nLeaves);
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