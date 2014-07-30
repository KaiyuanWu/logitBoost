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
    _nLeaves=nLeaves;
    _treeType=treeType;
    _shrinkage=shrinkage;
    
    if (minimumNodeSize == 0)
        _minimumNodeSize = 1;
    else
        _minimumNodeSize = _data->_nTrainEvents * 0.01 * minimumNodeSize;    
    switch(_treeType){
        case _SCALE_TREE_:
            _df=new treeScalarDiretion(data,_nLeaves);
            break;
        case _VECTOR_TREE_:
            _df=new treeVectorDiretion(data,_nLeaves);
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