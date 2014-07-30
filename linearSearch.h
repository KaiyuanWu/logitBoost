/* 
 * File:   linearSearch.h
 * Author: kaiwu
 *
 * Created on December 4, 2013, 9:43 AM
 */

#ifndef LINEARSEARCH_H
#define	LINEARSEARCH_H

#include "dataManager.h"
#include "directionFunction.h"

//linearSearch class will encapsulate the minimization
//it will store all the training parameters
//it will organize the train iteration
class linearSearch {
public:
    enum _TREE_TYPE_{_SCALE_TREE_=0, _VECTOR_TREE_};
    linearSearch(dataManager*  data,int nLeaves,  double shrinkage,int minimumNodeSize, _TREE_TYPE_ treeType,directionFunction::_GAIN_TYPE_ gainType);
    virtual ~linearSearch();
    
    
    double minimization(int iRound=0);
    //parameter for training
    int _nLeaves;
    double _shrinkage;
    _TREE_TYPE_ _treeType;
    
    directionFunction* _df;
    dataManager*  _data;
    void updateDirection();
    
private:
    directionFunction::_GAIN_TYPE_ _gainType;
};

#endif	/* LINEARSEARCH_H */

