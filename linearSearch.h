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
#include "treeScalarDirection.h"
#include "treeVectorDirection.h"
//linearSearch class will encapsulate the minimization
//it will store all the training parameters
//it will organize the train iteration
class linearSearch {
public:
    linearSearch(dataManager*  data,int nLeaves,double shrinkage,int minimumNodeSize,directionFunction::_TREE_TYPE_ treeType);
    virtual ~linearSearch();
    
    double minimization(int iRound=0);
    //parameter for training
    directionFunction::_TREE_TYPE_ _treeType;
    
    int _nLeaves;
    double _shrinkage;
    int _minimumNodeSize;
    
    int _nClass;
    int _nTrainEvents;
    int _nTestEvents;
    int _nDimension;
    
    int _nDirection;
    directionFunction** _df;
    dataManager*  _data;
    void updateDirection();
    void buildDirection();
    
    void updateDirection1();
    void updateDirection2();
    
    //variables for the "FAST abcLogitBoost"
    int _g,_G,_baseClass;
    double *_F;
};

#endif	/* LINEARSEARCH_H */

