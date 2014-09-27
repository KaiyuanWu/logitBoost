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
    linearSearch(dataManager*  data,int nLeaves,float  shrinkage,int minimumNodeSize,directionFunction::_TREE_TYPE_ treeType);
    virtual ~linearSearch();
    
    float  minimization(int iRound=0);
    //parameter for training
    directionFunction::_TREE_TYPE_ _treeType;
    
    int _nLeaves;
    float  _shrinkage;
    int _minimumNodeSize;
    
    int _nClass;
    int _nTrainEvents;
    int _nTestEvents;
    int _nDimension;
    
    int _nDirection;
    directionFunction** _df;
    dataManager*  _data;
    void updateDirection(int iRound);
    void buildDirection();
    void saveDirection(ofstream& fileDB);
    
    void updateDirection1();
    void updateDirection2();
    
    //variables for the "FAST abcLogitBoost"
    int _g,_G,_baseClass;
    float  *_F;
};

#endif	/* LINEARSEARCH_H */

