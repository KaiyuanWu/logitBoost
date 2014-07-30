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
    linearSearch(dataManager*  data,int nLeaves,  double shrinkage,int miniNodeSize=1);
    virtual ~linearSearch();
    
    double minimization(int iRound=0);
    //parameter for training
    int _nLeaves;
    double _shrinkage;
    
    directionFunction* _df;
    dataManager*  _data;
    void updateDirection();
    
private:
    
};

#endif	/* LINEARSEARCH_H */

