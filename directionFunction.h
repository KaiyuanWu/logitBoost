/* 
 * File:   directionFunction.h
 * Author: kaiwu
 *
 * Created on December 4, 2013, 9:41 AM
 */
#ifndef  DIRECTIONFUNCTION_H
#define	 DIRECTIONFUNCTION_H
#include <string>
#include <fstream>
#include <iostream>
using namespace std;
class directionFunction {
public:
    directionFunction();
    virtual void eval(float * pnt, float * direction)= 0;
    virtual void buildDirection()= 0;
    virtual void saveTree(ofstream& fileDB)= 0;
    virtual ~directionFunction();
    enum _TREE_TYPE_{_LOGITBOOST_=0,_ABC_LOGITBOOST_,_MART_,_AOSO_LOGITBOOST_,_SLOGITBOOST_};
    
public:
    int _nDimension;
    int _nClass;
    int _nEvents;
    float  _zMax;
    int _round;
    //bagging probability
    int _nLeaves;
    int _minimumNodeSize;
    _TREE_TYPE_ _treeType;
};

#endif	/* DIRECTIONFUNCTION_H */

