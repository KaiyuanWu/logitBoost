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
    virtual void eval(double* pnt, double* direction)= 0;
    virtual void buildDirection()= 0;
    virtual ~directionFunction();
    
public:
    int _nDimension;
    int _nClass;
    int _nEvents;
    int _round;
    int _nLeaves;
    double _shrinkage;
    double _zMax;
    //bagging probability
    double _minimumNodeSize;
private:
    
};

#endif	/* DIRECTIONFUNCTION_H */

