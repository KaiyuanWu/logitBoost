/* 
 * File:   LossFunction.h
 * Author: kaiwu
 *
 * Created on June 14, 2014, 1:55 PM
 */

#ifndef LOSSFUNCTION_H
#define	LOSSFUNCTION_H
#include <iostream>
#include "math.h"
#include "directionFunction.h"
using namespace std;

class LossFunction {
public:
    LossFunction(directionFunction::_TREE_TYPE_ treeType,int nDimension,int nClass);
    virtual ~LossFunction();
    float  loss(float *f,int y,float & gradient, float & hessian, int c1);
private:
    directionFunction::_TREE_TYPE_ _treeType;
    int _nDimension;
    int _nClass;
    
    float  lossSNewton(float * f, int y, int c, float & gradient, float & hessian);
    float  lossNewton(float * f, int y, int c, float & gradient, float & hessian);
    float  lossabcLogitNewton(float * f, int y, int c1, int c2, float & gradient, float & hessian);
    
private:
    float  _MIN_HESSIAN_;
};

#endif	/* LOSSFUNCTION_H */

