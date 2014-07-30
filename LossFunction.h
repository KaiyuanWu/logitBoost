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
    double loss(double*f,int y,double& gradient, double& hessian, int c1);
private:
    directionFunction::_TREE_TYPE_ _treeType;
    int _nDimension;
    int _nClass;
    
    double lossSNewton(double* f, int y, int c, double& gradient, double& hessian);
    double lossNewton(double* f, int y, int c, double& gradient, double& hessian);
    double lossabcLogitNewton(double* f, int y, int c1, int c2, double& gradient, double& hessian);
    
private:
    double _MIN_HESSIAN_;
};

#endif	/* LOSSFUNCTION_H */

