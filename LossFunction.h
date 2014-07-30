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
using namespace std;

class LossFunction {
public:
    enum _LOSSTYPE{_S_NEWTON_=0,_ABCLOGIT_NEWTON_};
    LossFunction(_LOSSTYPE lossType,int nDimension,int nClass);
    virtual ~LossFunction();
    double loss(double*f,int y,double& gradient, double& hessian, int c1, int c2=-1);
private:
    _LOSSTYPE _lossType;
    int _nDimension;
    int _nClass;
    
    double lossSNewton(double* f, int y, int c, double& gradient, double& hessian);
    double lossabcLogitNewton(double* f, int y, int c1, int c2, double& gradient, double& hessian);
    
private:
    double _MIN_HESSIAN_;
};

#endif	/* LOSSFUNCTION_H */

