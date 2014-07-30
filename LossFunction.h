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
    enum _LOSSTYPE{_ONE_VS_ONE=0, _ONE_VS_ALL, _CONSTRAIN_COMPARE_,_COUPLEDLOGISTIC_};
    enum _PHIFUNCTION{_EXP_=0,_LIKELIHOOD_,_L2_};
    LossFunction(_LOSSTYPE lossType,_PHIFUNCTION phiFunction,int nDimension,int nClass);
    virtual ~LossFunction();
    double loss(double*f,int y,int c,double& gradient, double& hessian);
    double _C;
private:
    _LOSSTYPE _lossType;
    _PHIFUNCTION _phiFunction;
    int _nDimension;
    int _nClass;
    
    inline double phiExp(double x);
    inline double phiExpGradient(double x);
    inline double phiExpHessian(double x);
    
    inline double phiLikelihood(double x);
    inline double phiLikelihoodGradient(double x);
    inline double phiLikelihoodHessian(double x);
    
    inline double phiL2(double x);
    inline double phiL2Gradient(double x);
    inline double phiL2Hessian(double x);
    
    double loss1vs1(double*f, int y,int c,double& gradient, double& hessian);
    double loss1vsA(double*f, int y,int c,double& gradient, double& hessian);
    double lossConstrainCompare(double*f, int y,int c,double& gradient, double& hessian);
    double lossCoupledLogistic(double* f, int y, int c, double& gradient, double& hessian);
    
private:
    double _MIN_HESSIAN_;

};

#endif	/* LOSSFUNCTION_H */

