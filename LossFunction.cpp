/* 
 * File:   LossFunction.cpp
 * Author: kaiwu
 * 
 * Created on June 14, 2014, 1:55 PM
 */

#include "LossFunction.h"

LossFunction::LossFunction(_LOSSTYPE lossType,_PHIFUNCTION phiFunction,int nDimension,int nClass) {
    _lossType=lossType;
    _phiFunction=phiFunction;
    _nDimension=nDimension;
    _nClass=nClass;
    _MIN_HESSIAN_=1.e-20;
    _C=0.;
}

LossFunction::~LossFunction() {
    
}

double LossFunction::loss(double*f,int y,int c,double& gradient, double& hessian){
    double ret=0.;
    switch(_lossType){
        case _ONE_VS_ONE:
            ret=loss1vs1(f,y,c,gradient,hessian);
            break;
        case _ONE_VS_ALL:
            ret=loss1vsA(f,y,c,gradient,hessian);
            break;
        case _CONSTRAIN_COMPARE_:
            ret=lossConstrainCompare(f,y,c,gradient,hessian);
            break;
        case _COUPLEDLOGISTIC_:
            ret=lossCoupledLogistic(f,y,c,gradient,hessian);
            break;
        default:
            cout<<"Sorry the _lossType= "<<_lossType<<" has not been implemented!"<<endl;
            break;
    }
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return ret;
}
//loss function is
//sum_{k!=c}{phi(fc-fk)}
double LossFunction::loss1vs1(double*f, int y,int c,double& gradient, double& hessian){
    double ret=0.;
    gradient=0.;
    hessian=0.;
    switch(_phiFunction){
        case _EXP_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiExp(f[y]-f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiExpGradient(f[y]-f[iClass]);
                }
                gradient*=_nClass;
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiExpHessian(f[y]-f[iClass]);
                }
                hessian*=_nClass*_nClass;
            }
            else{
                //sum the gradient
                gradient=-phiExpGradient(f[y]-f[c])*_nClass;
                hessian=phiExpHessian(f[y]-f[c])*_nClass*_nClass;
            }
            break;
        case _LIKELIHOOD_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiLikelihood(f[y]-f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiLikelihoodGradient(f[y]-f[iClass]);
                }
                gradient*=_nClass;
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiLikelihoodHessian(f[y]-f[iClass]);
                }
                hessian*=_nClass*_nClass;
            }
            else{
                //sum the gradient
                gradient=-phiLikelihoodGradient(f[y]-f[c])*_nClass;
                hessian=phiLikelihoodHessian(f[y]-f[c])*_nClass*_nClass;
            }
            break;
        case _L2_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiL2(f[y]-f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiL2Gradient(f[y]-f[iClass]);
                }
                gradient*=_nClass;
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                    hessian+=phiL2Hessian(f[y]-f[iClass]);
                }
                hessian*=_nClass*_nClass;
            }
            else{
                //sum the gradient
                gradient=-phiL2Gradient(f[y]-f[c])*_nClass;
                hessian=phiL2Hessian(f[y]-f[c])*_nClass*_nClass;
            }
            break;            
        default:
            cout<<"Sorry the _phiFunction= "<<_phiFunction<<" has not been implemented!"<<endl;
            break;
    }
    gradient=-1.*gradient;
    return ret;
}
double LossFunction::loss1vsA(double*f, int y,int c,double& gradient, double& hessian){
    double ret=0.;
    gradient=0.;
    hessian=0.;
    switch(_phiFunction){
        case _EXP_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiExp(-f[iClass]);
                else
                    ret+=phiExp(f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiExpGradient(-f[iClass]);
                    else
                        gradient+=phiExpGradient(f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiExpHessian(-f[iClass]);
                    else
                        hessian+=phiExpHessian(f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            else{
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        gradient+=phiExpGradient(-f[iClass]);
                    else if(iClass==y)
                        gradient-=phiExpGradient(f[iClass]);
                    else
                        gradient-=phiExpGradient(-f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        hessian+=phiExpHessian(-f[iClass]);
                    else if(iClass==y)
                        hessian+=phiExpHessian(f[iClass]);
                    else
                        hessian+=phiExpHessian(-f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            break;
        case _LIKELIHOOD_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiLikelihood(-f[iClass]);
                else
                    ret+=phiLikelihood(f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiLikelihoodGradient(-f[iClass]);
                    else
                        gradient+=phiLikelihoodGradient(f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiLikelihoodHessian(-f[iClass]);
                    else
                        hessian+=phiLikelihoodHessian(f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            else{
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        gradient+=phiLikelihoodGradient(-f[iClass]);
                    else if(iClass==y)
                        gradient-=phiLikelihoodGradient(f[iClass]);
                    else
                        gradient-=phiLikelihoodGradient(-f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        hessian+=phiLikelihoodHessian(-f[iClass]);
                    else if(iClass==y)
                        hessian+=phiLikelihoodHessian(f[iClass]);
                    else
                        hessian+=phiLikelihoodHessian(-f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            break;
        case _L2_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiL2(-f[iClass]);
                else
                    ret+=phiL2(f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiL2Gradient(-f[iClass]);
                    else
                        gradient+=phiL2Gradient(f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiL2Hessian(-f[iClass]);
                    else
                        hessian+=phiL2Hessian(f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            else{
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        gradient+=phiL2Gradient(-f[iClass]);
                    else if(iClass==y)
                        gradient-=phiL2Gradient(f[iClass]);
                    else
                        gradient-=phiL2Gradient(-f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        hessian+=phiL2Hessian(-f[iClass]);
                    else if(iClass==y)
                        hessian+=phiL2Hessian(f[iClass]);
                    else
                        hessian+=phiL2Hessian(-f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            break;
        default:
            cout<<"Sorry the _phiFunction= "<<_phiFunction<<" has not been implemented!"<<endl;
            break;
    }
    gradient=-1.*gradient;
    return ret;
}
double LossFunction::lossConstrainCompare(double*f, int y,int c,double& gradient, double& hessian){
    double ret=0.;
    gradient=0.;
    hessian=0.;
    switch(_phiFunction){
        case _EXP_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiExp(-f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiExpGradient(-f[iClass]);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiExpHessian(-f[iClass]);
                }
            }
            else{
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        gradient+=phiExpGradient(-f[iClass]);
                    else if(iClass!=y)
                        gradient-=phiExpGradient(-f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        hessian+=phiExpHessian(-f[iClass]);
                    else if(iClass!=y)
                        hessian+=phiExpHessian(-f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            break;
        case _LIKELIHOOD_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiLikelihood(-f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiLikelihoodGradient(-f[iClass]);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiLikelihoodHessian(-f[iClass]);
                }
            }
            else{
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        gradient+=phiLikelihoodGradient(-f[iClass]);
                    else if(iClass!=y)
                        gradient-=phiLikelihoodGradient(-f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        hessian+=phiLikelihoodHessian(-f[iClass]);
                    else if(iClass!=y)
                        hessian+=phiLikelihoodHessian(-f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            break;
        case _L2_:
            //sum the loss
            for(int iClass=0;iClass<_nClass;iClass++){
                if(iClass!=y)
                    ret+=phiL2(-f[iClass]);
            }
            if(y==c){
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        gradient+=phiL2Gradient(-f[iClass]);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y)
                        hessian+=phiL2Hessian(-f[iClass]);
                }
            }
            else{
                //sum the gradient
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        gradient+=phiL2Gradient(-f[iClass]);
                    else if(iClass!=y)
                        gradient-=phiL2Gradient(-f[iClass])*(_nClass-1);
                }
                //sum the hessian
                for(int iClass=0;iClass<_nClass;iClass++){
                    if(iClass!=y&&iClass!=c)
                        hessian+=phiL2Hessian(-f[iClass]);
                    else if(iClass!=y)
                        hessian+=phiL2Hessian(-f[iClass])*(_nClass-1)*(_nClass-1);
                }
            }
            break;
        default:
            cout<<"Sorry _phiFunction= "<<_phiFunction<<" has not been implemented!"<<endl;
            break;
    }
    gradient=-1.*gradient;
    return ret;
}

double LossFunction::phiExp(double x){
    return exp(-x);
}
double LossFunction::phiExpGradient(double x){
    return -exp(-x);
}
double LossFunction::phiExpHessian(double x){
    return exp(-x);
}
double LossFunction::phiLikelihood(double x){
    return log(1+exp(-x));
}

double LossFunction::phiLikelihoodGradient(double x){
    return -exp(-x)/(1+exp(-x));
}
double LossFunction::phiLikelihoodHessian(double x){
    return exp(-x)/((1+exp(-x))*(1+exp(-x)));
}

double LossFunction::phiL2(double x){
    return (1-x)*(1-x);
}
double LossFunction::phiL2Gradient(double x){
    return -2*(1-x);
}
double LossFunction::phiL2Hessian(double x){
    return 2;
}

double LossFunction::lossCoupledLogistic(double* f, int y, int c, double& gradient, double& hessian){
    double loss=0.;
    double maxf=f[0];
    for(int iClass=1;iClass<_nClass;iClass++){
        if(f[iClass]>maxf)
            maxf=f[iClass];
    }
    double pc, sumF=0.,t,py;
    for(int iClass=0;iClass<_nClass;iClass++){
        t=exp(f[iClass]-maxf);
        if(iClass==c)
            pc=t;
        if(iClass==y)
            py=t;
        sumF+=t;
    }
    pc/=sumF;
    py/=sumF;
    if(py>0)
        loss=-log(py);
    else
        loss=100;
    gradient=_nClass*pc;
    if(y==c) gradient-=_nClass;
//    gradient+=2*_C*_nClass*f[y];
//    double sumF2=0.;
//    for(int i=0;i<_nClass;i++)
//        sumF2+=f[i]*f[i];
//    gradient+=_C*2*_nClass*f[c]/(1-sumF2);
    gradient=-1.*gradient;
    hessian=_nClass*_nClass*pc*(1.-pc);
//    hessian+=2*_C*_nClass*(_nClass-1);
//    hessian+=_C*(_nClass*(_nClass-1)/(1.-sumF2)+4*_nClass*_nClass*f[c]*f[c]/((1.-sumF2)*(1.-sumF2)));
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return loss;
}