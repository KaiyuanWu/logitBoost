/* 
 * File:   LossFunction.cpp
 * Author: kaiwu
 * 
 * Created on June 14, 2014, 1:55 PM
 */

#include "LossFunction.h"

LossFunction::LossFunction(_LOSSTYPE lossType,int nDimension,int nClass) {
    _lossType=lossType;
    _nDimension=nDimension;
    _nClass=nClass;
    _MIN_HESSIAN_=1.e-300;
    _C=0.;
}

LossFunction::~LossFunction() {
    
}

double LossFunction::loss(double*f,int y,double& gradient, double& hessian,int c1,int c2){
    double ret=0.;
    switch(_lossType){
        case _S_NEWTON_:
            ret=lossSNewton(f,y,c1,gradient,hessian);
            break;
        case _ABCLOGIT_NEWTON_:
            if(c2==-1){
                cout<<"Please check the input for the loss Type= _ABCLOGIT_NEWTON_: the second class c2= "<<c2<<endl;
                exit(-1);
            }
            ret=lossabcLogitNewton(f,y,c1,c2,gradient,hessian);
            break;
        default:
            cout<<"Sorry the _lossType= "<<_lossType<<" has not been implemented!"<<endl;
            exit(-1);
            break;
    }
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return ret;
}

double LossFunction::lossSNewton(double* f, int y, int c, double& gradient, double& hessian){
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
        loss=1.0e300;
    gradient=_nClass*pc;
    if(y==c) gradient-=_nClass;
    gradient=-1.*gradient;
    hessian=_nClass*_nClass*pc*(1.-pc);
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return loss;
}

double LossFunction::lossabcLogitNewton(double* f, int y, int c1,int c2, double& gradient, double& hessian){
    double loss=0.;
    double maxf=f[0];
    for(int iClass=1;iClass<_nClass;iClass++){
        if(f[iClass]>maxf)
            maxf=f[iClass];
    }
    double pc1=1.,pc2=1., sumF=0.,t=1.,py=1.;
    for(int iClass=0;iClass<_nClass;iClass++){
        t=exp(f[iClass]-maxf);
        if(iClass==c1)
            pc1=t;
        if(iClass==c2)
            pc2=t;
        if(iClass==y)
            py=t;
        sumF+=t;
    }
    pc1/=sumF;
    pc2/=sumF;
    py/=sumF;
    if(py>0)
        loss=-log(py);
    else
        loss=100;
    gradient=0;
    if(y==c1){
        gradient+=pc1-1;
    }
    else{
        gradient+=pc1;
    }
    if(y==c2)
        gradient-=(pc2-1);
    else
        gradient-=pc2;
    gradient+=2*_C*(f[c1]-f[c2]);
    gradient*=-1;
    hessian=pc1*(1.-pc1)+pc2*(1.-pc2)+2.*pc1*pc2;
    hessian+=4*_C;
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return loss;

}

