/* 
 * File:   LossFunction.cpp
 * Author: kaiwu
 * 
 * Created on June 14, 2014, 1:55 PM
 */

#include "LossFunction.h"

LossFunction::LossFunction(directionFunction::_TREE_TYPE_ treeType,int nDimension,int nClass) {
    _treeType=treeType;
    _nDimension=nDimension;
    _nClass=nClass;
    _MIN_HESSIAN_=1.e-300;
}

LossFunction::~LossFunction() {
    
}

double LossFunction::loss(double*f,int y,double& gradient, double& hessian,int c1){
    double ret=0.;
    int class1=c1/_nClass;
    int class2=c1%_nClass;
    switch(_treeType){
        case directionFunction::_ABC_LOGITBOOST_:
        case directionFunction::_AOSO_LOGITBOOST_:
            ret=lossabcLogitNewton(f,y,class1,class2,gradient,hessian);
            if (class1!=class2&&hessian < _MIN_HESSIAN_)
                hessian = _MIN_HESSIAN_;
            if(class1==class2){
                gradient=0.;
                hessian =0.;
            }
            break;
        case directionFunction::_SLOGITBOOST_:
            ret=lossSNewton(f,y,class1,gradient,hessian);
            break;
        case directionFunction::_MART_:
        case directionFunction::_LOGITBOOST_:
            ret=lossNewton(f,y,class1,gradient,hessian);
            break;
        default:
            cout<<"Sorry the _lossType= "<<_treeType<<" has not been implemented!"<<endl;
            exit(-1);
            break;
    }
    return ret;
}

double LossFunction::lossNewton(double* f, int y, int c, double& gradient, double& hessian){
    double loss=0.;
    double maxf=f[0];
    for(int iClass=1;iClass<_nClass;iClass++){
        if(f[iClass]>maxf)
            maxf=f[iClass];
    }
    double pc=1.,sumF=0.,t=1.,py=1.;
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
    gradient=0;
    if(y==c)
        gradient=1-pc;
    else
        gradient=-pc;

    hessian=pc*(1.-pc);
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return loss;
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
        loss=1.e300;
    gradient=0;
    if(y==c1)
        gradient+=pc1-1;
    else
        gradient+=pc1;
    if(y==c2)
        gradient-=(pc2-1);
    else
        gradient-=pc2;
    gradient*=-1;
    hessian=pc1*(1.-pc1)+pc2*(1.-pc2)+2.*pc1*pc2;
    if(hessian<_MIN_HESSIAN_)
        hessian=_MIN_HESSIAN_;
    return loss;

}

