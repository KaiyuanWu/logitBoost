/* 
 * File:   test.cpp
 * Author: kaiwu
 * 
 * Created on August 6, 2014, 2:49 PM
 */

#include "test.h"

test::test(int dataID) {
    _dataID=dataID;
}

test::test(const test& orig) {
}

test::~test() {
}
int test::getBestAccuracy(string& modelFile,string& dataFile, double& bestAccuracy){
    int bestIteration=-1;
    bestAccuracy=-1;
    application app(modelFile.c_str());
    int nVariable=app._nVariable;
    int nClass=app._nClass;
    int nMaxIteration=app._nMaximumIteration;
    int nEvents=getNEvents(dataFile);
    double* x=new double[nVariable*nEvents];
    int* l=new int[nEvents];
    double* f=new double[nClass*nEvents];
    //reset f
    memset(f,0,sizeof(double)*nClass*nEvents);
    //read in data
    ifstream infile(dataFile.c_str(),ifstream::in);
    for(int iEvent=0;iEvent<nEvents;iEvent++){
        for(int iV=0;iV<nVariable;iV++)
            infile>>x[iEvent*nVariable+iV];
        infile>>l[iEvent];
    }
    for(int iIteration=0;iIteration<nMaxIteration;iIteration++){
        double accuracy=0.;
        double maxF=-1.e300;
        int maxI=0;
        for(int iEvent=0;iEvent<nEvents;iEvent++){
            maxF=-1.e300;
            maxI=0;
            app.eval(x+iEvent*nVariable,iIteration);
            for(int iClass=0;iClass<nClass;iClass++){
                f[iEvent*nClass+iClass]+=app._direction[iClass];
                if(f[iEvent*nClass+iClass]>maxF){
                    maxF=f[iEvent*nClass+iClass];
                    maxI=iClass;
                }
            }
            if(maxI==l[iEvent]){
                accuracy+=1.;
            }
        }
        accuracy/=nEvents;
        if(accuracy>bestAccuracy){
            bestAccuracy=accuracy;
            bestIteration=iIteration;
        }
    }
    delete[] x;
    delete[] l;
    delete[] f;
    return bestIteration;
}
double test::getBestAccuracy(string& modelFile,string& dataFile,int iIteration){
    application app(modelFile.c_str());
    int nVariable=app._nVariable;
    int nClass=app._nClass;
    int nMaxIteration=app._nMaximumIteration;
    int nEvents=getNEvents(dataFile);
    double* x=new double[nVariable*nEvents];
    int* l=new int[nEvents];
    double* f=new double[nClass*nEvents];
    //reset f
    memset(f,0,sizeof(double)*nClass*nEvents);
    //read in data
    ifstream infile(dataFile.c_str(),ifstream::in);
    for(int iEvent=0;iEvent<nEvents;iEvent++){
        for(int iV=0;iV<nVariable;iV++)
            infile>>x[iEvent*nVariable+iV];
        infile>>l[iEvent];
    }

    double accuracy = 0.;
    double maxF = -1.e300;
    int maxI = 0;
    for (int iEvent = 0; iEvent < nEvents; iEvent++) {
        maxF = -1.e300;
        maxI = 0;
        app.eval(x + iEvent*nVariable,f+iEvent*nClass,iIteration);
        for (int iClass = 0; iClass < nClass; iClass++) {
            if (f[iEvent * nClass + iClass] > maxF) {
                maxF = f[iEvent * nClass + iClass];
                maxI = iClass;
            }
        }
        if (maxI == l[iEvent]) {
            accuracy += 1.;
        }
    }
    accuracy/=nEvents;
    delete[] x;
    delete[] l;
    delete[] f;
    return accuracy;
}
int test::getNEvents(string& file){
    int ret=0;
    ifstream infile(file.c_str(),ifstream::in);
    if(!infile.good()){
        cout<<"Can not open "<<file<<endl;
        exit(0);
    }
    while(infile.good()){
        string s;
        getline(infile,s);
        if(s.size()>0)
            ret++;
    }
    infile.close();
    return ret;
}
void test::start(){
    
}