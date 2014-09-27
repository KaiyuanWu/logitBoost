/* 
 * File:   test.cpp
 * Author: kaiwu
 * 
 * Created on August 6, 2014, 2:49 PM
 */
#include "stdlib.h"
#include "string.h"
#include "test.h"

test::test(int iTask,int iDataset) {
    _iTask=iTask;
    _iDataset=iDataset;
}

test::~test() {
}
int test::getBestAccuracy(string& modelFile,string& dataFile, float & bestAccuracy){
    int bestIteration=-1;
    bestAccuracy=-1;
    application app(modelFile.c_str());
    int nVariable=app._nVariable;
    int nClass=app._nClass;
    int nMaxIteration=app._nMaximumIteration;
    int nEvents=getNEvents(dataFile);
    float * x=new float [nVariable*nEvents];
    int* l=new int[nEvents];
    float * f=new float [nClass*nEvents];
    //reset f
    memset(f,0,sizeof(float )*nClass*nEvents);
    //read in data
    ifstream infile(dataFile.c_str(),ifstream::in);
    for(int iEvent=0;iEvent<nEvents;iEvent++){
        for(int iV=0;iV<nVariable;iV++)
            infile>>x[iEvent*nVariable+iV];
        infile>>l[iEvent];
    }
    for(int iIteration=0;iIteration<nMaxIteration;iIteration++){
        float  accuracy=0.;
        float  maxF=-1.e300;
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
//             if(iIteration<20){
//                 for(int iClass=0;iClass<nClass;iClass++){
//                     cout<<f[iEvent*nClass+iClass]<<" ";
//                 }
//                 cout<<maxI<<", "<<maxF<<": "<<l[iEvent]<<endl;
//             }
        }
        accuracy/=nEvents;
//        if(iIteration<20){
//            cout<<accuracy<<endl;
//        }
        if(accuracy>bestAccuracy){
            bestAccuracy=accuracy;
            bestIteration=iIteration;
        }
    }
//    exit(0);
    delete[] x;
    delete[] l;
    delete[] f;
    return bestIteration;
}
float  test::getBestAccuracy(string& modelFile,string& dataFile,int iIteration){
    application app(modelFile.c_str());
    int nVariable=app._nVariable;
    int nClass=app._nClass;
    int nMaxIteration=app._nMaximumIteration;
    if(iIteration>=nMaxIteration)
        iIteration=nMaxIteration-1;
    int nEvents=getNEvents(dataFile);
    float * x=new float [nVariable*nEvents];
    int* l=new int[nEvents];
    float * f=new float [nClass*nEvents];
    //reset f
    memset(f,0,sizeof(float )*nClass*nEvents);
    //read in data
    ifstream infile(dataFile.c_str(),ifstream::in);
    for(int iEvent=0;iEvent<nEvents;iEvent++){
        for(int iV=0;iV<nVariable;iV++)
            infile>>x[iEvent*nVariable+iV];
        infile>>l[iEvent];
    }

    float  accuracy = 0.;
    float  maxF = -1.e300;
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
void test::start() {
    string modelFileNamePrefix="/Users/kaiwu/Documents/MulticlassLossFunctionComparision/files/output/expAllinOneProgram/ws/";
    string dataFileNamePrefix="/Users/kaiwu/Documents/MulticlassLossFunctionComparision/files/datasets/trValTestData/";
    char* datasets[] = {"dna", "letter", "optdigits", "pendigits", "satImage", "shuttle"};
    int nClass[] = {3, 26, 10, 10, 6, 7};
    int nDatasets = sizeof (datasets) / sizeof (char*);
    int nLeaves[] = {2, 8, 11, 21};
    int nL = sizeof (nLeaves) / sizeof (int);
    char* exes[] = {"abcLogit", "aosoLogit", "logit", "mart", "slogit"};
    int nExes = sizeof (exes) / sizeof (char*);
    int nTasks = 60;
    int nMax = 1000;

    if(_iTask>nTasks){
        cout<<"This task is not available! "<<_iTask<<endl;
        exit(-1);
    }
    
    float * bestValAccuracy = new float [nL * nExes];
    int* bestValIteration = new int[nL * nExes];
    int* bestValLeaves = new int[nExes];
    float * testAccuracy = new float [nExes];
    
    for(int iExe=0;iExe<nExes;iExe++){
        for(int iL=0;iL<nL;iL++){
            char s[1024];
            sprintf(s,"%sTr_%d.dat%s_shrinkage0.100000_nLeave%d_minimumNodeSize1_nMaxIteration%d.model",
                    datasets[_iDataset],_iTask,exes[iExe],nLeaves[iL],nMax);
            string modelFile=modelFileNamePrefix+s;
            sprintf(s,"%sVal_%d.dat",datasets[_iDataset],_iTask);
            string dataFile=dataFileNamePrefix+s;
//            cout<<dataFile<<endl;
            bestValIteration[iExe*nL+iL]=getBestAccuracy(modelFile,dataFile,bestValAccuracy[iExe*nL+iL]);
            cout<<exes[iExe]<<" nLeave= "<<nLeaves[iL]<<", bestValAccuracy= "<<bestValAccuracy[iExe*nL+iL]<<", bestValIteration= "<<bestValIteration[iExe*nL+iL]<<endl;
        }
    }
    for(int iExe=0;iExe<nExes;iExe++){
        float  bestA=-1;
        for(int iL=0;iL<nL;iL++){
            if(bestValAccuracy[iExe*nL+iL]>bestA){
                bestA=bestValAccuracy[iExe*nL+iL];
                bestValLeaves[iExe]=iL;
            }
        }
    }
    
    cout<<"testing ...."<<endl;
    for(int iExe=0;iExe<nExes;iExe++) {
        char s[1024];
        sprintf(s, "%sTr_%d.dat%s_shrinkage0.100000_nLeave%d_minimumNodeSize1_nMaxIteration%d.model",
                datasets[_iDataset], _iTask, exes[iExe], nLeaves[bestValLeaves[iExe]], nMax);
        string modelFile = modelFileNamePrefix + s;
        sprintf(s, "%sT_%d.dat", datasets[_iDataset], _iTask);
        string dataFile = dataFileNamePrefix + s;
        testAccuracy[iExe]=getBestAccuracy(modelFile,dataFile,bestValIteration[iExe*nL+bestValLeaves[iExe]]);
        cout<<exes[iExe]<<" nLeave= "<<nLeaves[bestValLeaves[iExe]]<<", bestValAccuracy= "<<testAccuracy[iExe]<<", bestValIteration= "<<bestValIteration[iExe*nL+bestValLeaves[iExe]]<<endl;
        
    }
    
    delete[] bestValAccuracy;
    delete[] bestValIteration;
    delete[] bestValLeaves;
    delete[] testAccuracy;
    char outFileName[1024];
    sprintf(outFileName,"task%d_data%d.dat",_iTask,_iDataset);
    ofstream outf(outFileName,ofstream::out);
    for(int iExe=0;iExe<nExes;iExe++){
        outf<<nLeaves[bestValLeaves[iExe]]<<" "<<bestValIteration[bestValLeaves[iExe]*nExes+iExe]<<" "<<testAccuracy[iExe]<<endl;
    }
    outf.close();
}
