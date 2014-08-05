/* 
 * File:   main.cpp
 * Author: kaiwu
 *
 * Created on July 29, 2014, 11:35 AM
 */

#include <cstdlib>
#include "train.h"
#include "directionFunction.h"
#include "application.h"

using namespace std;
//jobID
//0 --> letter
//1 --> optdigits
//2 --> pendigits
//3 --> satImage
//4 --> shuttle
//5 --> dna
void test1(int jobID,int fold,int nLeaves, int minimumNodeSize, int nMaxIteration,char* prefix){
    char fTrain[256];
    char fTest[256];
    char fOut[256];
    int nTrainEvents;
    int nTestEvents;
    int nClass;
    int nVariables;
    char* datasetNames[]={"letter","optdigits","pendigits","satImage","shuttle","dna"};
    sprintf(fTrain,"%s/datasets/splitData/%sTr_%d.dat",prefix,datasetNames[jobID],fold);
    sprintf(fTest,"%s/datasets/splitData/%sT_%d.dat",prefix,datasetNames[jobID],fold);
    sprintf(fOut,"%s/output/logitBoost_Fold%dDataset%snLeaves%dminimumNodeSize%dMaxiteration%d.dat",prefix,fold,datasetNames[jobID],nLeaves,minimumNodeSize,nMaxIteration);
    
    directionFunction::_TREE_TYPE_ treeType=directionFunction::_ABC_LOGITBOOST_;
    double shrinkage=0.1;
    
    int nTrainEventsA[]={
        16000, 16000, 16000, 16000, 16000,
        4496, 4496, 4496, 4496, 4496,
        8793, 8793, 8794, 8794, 8794, 
        5148, 5148, 5148, 5148, 5148,
        46400, 46400, 46400, 46400, 46400, 
        2548, 2549, 2549, 2549, 2549
    };
    int nTestEventsA[]={
        4000, 4000, 4000, 4000, 4000,
        1124, 1124, 1124, 1124, 1124,
        2199, 2199, 2198, 2198, 2198,
        1287, 1287, 1287, 1287, 1287,
        11600, 11600, 11600, 11600, 11600,
        638, 637, 637, 637, 637
    };
    int nClassA[]={26,10,10,6,7,3};
    int nVariablesA[]={16,64,16,36,9,180};
    nTrainEvents=nTrainEventsA[jobID*5+fold];
    nTestEvents=nTestEventsA[jobID*5+fold];
    nClass=nClassA[jobID];
    nVariables=nVariablesA[jobID];
    
    train t(fTrain, fTest, fOut, nTrainEvents,nTestEvents,nClass,nVariables,treeType, shrinkage,nLeaves,minimumNodeSize,nMaxIteration);
    t.init();
    t.start();
    t.saveResult();
}

void testApplication(int argc, char** argv){
    char* model=argv[1];
    char* data=argv[2];
    application app(model);
    //read in test data
    int nVariable=app._nVariable;
    int nClass=app._nClass;
    double* x=new double[nVariable];
    int l;
    double* f=new double[nClass];
    ifstream infs(data,ifstream::in);
    if(!infs.good()){
        cout<<"Can not open "<<data<<endl;
        return;
    }
    ofstream outf("testApplication.dat",ofstream::out);
    if(!outf.good()){
        cout<<"Can not open testApplication.out"<<endl;
        return;
    }
    for(int iVar=0;iVar<nVariable;iVar++)
        infs>>x[iVar];
    infs>>l;
    while(infs.good()){
        app.eval(x,f);
        for(int iClass=0;iClass<nClass;iClass++)
            outf<<f[iClass]<<" ";
        outf << endl;
        for (int iVar = 0; iVar < nVariable; iVar++)
            infs >> x[iVar];
        infs>>l;
    }
    infs.close();
    outf.close();
}
int main(int argc, char** argv) {
    //srand((unsigned)time(NULL));
    //job(argc,argv);
    
//    int jobID=atoi(argv[1]);
//    int iFold=atoi(argv[2]);
//    int nLeaves=atoi(argv[3]);
//    int minimumSize=atoi(argv[4]);
//    int maxIterations=atoi(argv[5]);
//    char* prefix=argv[6];
//    test1(jobID,iFold,nLeaves,minimumSize,maxIterations,prefix);
    if (argc > 3) {
        char* data = argv[1];
        int nLeaves = atoi(argv[2]);
        int minimumNodeSize = atoi(argv[3]);
        int maxIterations = atoi(argv[4]);
        
        train t(data, directionFunction::_AOSO_LOGITBOOST_, 0.1, nLeaves, minimumNodeSize, maxIterations);
        t.init();
        t.start();
        t.saveResult();
    } else
        testApplication(argc, argv);
    return 0;
}
