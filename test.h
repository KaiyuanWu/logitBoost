/* 
 * File:   test.h
 * Author: kaiwu
 *
 * Created on August 6, 2014, 2:49 PM
 */

#ifndef TEST_H
#define	TEST_H
#include <string>
#include <fstream>
#include <iostream>
#include "application.h"
#include "dataManager.h"

using namespace std;
class test {
public:
    test(int iTask,int iDataset);
    test(const test& orig);
    void start();
    void start2(int iExe,int iLeave, int nMaxIterations);
    virtual ~test();
private:
    int _iTask;
    int _iDataset;
    int getBestAccuracy(string& modelFile,string& dataFile, float & bestAccuracy);
    float  getBestAccuracy(string& modelFile,string& dataFile,int iIteration);
    int getNEvents(string& file);
};

#endif	/* TEST_H */

