/* 
 * File:   bitArray.h
 * Author: kaiwu
 *
 * Created on March 15, 2014, 1:16 PM
 */

#ifndef BITARRAY_H
#define	BITARRAY_H
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include<iostream>
using namespace std;
class bitArray {
public:
    bitArray(int nElement);
    bool test(int index);

    bool set(int index);
    void reset(int start,int end);
    void print();
    void randomMask(float  r);
    void randomMask(int r);
    virtual ~bitArray();
private:
    char* _data;
    int* _index;
    int _nElement;
    int _nChar;
};

#endif	/* BITARRAY_H */

