/* 
 * File:   bitArray.cpp
 * Author: kaiwu
 * 
 * Created on March 15, 2014, 1:16 PM
 */
#include "bitArray.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"

const char _mask[8] = {0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
const char _shadowLow[8] = {0x0, 0x1, 0x3, 0x7, 0xf, 0x1f, 0x3f, 0x7f};
const char _shadowHigh[8] = {0xfe,0xfc,0xf8,0xf0,0xe0,0xc0,0x80,0x0};
//const char _shadowLow[8]={0b0,0b1,0b11,0b111,0b1111,0b11111,0b111111,0b1111111};
//const char _shadowHigh[8]={0b11111110,0b11111100,0b11111000,0b11110000,0b11100000,0b11000000,0b10000000,0b0};
bitArray::bitArray(int nElement) {
    _nChar=nElement/8+1;
    _nElement=nElement;
    _data=new char[_nChar];
    _index=new int[_nElement];
    memset(_data,0,_nChar);
    
}
void bitArray::randomMask(double r){
    for(int i=0;i<_nElement;i++)
        _index[i]=i;
    int maxMask=ceil(r*_nElement);
    for(int iE=0;iE<maxMask;iE++){
        int ie=(_nElement-iE)*(double(rand())/RAND_MAX);
        int ii=_index[ie];
        set(ii);
        _index[ie]=_index[_nElement-iE-1];
    }
}
void bitArray::randomMask(int r){
    for(int i=0;i<_nElement;i++)
        _index[i]=i;
    for(int iE=0;iE<r;iE++){
        int ie=(_nElement-iE)*(double(rand())/RAND_MAX);
        int ii=_index[ie];
        set(ii);
        _index[ie]=_index[_nElement-iE-1];
    }
}
bool bitArray::test(int index){
    if(index>_nElement){
        cout<<"[Test] Caution "<<index<<" is greater than "<<_nElement<<" in bit array!"<<endl;
        exit(-1);
        return false;
    }
    int i=index/8;
    int j=index%8;
    return bool(_data[i]&_mask[j]);
}
bool bitArray::set(int index){
    if(index>_nElement){
        cout<<"[Set] Caution "<<index<<" is greater than "<<_nElement<<" in bit array!"<<endl;
        return false;
    }
    int i=index/8;
    int j=index%8;
    _data[i]=(_data[i]|_mask[j]);
    return true;
}
void bitArray::reset(int start, int end){
    int iStart,iEnd;
    int jStart,jEnd;
    if(end>=_nElement)
        end=_nElement-1;
    iStart=start/8;
    jStart=start%8;
    iEnd=end/8;
    jEnd=end%8;
    if (iStart < iEnd) {
        memset(_data + iStart+bool(jStart), 0, (iEnd - iStart) * sizeof (char));
        _data[iStart] = (_data[iStart] & _shadowLow[jStart]);
        _data[iEnd] = (_data[iEnd] & _shadowHigh[jEnd]);
    }
    else{
        char mask=(_shadowHigh[jEnd]|_shadowLow[jStart]);
        _data[iStart] = (_data[iStart] & mask);
    }
}
void bitArray::print(){
    for(int i=0;i<_nElement;i++){
        if(test(i))
            printf("1");
        else
            printf("0");
    }
    printf("\n");
}
bitArray::~bitArray() {
    if(_data)
        delete[] _data;
    if(_index)
        delete[] _index;
}

