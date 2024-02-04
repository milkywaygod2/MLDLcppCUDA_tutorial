//#pragma once
//#include "_pch.h"
//class CcuMatSparse {
//private:
//
//public:
//    int rows = 0;
//    int cols = 0;
//    int numVals = 0;
//
//    cusparseHandle_t cuHandle; //?
//    cusparseMatDescr_t descr; //?
//
//    float* csrVal = NULL; //?
//    int* csrRowPtr = NULL; //?
//    int* csrColInd = NULL; //?
//
//    float* csrValDevice = NULL; //?
//    int* csrRowPtrDevice = NULL; //?
//    int* csrColIndDevice = NULL; //?
//
//    CcuMat rt, bt; //?
//
//public:
//    void new_matrix(int _rows, int _cols, int _numVals) {
//        this->rows = _rows;
//        this->cols = _cols;
//        this->numVals = _numVals;
//        
//        cudaError_t error;
//        error = cudaMalloc((void**)&csrValDevice, _numVals * sizeof(*csrValDevice));
//        error = cudaMalloc((void**)&csrRowPtrDevice, (rows + 1) * sizeof(*csrRowPtrDevice)); //?
//        error = cudaMalloc((void**)&csrColIndDevice, _numVals * sizeof(*csrColIndDevice)); //?
//       
//        cudaMemset(csrValDevice, 0x00, _numVals * sizeof(*csrValDevice));
//        cudaMemset(csrRowPtrDevice, 0x00, (rows + 1) * sizeof(*csrRowPtrDevice));
//        cudaMemset(csrColIndDevice, 0x00, _numVals * sizeof(*csrColIndDevice));
//    }
//};
//
