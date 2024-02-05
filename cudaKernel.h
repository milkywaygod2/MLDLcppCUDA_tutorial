#pragma once

//--- mat_sqrt
#ifndef _mat_sqrt_kernel_

    #define _mat_sqrt_kernel_
    __device__ __forceinline__ float mat_sqrt(float _srcEachArr1D, float _alpha); //GPU함수
    __global__ void mat_sqrt_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrRows, int _srcArrCols, float _alpha); //커널함수

    #ifdef __cplusplus 
        extern "C" { 
    #endif
            void mat_sqrt_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrRows, int _srcArrCols, float _alpha); //CPU함수
    #ifdef __cplusplus
        };
    #endif

#endif

//--- mat_fill_1
#ifndef _mat_fill_1_kernel_
    #define _mat_fill_1_kernel_
    __global__ void mat_fill_1_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrRows, int _srcArrCols);

    #ifdef __cplusplus
        extern "C" {
    #endif
            void mat_fill_1_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrRows, int _srcArrCols);
    #ifdef __cplusplus
        };
    #endif
#endif
