#pragma once

#ifndef _mat_fill_1_kernel_
#define _mat_fill_1_kernel_
    __global__ void mat_fill_1_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows);

    #ifdef __cplusplus
    extern "C" {
    #endif
        void mat_fill_1_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
    };
    #endif
#endif

#ifndef _mat_sqrt_withplus_k_kernel_
#define _mat_sqrt_withplus_k_kernel_
    __device__ __forceinline__ float mat_sqrt_withplus_k(float _srcEachArr1D, float _k); //GPU함수
    __global__ void mat_sqrt_withplus_k_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows, float _k); //커널함수

    #ifdef __cplusplus 
        extern "C" { 
    #endif
            void mat_sqrt_withplus_k_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows, float _k); //CPU함수
    #ifdef __cplusplus
        };
    #endif

#endif

#ifndef _mat_sqrt_withplus_k_d_kernel_
#define _mat_sqrt_withplus_k_d_kernel_
    __device__ __forceinline__ float mat_sqrt_withplus_k_d(float _srcEachArr1D, float _k); //GPU함수
    __global__ void mat_sqrt_withplus_k_d_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows, float _k); //커널함수
    #ifdef __cplusplus 
            extern "C" {
    #endif
                void mat_sqrt_withplus_k_d_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows, float _k); //CPU함수
    #ifdef __cplusplus
           };
    #endif
#endif

#ifndef _mat_log_withplus_k_kernel_
#define _mat_log_withplus_k_kernel_
    __device__ __forceinline__ float mat_log_withplus_k(float _srcEachArr1D, float _k); //GPU함수
    __global__ void mat_log_withplus_k_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows, float _k);
    #ifdef __cplusplus
            extern "C" {
    #endif
                void mat_log_withplus_k_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows, float _k);
    #ifdef __cplusplus
            };
    #endif
#endif

#ifndef _mat_A_mul_B_kernel_
#define _mat_A_mul_B_kernel_
           __global__ void mat_A_mul_B_kernel(const float* __restrict__ _src1Arr1D, const float* __restrict__ _src2Arr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
           extern "C" {
    #endif
                void mat_A_mul_B_kernel_exec(const float* _src1Arr1D, const float* _src2Arr1D,  float* _dstArr1D, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
           };
    #endif
#endif

#ifndef _mat_aA_mul_bB_plusEqual_kernel_
#define _mat_aA_mul_bB_plusEqual_kernel_
        __global__ void mat_aA_mul_bB_plusEqual_kernel(const float* __restrict__ _src1Arr1D, const float* __restrict__ _src2Arr1D,  float* __restrict__ _dstArr1D, float _a, float _b, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
           extern "C" {
    #endif
               void mat_aA_mul_bB_plusEqual_kernel_exec(const float* _src1Arr1D, const float* _src2Arr1D, float* _dstArr1D, float _a, float _b, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
           };
    #endif
#endif

#ifndef _mat_A_div_B_kernel_
#define _mat_A_div_B_kernel_
        __global__ void mat_A_div_B_kernel(const float* __restrict__ _src1Arr1D, const float* __restrict__ _src2Arr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
            extern "C" {
    #endif
                void mat_A_div_B_kernel_exec(const float* _src1Arr1D, const float* _src2Arr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows);
    #ifdef __cplusplus
            };
    #endif
#endif
