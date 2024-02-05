#include "_pch.h"

#define BLOCK_SIZE 32

//--- mat_sqrt
__device__ __forceinline__ float mat_sqrt(float _srcEachArr1D, float _alpha) { return std::sqrt(_srcEachArr1D + _alpha); }
__global__ void mat_sqrt_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrRows, int _srcArrCols, float _alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; //전체 그리드내 블록번호 * 스레드번호 + 스레드내 인덱스
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrRows && col < _srcArrCols) { _dstArr1D[row * _srcArrCols + col] = mat_sqrt(_srcArr1D[row * _srcArrCols + col], _alpha); }
}
void mat_sqrt_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrRows, int _srcArrCols, float _alpha) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrCols + block.x - 1) / block.x, (_srcArrRows + block.y - 1) / block.y);

    /* lunch kernel */
    mat_sqrt_kernel KERNEL_ARG2(grid, block) (_srcArr1D, _dstArr1D, _srcArrRows, _srcArrCols, _alpha);
    cudaDeviceSynchronize();
}

//--- mat_fill_1
__global__ void mat_fill_1_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrRows, int _srcArrCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrRows && col < _srcArrCols) { _dstArr1D[row * _srcArrCols + col] = 1.0; }
}
void mat_fill_1_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrRows, int _srcArrCols) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrCols + block.x - 1) / block.x, (_srcArrRows + block.y - 1) / block.y);

    /* lunch kernel */
    mat_fill_1_kernel KERNEL_ARG2(grid, block) (_srcArr1D, _dstArr1D, _srcArrRows, _srcArrCols);
    cudaThreadSynchronize();
}
