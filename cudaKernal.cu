#include "_pch.h"

#define BLOCK_SIZE 32

//--- mat_fill_1
__global__ void mat_fill_1_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows) { //__device__호출까지 필요없는 단순한 연산 (그래도 for문보다는 단순커널이라도 더빠름)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] = 1.0; }
}
void mat_fill_1_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_fill_1_kernel KERNEL_ARG2(grid, block) (_srcArr1D, _dstArr1D, _srcArrCols, _srcArrRows);
    cudaThreadSynchronize();
}

//--- mat_sqrt_withplus_k
__device__ __forceinline__ float mat_sqrt_withplus_k(float _srcEachArr1D, float _k) { return std::sqrt(_srcEachArr1D + _k); }
__global__ void mat_sqrt_withplus_k_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows, float _k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; //전체 그리드내 블록번호 * 스레드번호 + 스레드내 인덱스
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] = mat_sqrt_withplus_k(_srcArr1D[row * _srcArrRows + col], _k); }
}
void mat_sqrt_withplus_k_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows, float _k) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_sqrt_withplus_k_kernel KERNEL_ARG2(grid, block) (_srcArr1D, _dstArr1D, _srcArrCols, _srcArrRows, _k);
    cudaDeviceSynchronize();
}

//--- mat_sqrt_withplus_k_d
__device__ __forceinline__ float mat_sqrt_withplus_k_d(float _srcEachArr1D, float _k) { return 0.5 * 1.0 / std::sqrt(_srcEachArr1D + _k); } /*return 0.5 * pow(a + alpha, -0.5f);*/
__global__ void mat_sqrt_withplus_k_d_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows, float _k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; //전체 그리드내 블록번호 * 스레드번호 + 스레드내 인덱스
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] = mat_sqrt_withplus_k_d(_srcArr1D[row * _srcArrRows + col], _k); }
}
void mat_sqrt_withplus_k_d_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows, float _k) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_sqrt_withplus_k_d_kernel KERNEL_ARG2(grid, block) (_srcArr1D, _dstArr1D, _srcArrCols, _srcArrRows, _k);
    cudaDeviceSynchronize();
}

//--- mat_mat_log_withplus_k
__device__ __forceinline__ float mat_log_withplus_k(float _srcEachArr1D, float _k) { return std::log(_srcEachArr1D + _k); }
__global__ void mat_log_withplus_k_kernel(const float* __restrict__ _srcArr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows, float _k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; //전체 그리드내 블록번호 * 스레드번호 + 스레드내 인덱스
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] = mat_log_withplus_k(_srcArr1D[row * _srcArrRows + col], _k); }
}
void mat_log_withplus_k_kernel_exec(const float* _srcArr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows, float _k) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_log_withplus_k_kernel KERNEL_ARG2(grid, block) (_srcArr1D, _dstArr1D, _srcArrCols, _srcArrRows, _k);
    cudaDeviceSynchronize();
}

//--- mat_A_mul_B
__global__ void mat_A_mul_B_kernel(const float* __restrict__ _src1Arr1D, const float* __restrict__ _src2Arr1D,  float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] = _src1Arr1D[row * _srcArrRows + col] * _src2Arr1D[row * _srcArrRows + col]; }
}
void mat_A_mul_B_kernel_exec(const float* _src1Arr1D, const float* _src2Arr1D,  float* _dstArr1D, int _srcArrCols, int _srcArrRows) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_A_mul_B_kernel KERNEL_ARG2(grid, block) (_src1Arr1D, _src2Arr1D, _dstArr1D, _srcArrCols, _srcArrRows);
    cudaThreadSynchronize();
}

//--- mat_aA_mul_bB_plusEqual
__global__ void mat_aA_mul_bB_plusEqual_kernel( const float* __restrict__ _src1Arr1D, const float* __restrict__ _src2Arr1D, float* __restrict__ _dstArr1D, float _a, float _b, int _srcArrCols, int _srcArrRows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] += _a * _src1Arr1D[row * _srcArrRows + col] * _b * _src2Arr1D[row * _srcArrRows + col]; }

}
void mat_aA_mul_bB_plusEqual_kernel_exec(const float* _src1Arr1D, const float* _src2Arr1D, float* _dstArr1D, float _a, float _b, int _srcArrCols, int _srcArrRows) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_aA_mul_bB_plusEqual_kernel KERNEL_ARG2(grid, block) (_src1Arr1D, _src2Arr1D, _dstArr1D, _a, _b, _srcArrCols, _srcArrRows);
    cudaThreadSynchronize();
}

//--- mat_A_div_B
__global__ void mat_A_div_B_kernel(const float* __restrict__ _src1Arr1D, const float* __restrict__ _src2Arr1D, float* __restrict__ _dstArr1D, int _srcArrCols, int _srcArrRows) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < _srcArrCols && col < _srcArrRows) { _dstArr1D[row * _srcArrRows + col] = _src1Arr1D[row * _srcArrRows + col] / _src2Arr1D[row * _srcArrRows + col]; }
}
void mat_A_div_B_kernel_exec(const float* _src1Arr1D, const float* _src2Arr1D, float* _dstArr1D, int _srcArrCols, int _srcArrRows) {
    /* specified block and grid size */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_srcArrRows + block.x - 1) / block.x, (_srcArrCols + block.y - 1) / block.y);

    /* lunch kernel */
    mat_A_div_B_kernel KERNEL_ARG2(grid, block) (_src1Arr1D, _src2Arr1D, _dstArr1D, _srcArrCols, _srcArrRows);
    cudaThreadSynchronize();
}


