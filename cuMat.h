#pragma once
#include "_pch.h"

/*!!!*/#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))
#define FatalError(_string) {                                               \
    std::stringstream _where, _message;                                     \
    _where << __FILE__ << ':' << __LINE__;                                  \
    _message << std::string(_string) + "\n" << _where                       \
    std::cerr << _message.str() << "\nAborting...\n";                       \
    cudaDeviceReset();                                                      \
    exit(EXIT_FAILURE);                                                     \
}
/*!!!*/#define checkCUDNN(status) {                                                \
    std::stringstream _error;                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                   \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status);    \
      FatalError(_error.str());                                             \
    }                                                                       \
}
#define checkCudaErrors(status) {                                           \
    std::stringstream _error;                                               \
    if (status != 0) {                                                     \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status);      \
      FatalError(_error.str());                                             \
    }                                                                       \
}
#define checkCublasErrors(status) {                                         \
    std::stringstream _error;                                               \
    if (status != 0) {                                                      \
      _error << "Cublas failure\nError code " << status;                    \
      FatalError(_error.str());                                             \
    }                                                                       \
}

class MallocCounter {
public:
    int num = 0;
    void up() { num++; }
    void down() { num--; }
    int get() { return num; }
};
extern MallocCounter cuMallocCount;

class cuMat {
private:
    //friend class boost::serialization::access;
    /*!!!*/template<class Archive> void serialize(Archive& ar, const unsigned int version) { 
        ar& mHostArray;
        ar& rows;
        ar& cols;
    }
public:
    //--- cuBLAS의 필요리소스영역 정의
    cublasHandle_t cudaHandle;
    float* mDeviceMemory = NULL; //GPU &Memory
    float* mHostMemory = NULL; //CPU &Memory
    int rows = 0;
    int cols = 0;
    /*!!!*/vector<float> mHostArray;

public:
    void new_matrix(int _rows, int _cols) {
        cout << "new_matrix : 필요시 새 매트릭스 생성.." << endl;
        if(this->rows != _rows || this->cols != _cols) {
            cout << "new_matrix : 새로운 규격의 매트릭스 생성" << endl;
            if(mDeviceMemory != NULL || mHostMemory != NULL) {
                cout << "new_matrix : 기할당된 정보있는 경우, 삭제후 생성" << endl;
                del_matrix();
            }
            /*<keyfunction>*/
            this->rows = _rows;
            this->cols = _cols;
            cudaError_t status = cudaMalloc((void**)&mDeviceMemory, rows * cols * sizeof(*mDeviceMemory));

            if(status != cudaSuccess) printf("cuMat::new_matrix 쿠다 동적할당 실패\n");
            cudaMemset(mDeviceMemory, 0x00, rows * cols * sizeof(*mDeviceMemory));
            cudaThreadSynchronize();
            cuMallocCount.up();
        }
    }
    void del_matrix() {
        if(mDeviceMemory != NULL) {
            cudaFree(mDeviceMemory);
            mDeviceMemory = NULL;
            cuMallocCount.down();
        }
        if(mHostMemory != NULL) {
            free(mHostMemory);
            mHostMemory = NULL;
        }
        cudaThreadSynchronize();
    }

    int getRows() { return this->rows; }
    int getCols() { return this->cols; }
    /*needKernel*/void fill_1() {
        //mat_ones_kernel_exec(mDevice, mDevice, cols, rows);
    }
    
    /* 연산기호로 오버로딩하지않고 맴버함수화 하는 이유 : 행렬연산은 순서가 중요하기때문에. 
     * C = aA+bB 구조에서 A를 중심으로 모든것을 함수화할 뿐만아니라
     * C = aA+bB 구조는 사실상 2차원평면위의 한 직선임으로 선형조합을 계산하는 문제임 
     * a유무,b유무,B유무,A~B간 합/차/곱 등 다양한 조합이 있으나 모두 구현은 x 
     * cublasSgeam() : C = aA + bB
     * cublasSgemm() : C = aA * B + cC
     */
    void A_plus_B(const cuMat & _B, cuMat & _R) {
        float a = 1, b = 1; //스칼라
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치(행과열을 바꾸는것)
                                                rows, cols, 
                                                &a, mDeviceMemory, rows,
                                                &b, _B.mDeviceMemory, _B.rows,
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void A_plus_b(const float _b, cuMat& _R) {
        cuMat B1(rows, cols); B1.fill_1();
        float a = 1;
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
            CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치
            rows, cols,
            &a, mDeviceMemory, rows,
            &_b, B1.mDeviceMemory, B1.rows, //인자도 없는데다 스칼라도 0이라 A,R어느것을 가져와도 의미없음
            _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void A_plus_bB(const float _b, cuMat& _B, cuMat& _R) {
        float a = 1;
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
            CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치
            rows, cols,
            &a, mDeviceMemory, rows,
            &_b, _B.mDeviceMemory, _B.rows, //행렬의 곱연산시 C는 B의 크기로 나옴
            _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void A_minus_B(const cuMat& _B, cuMat& _R) {
        float a = 1, b = -1; //스칼라
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치
                                                rows, cols,
                                                &a, mDeviceMemory, rows,
                                                &b, _B.mDeviceMemory, _B.rows,
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void aA(const float _a, cuMat& _R) {
        float r = 0; //스칼라
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치
                                                rows, cols,
                                                &_a, mDeviceMemory, rows,
                                                &r, _R.mDeviceMemory, _R.rows, //행렬의 곱연산시 C는 B의 크기로 나옴
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void aA_plusEqual_B(const float _a, cuMat& _R) { // B += aA, B = aA + B
        float b = 1; //스칼라
        cublasStatus_t status = cublasSgeam(_R.cudaHandle, //_B
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치
                                                rows, cols,
                                                &_a, mDeviceMemory, rows,
                                                &b, _R.mDeviceMemory, _R.rows, //행렬의 곱연산시 C는 B의 크기로 나옴
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void aA_plus_bB(float _a, float _b, cuMat& _B, cuMat& _R) {
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N은 그대로, T는 전치
                                                rows, cols,
                                                &_a, mDeviceMemory, rows,
                                                &_b, _B.mDeviceMemory, _B.rows, //행렬의 곱연산시 C는 B의 크기로 나옴
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    void A_dot_B(const cuMat& _B, cuMat& _R) { // C = A·B
        if(cols != _B.rows) { cout << "operator dot error => a.rows != b.cols || a.cols != b.rows" << endl; return; }
        float a = 1, c = 0;
        cublasStatus_t status = cublasSgemm(cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //A,B
                                                rows, _B.cols, cols, //AorC행(A고유), BorC열(B고유), A열orB행(AB공통)
                                                &a, mDeviceMemory, rows, 
                                                _B.mDeviceMemory, _B.rows,
                                                &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm dot" << endl;
        cudaThreadSynchronize();
    }
    cuMat A_dot_B(const cuMat& _B) { //A·B, return값자체가 cuMat이므로 dotEqual과는 다름!
        cuMat C(this->rows, _B.cols);
        A_dot_B(_B, C);
        return C;
    }
    void A_dot_B_plus_C(const cuMat& _B, cuMat& _R) { // C += A·B
        float a = 1, c = 1;
        cublasStatus_t status = cublasSgemm(cudaHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            rows, _B.cols, cols,
                                            &a, mDeviceMemory, rows,
                                            _B.mDeviceMemory, _B.rows,
                                            &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm A_dot_B_plus_C" << endl;
        cudaThreadSynchronize();
    }
    void tA_dot_B_plus_C(const cuMat& _B, cuMat& _R) { // C += A·B
        float a = 1, c = 1;
        cublasStatus_t status = cublasSgemm(cudaHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rows, _B.cols, cols,
            &a, mDeviceMemory, rows,
            _B.mDeviceMemory, _B.rows,
            &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm tA_dot_B_plus_C" << endl;
        cudaThreadSynchronize();
    }
    void A_dot_tB_plus_C(const cuMat& _B, cuMat& _R) { // C += A·B
        float a = 1, c = 1;
        cublasStatus_t status = cublasSgemm(cudaHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rows, _B.cols, cols,
            &a, mDeviceMemory, rows,
            _B.mDeviceMemory, _B.rows,
            &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm A_dot_tB_plus_C" << endl;
        cudaThreadSynchronize();
    }
    void tA(cuMat& _R) { //reverse 전치
        float a = 1, b = 0; //스칼라
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N, //N은 그대로, T는 전치(행과열을 바꾸는것)
                                            cols, rows,
                                            &a, mDeviceMemory, rows,
                                            &b, _R.mDeviceMemory, cols, /*_R.?명확하지않음*/
                                            _R.mDeviceMemory, cols);    /*_R.?명확하지않음*/
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaThreadSynchronize();
    }
    cuMat tA() {
        cuMat R(cols, rows);
        tA(R);
        return R;
    }
    /*needKernel*/void mul(const cuMat& m, cuMat& r) {
        //mat_mul_elementwise_kernel_exec(mDevice, m.mDevice, r.mDevice, cols, rows);
    }
    /*needKernel*/void mul_plus(const cuMat& m, cuMat& r, float alpha, float beta) {
        //mat_mul_elementwise_plus_kernel_exec(mDevice, m.mDevice, r.mDevice, alpha, beta, cols, rows);
    }
public:
    cuMat() {
        cublasCreate(&cudaHandle);
        cudaThreadSynchronize();
        rows = cols = 0;
    }
    cuMat(int _rows, int _cols) {
        cublasCreate(&cudaHandle);
        cudaThreadSynchronize();
        new_matrix(rows, cols);
    }
    cuMat(const cuMat& _A) {
        cublasCreate(&cudaHandle);
        cudaThreadSynchronize();
        new_matrix(_A.rows, _A.cols);
        cudaError_t cuError = cudaMemcpy(mDeviceMemory, _A.mDeviceMemory, rows*cols*sizeof(*mDeviceMemory), cudaMemcpyDeviceToDevice); //*mDevice?
        if(cuError != cudaSuccess) printf("cuMat::cudaMemcpy 쿠다 메모리복사 실패\n");
    }
    ~cuMat() {
        del_matrix();
        cublasDestroy(cudaHandle);
    }
};