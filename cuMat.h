#pragma once
#include "_pch.h"

#define IDX_T1D_FROM2D(i,j,leadingDimension) ((((j))*(leadingDimension))+((i))) //C��� ���迭�� ��켱, cuBLAS�� ���迭�� ���켱
#define FatalError(_string) {                                                   \
    std::stringstream _where, _message;                                         \
    /*_where << __FILE__ << ':' << __LINE__;*/                                      \
    _message << std::string(_string) + "\n" << __FILE__ << ':' << __LINE__;     \
    std::cerr << _message.str() << "\nAborting...\n";                           \
    cudaDeviceReset();                                                          \
    exit(EXIT_FAILURE);                                                         \
}                                                                               
/*!!!*/#define checkCUDNN(status) {                                             \
    std::stringstream _error;                                                   \
    if (status != CUDNN_STATUS_SUCCESS) {                                       \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status);        \
      FatalError(_error.str());                                                 \
    }                                                                           \
}                                                                               
#define checkCudaErrors(status) {                                               \
    std::stringstream _error;                                                   \
    if (status != 0) {                                                          \
      _error << "Cuda failure\nError: " << status;          \
      FatalError(_error.str());                                                 \
    }                                                                           \
}                                                                               
#define checkCublasErrors(status) {                                             \
    std::stringstream _error;                                                   \
    if (status != 0) {                                                          \
      _error << "Cublas failure\nError code " << cublasGetStatusString(status); \
      FatalError(_error.str());                                                 \
    }                                                                           \
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
    //--- cuBLAS�� �ʿ丮�ҽ����� ����
    cublasHandle_t cudaHandle;
    float* mDeviceMemory = NULL; //GPU &Memory
    float* mHostMemory = NULL; //CPU &Memory
    int rows = 0;
    int cols = 0;
    /*!!!*/vector<float> mHostArray;

public:
    void new_matrix(int _rows, int _cols) {
        cout << "new_matrix : �ʿ�� �� ��Ʈ���� ����.." << endl;
        if(this->rows != _rows || this->cols != _cols) {
            cout << "new_matrix : ���ο� �԰��� ��Ʈ���� ����" << endl;
            if(mDeviceMemory != NULL || mHostMemory != NULL) {
                cout << "new_matrix : ���Ҵ�� �����ִ� ���, ������ ����" << endl;
                del_matrix();
            }
            /*<keyfunction>*/
            this->rows = _rows;
            this->cols = _cols;
            cudaError_t error = cudaMalloc((void**)&mDeviceMemory, rows * cols * sizeof(*mDeviceMemory));

            if(error != cudaSuccess) printf("cuMat::new_matrix ��� �����Ҵ� ����\n");
            cudaMemset(mDeviceMemory, 0x00, rows * cols * sizeof(*mDeviceMemory));
            cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
    }
    int getRows() { return this->rows; }
    int getCols() { return this->cols; }

    /*needKernel*/void fill_1() {
        mat_ones_kernel_exec(mDevice, mDevice, cols, rows);
    }
    void fill_a(float _a) {
        this->fill_1();
        this->aA(_a, *this);
    }
    void copy_from_B(const cuMat& _B) {
        if(rows != _B.rows || cols != _B.cols) { cout << "cuMat copy error rows != _B.rows || cols != _B.cols" << endl; }
        cudaError_t error = cudaMemcpy(mDeviceMemory, _B.mDeviceMemory, rows * cols * sizeof(*mDeviceMemory), cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess) printf("cudaMemcpy error\n");
    }

    /* �����ȣ�� �����ε������ʰ� �ɹ��Լ�ȭ �ϴ� ���� : ��Ŀ����� ������ �߿��ϱ⶧����. 
     * C = aA+bB �������� A�� �߽����� ������ �Լ�ȭ�� �Ӹ��ƴ϶�, �ǻ� 2����������� �� ���������� ���������� ����ϴ� ������ 
     * a����,b����,B����,A~B�� ��/��/�� �� �پ��� ������ ������ ��� ������ x 
     * cublasSgeam() : C = aA + bB
     * cublasSgemm() : C = aA * B + cC
     * 
     */
    void A_plus_B(const cuMat & _B, cuMat & _R) {
        float a = 1, b = 1; //��Į��
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ(������� �ٲٴ°�)
                                                rows, cols, 
                                                &a, mDeviceMemory, rows,
                                                &b, _B.mDeviceMemory, _B.rows,
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void A_plus_b(const float _b, cuMat& _R) {
        cuMat B1(rows, cols); B1.fill_1();
        float a = 1;
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
            CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ
            rows, cols,
            &a, mDeviceMemory, rows,
            &_b, B1.mDeviceMemory, B1.rows, //���ڵ� ���µ��� ��Į�� 0�̶� A,R������� �����͵� �ǹ̾���
            _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void A_plus_bB(const float _b, cuMat& _B, cuMat& _R) {
        float a = 1;
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
            CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ
            rows, cols,
            &a, mDeviceMemory, rows,
            &_b, _B.mDeviceMemory, _B.rows, //����� ������� C�� B�� ũ��� ����
            _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void A_minus_B(const cuMat& _B, cuMat& _R) {
        float a = 1, b = -1; //��Į��
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ
                                                rows, cols,
                                                &a, mDeviceMemory, rows,
                                                &b, _B.mDeviceMemory, _B.rows,
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void aA(const float _a, cuMat& _R) {
        float r = 0; //��Į��
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ
                                                rows, cols,
                                                &_a, mDeviceMemory, rows,
                                                &r, _R.mDeviceMemory, _R.rows, //����� ������� C�� B�� ũ��� ����
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void aA_plusEqual_B(const float _a, cuMat& _R) { // B += aA, B = aA + B
        float b = 1; //��Į��
        cublasStatus_t status = cublasSgeam(_R.cudaHandle, //_B
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ
                                                rows, cols,
                                                &_a, mDeviceMemory, rows,
                                                &b, _R.mDeviceMemory, _R.rows, //����� ������� C�� B�� ũ��� ����
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void aA_plus_bB(float _a, float _b, cuMat& _B, cuMat& _R) {
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //N�� �״��, T�� ��ġ
                                                rows, cols,
                                                &_a, mDeviceMemory, rows,
                                                &_b, _B.mDeviceMemory, _B.rows, //����� ������� C�� B�� ũ��� ����
                                                _R.mDeviceMemory, _R.rows);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
    }
    void A_dot_B(const cuMat& _B, cuMat& _R) { // C = A��B
        if(cols != _B.rows) { cout << "operator dot error => a.rows != b.cols || a.cols != b.rows" << endl; return; }
        float a = 1, c = 0;
        cublasStatus_t status = cublasSgemm(cudaHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N, //A,B
                                                rows, _B.cols, cols, //AorC��(A����), BorC��(B����), A��orB��(AB����)
                                                &a, mDeviceMemory, rows, 
                                                _B.mDeviceMemory, _B.rows,
                                                &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm dot" << endl;
        cudaDeviceSynchronize();
    }
    cuMat A_dot_B(const cuMat& _B) { //A��B, return����ü�� cuMat�̹Ƿ� dotEqual���� �ٸ�!
        cuMat C(this->rows, _B.cols);
        A_dot_B(_B, C);
        return C;
    }
    void A_dot_B_plus_C(const cuMat& _B, cuMat& _R) { // C += A��B
        float a = 1, c = 1;
        cublasStatus_t status = cublasSgemm(cudaHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            rows, _B.cols, cols,
                                            &a, mDeviceMemory, rows,
                                            _B.mDeviceMemory, _B.rows,
                                            &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm A_dot_B_plus_C" << endl;
        cudaDeviceSynchronize();
    }
    void tA_dot_B_plus_C(const cuMat& _B, cuMat& _R) { // C += A��B
        float a = 1, c = 1;
        cublasStatus_t status = cublasSgemm(cudaHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            rows, _B.cols, cols,
                                            &a, mDeviceMemory, rows,
                                            _B.mDeviceMemory, _B.rows,
                                            &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm tA_dot_B_plus_C" << endl;
        cudaDeviceSynchronize();
    }
    void A_dot_tB_plus_C(const cuMat& _B, cuMat& _R) { // C += A��B
        float a = 1, c = 1;
        cublasStatus_t status = cublasSgemm(cudaHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rows, _B.cols, cols,
            &a, mDeviceMemory, rows,
            _B.mDeviceMemory, _B.rows,
            &c, _R.mDeviceMemory, _R.rows);
        checkCublasErrors(status);
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgemm A_dot_tB_plus_C" << endl;
        cudaDeviceSynchronize();
    }
    void tA(cuMat& _R) { //reverse ��ġ
        float a = 1, b = 0; //��Į��
        cublasStatus_t status = cublasSgeam(_R.cudaHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N, //N�� �״��, T�� ��ġ(������� �ٲٴ°�)
                                            cols, rows,
                                            &a, mDeviceMemory, rows,
                                            &b, _R.mDeviceMemory, cols, /*_R.?��Ȯ��������*/
                                            _R.mDeviceMemory, cols);    /*_R.?��Ȯ��������*/
        if(status != CUBLAS_STATUS_SUCCESS) cout << "cannot cublasSgeam" << endl;
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
        rows = cols = 0;
    }
    cuMat(int _rows, int _cols) {
        cublasCreate(&cudaHandle);
        cudaDeviceSynchronize();
        new_matrix(rows, cols);
    }
    cuMat(const cuMat& _A) {
        cublasCreate(&cudaHandle);
        cudaDeviceSynchronize();
        new_matrix(_A.rows, _A.cols);
        cudaError_t error = cudaMemcpy(mDeviceMemory, _A.mDeviceMemory, rows*cols*sizeof(*mDeviceMemory), cudaMemcpyDeviceToDevice); //*mDevice?
        if(error != cudaSuccess) printf("cuMat::cudaMemcpy ��� �޸𸮺��� ����\n");
    }
    ~cuMat() {
        del_matrix();
        cublasDestroy(cudaHandle);
    }
};