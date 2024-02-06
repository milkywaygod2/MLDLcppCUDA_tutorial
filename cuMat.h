#pragma once
#include "_pch.h"

#define IDX_TRANSPOSE_1D(_i, _j, _leadingDimension) ((((_j))*(_leadingDimension))+((_i))) //C��� ���迭�� ��켱, cuBLAS�� ���迭�� ���켱
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
    void memMallocHost() {
        mHostMemory = (float*)malloc(rows * cols * sizeof(*mHostMemory));
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                mHostMemory[IDX2F(i, j, rows)] = 0.0;
            }
        }
    }
    void memMallocDevice() {
        cudaError_t error = cudaMalloc((void**)&mDeviceMemory,
            rows * cols * sizeof(*mDeviceMemory));
        if(error != cudaSuccess) printf("cudaMemcpy error\n");
        cudaMemset(mDeviceMemory, 0x00, rows * cols * sizeof(*mDeviceMemory));
        cudaThreadSynchronize();

    }
    void memHostToDevice() {
        cudaError_t error = cudaMemcpy(mDeviceMemory, mHostMemory,
            rows * cols * sizeof(*mDeviceMemory), cudaMemcpyHostToDevice);
        if(error != cudaSuccess) printf("memHostToDevice cudaMemcpy error\n");
    }
    void memDeviceToHost() {
        if(mHostMemory == NULL) this->memMallocHost();
        cudaError_t error = cudaMemcpy(mHostMemory, mDeviceMemory,
            rows * cols * sizeof(*mDeviceMemory), cudaMemcpyDeviceToHost);
        if(error != cudaSuccess)
            printf("memDeviceToHost cudaMemcpy error\n");
    }
    void memSetHost(int i, int j, float val) {
        if(mHostMemory == NULL)
            this->memMallocHost();

        mHostMemory[IDX2F(i, j, rows)] = val;
    }
    void memSetHost(float* v) {
        if(mHostMemory == NULL)
            this->memMallocHost();
        if(mDeviceMemory == NULL)
            cout << "memSetHost mDeviceMemory is null" << endl;

        cudaError_t error = cudaMemcpy(mDeviceMemory, v,
            rows * cols * sizeof(*mDeviceMemory), cudaMemcpyHostToDevice);
        if(error != cudaSuccess)
            printf("memSetHost cudaMemcpy error\n");
    }
    void memSetDevice(float* v) {
        cudaError_t error = cudaMemcpy(mDeviceMemory, v,
            rows * cols * sizeof(*mDeviceMemory), cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess)
            printf("memSetDevice cudaMemcpy error\n");
    }
    void memSetDeviceRow(float* v, int row_index) {
        cudaError_t error = cudaMemcpy(mDeviceMemory + row_index * cols, v,
            cols * sizeof(float), cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess)
            printf("memSetDeviceRow cudaMemcpy error\n");
    }
    void memSetDeviceCol(float* v, int col_index) {
        cudaError_t error = cudaMemcpy(mDeviceMemory + col_index * rows, v,
            rows * sizeof(float), cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess)
            printf("memSetDeviceCol cudaMemcpy error\n");
    }
    void toHostArray() {
        //cout << "toHostArray" << endl;
        if(mHostMemory == NULL) this->memMallocHost();
        memDeviceToHost();

        mHostArray.resize(rows * cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                mHostArray[IDX2F(i, j, rows)] = mHostMemory[IDX2F(i, j, rows)];
            }
        }
    }
    void fromHostArray() {
        if(mDeviceMemory == NULL) this->memMallocDevice();
        if(mHostMemory == NULL) this->memMallocHost();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                mHostMemory[IDX2F(i, j, rows)] = mHostArray[IDX2F(i, j, rows)];
            }
        }

        memHostToDevice();
    }


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
    friend void printRows(ostream& output, cuMat& a, int i) {
        output << "[";
        if(a.cols < 11) {
            for(int j = 0; j < a.cols; j++)  output << a.mHostMemory[IDX2F(i, j, a.rows)] << " ";
        } else {
            for(int j = 0; j < 3; j++)  output << a.mHostMemory[IDX2F(i, j, a.rows)] << " ";
            cout << "..., ";
            for(int j = a.cols - 2; j < a.cols; j++)  output << a.mHostMemory[IDX2F(i, j, a.rows)] << " ";
        }
        output << "]";
    }
    cuMat sliceRows(int offset, int len) {
        cuMat r(len, this->cols);

        slice_rows_kernel_exec(mDeviceMemory, r.mDeviceMemory, cols, rows, offset, len);

        return r;
    }
    void joinRows(cuMat& a, int offset, int len) {
        join_rows_kernel_exec(a.mDeviceMemory, mDeviceMemory, cols, rows, offset, len);
    }


    void fill_1() {
        mat_fill_1_kernel_exec(mDeviceMemory, mDeviceMemory, cols, rows);
    }
    void fill_a(float _a) {
        this->fill_1();
        this->aA(_a, *this);
    }
    cuMat sqrt() {
        cuMat R(rows, cols);
        sqrt_withplus_k(R, 1e-8);
        return R;
    }

    cuMat sqrt_d() {
        cuMat R(rows, cols);
        sqrt_withplus_k_d(R, 1e-8);
        return R;
    }
    void sqrt_withplus_k(cuMat& _R, float _k) {
        mat_sqrt_withplus_k_kernel_exec(mDeviceMemory, _R.mDeviceMemory, cols, rows, _k);
    }
    void sqrt_withplus_k_d(cuMat& _R, float _k) {
        mat_sqrt_withplus_k_d_kernel_exec(mDeviceMemory, _R.mDeviceMemory, cols, rows, _k);
    }
    cuMat log() {
        cuMat R(rows, cols);
        log_withplus_k(R, 0.0);
        return R;
    }
    void log_withplus_k(cuMat& _R, float _k) {
        mat_log_withplus_k_kernel_exec(mDeviceMemory, _R.mDeviceMemory, cols, rows, _k);
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
    void A_mul_B(const cuMat& _B, cuMat& _R) {
        mat_A_mul_B_kernel_exec(mDeviceMemory, _B.mDeviceMemory, _R.mDeviceMemory, cols, rows);
    }
    void aA_mul_bB_plusEqual(const cuMat& _B, cuMat& _R, float _a, float _b) {
        mat_aA_mul_bB_plusEqual_kernel_exec(mDeviceMemory, _B.mDeviceMemory, _R.mDeviceMemory, _a, _b, cols, rows);
    }
    void A_div_B(const cuMat& _B, cuMat& _R) {
        mat_A_div_B_kernel_exec(mDeviceMemory, _B.mDeviceMemory, _R.mDeviceMemory, cols, rows);
    }
    /*???*/void A_div_p(const float _p, cuMat& _R) {
        //matmod_kernel_exec(mDeviceMemory, _R.mDeviceMemory, cols, rows, _p);
    }

    void A_dot_B(const cuMat& _B, cuMat& _R) { // C = A��B
        if(cols != _B.rows) { cout << "operator dot error => _B.rows != _B.cols || _B.cols != _B.rows" << endl; return; }
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

public: //���۷�����
    cuMat& operator=(const cuMat& _B) {
        new_matrix(_B.rows, _B.cols);

        cudaError_t error = cudaMemcpy(mDeviceMemory, _B.mDeviceMemory,
            rows * cols * sizeof(*mDeviceMemory), cudaMemcpyDeviceToDevice);
        if(error != cudaSuccess)
            printf("cuMat operator= cudaMemcpy error\n");

        return *this;
    }
    float operator()(int _i, int _j) {
        if(mHostMemory == NULL)
            this->memMallocHost();

        this->memDeviceToHost();

        return mHostMemory[IDX_TRANSPOSE_1D(_i, _j, rows)];

    }
    friend ostream& operator<<(ostream& _output, cuMat& _B) {

        if(_B.mDeviceMemory == NULL) {
            printf("cuMat operator<< _B.mDeviceMemory is NULL\n");
            if(_B.mHostMemory == NULL) {
                printf("also cuMat operator<< _B.mHostMemory is NULL\n");
            }
        }
        if(_B.mHostMemory == NULL) _B.memMallocHost();


        cudaError_t error = cudaMemcpy(_B.mHostMemory, _B.mDeviceMemory,
            _B.rows * _B.cols * sizeof(*_B.mDeviceMemory), cudaMemcpyDeviceToHost);
        if(error != cudaSuccess)
            printf("cuMat operator<< cudaMemcpy error\n");

        _output << "matrix rows:" << _B.rows << " cols:" << _B.cols << endl;
        _output << "[";
        if(_B.rows < 11) {
            for(int i = 0; i < _B.rows; i++) {
                printRows(_output, _B, i);
                if(i != _B.rows - 1) _output << endl;
                else _output << "]" << endl;
            }
        } else {
            for(int i = 0; i < 5; i++) {
                printRows(_output, _B, i);
                _output << endl;
            }
            _output << "...," << endl;
            for(int i = _B.rows - 5; i < _B.rows; i++) {
                printRows(_output, _B, i);
                if(i != _B.rows - 1) _output << endl;
                else _output << "]" << endl;
            }
        }

        return _output;
    }
    friend cuMat operator+(const cuMat& _A, const cuMat& _B) {
        cuMat R = _A;
        R.A_plus_B(_B, R);
        return R;
    }
    friend cuMat operator+(float _k, cuMat& _A) { //��_A�� ������ ���⿡ ó�ߴµ� �� ���� const���ϰ� _A.���� ���������� ��¥�� R��ȯ�ε�......���������Ѵ�..?
        cuMat R = _A;
        _A.A_plus_b(_k, R);
        return R;
    }
    friend cuMat operator+(const cuMat& _A, float _k) {
        cuMat R = _A;
        R.A_plus_b(_k, R);
        return R;
    }
    friend cuMat operator-(const cuMat& _A, const cuMat& _B) {
        cuMat R = _A;
        R.A_minus_B(_B, R);
        return R;
    }
    friend cuMat operator*(const cuMat& _A, const cuMat& _B) {
        cuMat R = _A;
        R.A_mul_B(_B, R);
        return R;
    }
    friend cuMat operator*(float _k, const cuMat& _A) {
        cuMat R = _A;
        R.aA(_k, R);
        return R;
    }
    friend cuMat operator*(const cuMat& _A, float _k) {
        cuMat R = _A;
        R.aA(_k, R);
        return R;
    }
    /*???*/friend cuMat operator/(float _k, cuMat& _A) {
        cuMat R = _A;
        _A.A_div_p(_k, R);
        return R;
    }
    /*???*/friend cuMat operator/(const cuMat& _A, float _k) {
        cuMat R = _A;
        R.A_div_p(1.0 / _k, R);
        return R;
    }
    friend cuMat operator/(const cuMat& _A, const cuMat& _B) {
        cuMat R = _A;
        R.A_div_B(_B, R);
        return R;
    }
    cuMat& operator+=(const cuMat& _B) {
        A_plus_B(_B, *this);
        return *this;
    }
    cuMat& operator+=(float _k) {
        A_plus_b(_k, *this);
        return *this;
    }
    cuMat& operator-=(const cuMat& _B) {
        A_minus_B(_B, *this);
        return *this;
    }
    cuMat& operator-=(float _k) {
        A_plus_b(-_k, *this);
        return *this;
    }
    cuMat& operator*=(cuMat& _B) {
        A_mul_B(_B, *this);
        return *this;
    }
    cuMat& operator*=(float _k) {
        aA(_k, *this);
        return *this;
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