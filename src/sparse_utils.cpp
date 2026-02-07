#include "sparse_utils.h"
#include <stdio.h>

// 常量定义
const float floatone = 1.0f;
const float floatzero = 0.0f;

// ============================================================================
// CUBLASWrapper 实现
// ============================================================================

CUBLASWrapper::CUBLASWrapper() {
    CHECK_CUBLAS(cublasCreate(&handle_));
}

CUBLASWrapper::~CUBLASWrapper() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

float CUBLASWrapper::dot(int n, const float* x, const float* y) {
    float result = 0.0f;
    CHECK_CUBLAS(cublasSdot(handle_, n, x, 1, y, 1, &result));
    return result;
}

void CUBLASWrapper::axpy(int n, float alpha, const float* x, float* y) {
    CHECK_CUBLAS(cublasSaxpy(handle_, n, &alpha, x, 1, y, 1));
}

void CUBLASWrapper::copy(int n, const float* x, float* y) {
    CHECK_CUBLAS(cublasScopy(handle_, n, x, 1, y, 1));
}

void CUBLASWrapper::scal(int n, float alpha, float* x) {
    CHECK_CUBLAS(cublasSscal(handle_, n, &alpha, x, 1));
}

// ============================================================================
// CUSparseWrapper 实现
// ============================================================================

CUSparseWrapper::CUSparseWrapper() {
    CHECK_CUSPARSE(cusparseCreate(&handle_));
}

CUSparseWrapper::~CUSparseWrapper() {
    if (handle_) {
        cusparseDestroy(handle_);
    }
}

void CUSparseWrapper::spmv(cusparseSpMatDescr_t matA,
                           const cusparseDnVecDescr_t vecX,
                           cusparseDnVecDescr_t vecY,
                           float alpha, float beta, void* buffer) {
    CHECK_CUSPARSE(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
}

void CUSparseWrapper::ilu0_setup(int n, int nz,
                                 cusparseMatDescr_t mat_descr,
                                 float* d_valsILU0,
                                 const int* d_row_ptr,
                                 const int* d_col_ind,
                                 csrilu02Info_t ilu0_info,
                                 void** d_buffer, int* buffer_size) {
    // 查询buffer大小
    CHECK_CUSPARSE(cusparseScsrilu02_bufferSize(handle_, n, nz, mat_descr,
                                                 d_valsILU0, d_row_ptr, d_col_ind,
                                                 ilu0_info, buffer_size));

    // 分配buffer
    if (*d_buffer == nullptr) {
        CHECK_CUDA(cudaMalloc(d_buffer, *buffer_size));
    }

    // 执行analysis
    CHECK_CUSPARSE(cusparseScsrilu02_analysis(handle_, n, nz, mat_descr,
                                              d_valsILU0, d_row_ptr, d_col_ind,
                                              ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                              *d_buffer));
}

void CUSparseWrapper::ilu0_compute(int n, int nz,
                                   cusparseMatDescr_t mat_descr,
                                   float* d_valsILU0,
                                   const int* d_row_ptr,
                                   const int* d_col_ind,
                                   csrilu02Info_t ilu0_info,
                                   void* d_buffer) {
    CHECK_CUSPARSE(cusparseScsrilu02(handle_, n, nz, mat_descr,
                                     d_valsILU0, d_row_ptr, d_col_ind,
                                     ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                     d_buffer));
}

void CUSparseWrapper::triangular_solve_setup(cusparseSpMatDescr_t matM,
                                             cusparseSpSVDescr_t spsv_descr,
                                             const cusparseDnVecDescr_t vecX,
                                             const cusparseDnVecDescr_t vecY,
                                             void** d_buffer, size_t* buffer_size) {
    // 查询buffer大小
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &floatone, matM, vecX, vecY,
                                          CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                                          spsv_descr, buffer_size));

    // 分配buffer
    if (*d_buffer == nullptr) {
        CHECK_CUDA(cudaMalloc(d_buffer, *buffer_size));
    }

    // 执行analysis
    CHECK_CUSPARSE(cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &floatone, matM, vecX, vecY,
                                        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                                        spsv_descr, *d_buffer));
}

void CUSparseWrapper::triangular_solve(cusparseSpMatDescr_t matM,
                                       cusparseSpSVDescr_t spsv_descr,
                                       const cusparseDnVecDescr_t vecX,
                                       cusparseDnVecDescr_t vecY) {
    CHECK_CUSPARSE(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &floatone, matM, vecX, vecY,
                                     CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                                     spsv_descr));
}

// ============================================================================
// SparseMatrix 实现
// ============================================================================

SparseMatrix::SparseMatrix(int rows, int cols, int nn)
    : rows(rows), cols(cols), nnz(nn),
      row_ptr(rows + 1), col_ind(nn), values(nn),
      d_row_ptr(nullptr), d_col_ind(nullptr), d_values(nullptr) {
}

SparseMatrix::~SparseMatrix() {
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_ind) cudaFree(d_col_ind);
    if (d_values) cudaFree(d_values);
}

void SparseMatrix::upload_to_gpu() {
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, col_ind.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, values.data(), nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
}

cusparseSpMatDescr_t SparseMatrix::create_sparse_descr() const {
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, rows, cols, nnz,
                                     d_row_ptr, d_col_ind, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    return matA;
}

// ============================================================================
// GPUVector 实现
// ============================================================================

GPUVector::GPUVector(size_t n) : n(n), d_data(nullptr) {
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));
}

GPUVector::~GPUVector() {
    if (d_data) {
        cudaFree(d_data);
    }
}

// 移动构造函数
GPUVector::GPUVector(GPUVector&& other) noexcept
    : n(other.n), d_data(other.d_data) {
    other.n = 0;
    other.d_data = nullptr;
}

// 移动赋值运算符
GPUVector& GPUVector::operator=(GPUVector&& other) noexcept {
    if (this != &other) {
        // 释放当前资源
        if (d_data) {
            cudaFree(d_data);
        }

        // 窃取其他对象的资源
        n = other.n;
        d_data = other.d_data;

        // 将其他对象置为有效但空的状态
        other.n = 0;
        other.d_data = nullptr;
    }
    return *this;
}

void GPUVector::upload_from_host(const float* h_data) {
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
}

void GPUVector::download_to_host(float* h_data) const {
    CHECK_CUDA(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
}

cusparseDnVecDescr_t GPUVector::create_dnvec_descr() const {
    cusparseDnVecDescr_t vec;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec, n, d_data, CUDA_R_32F));
    return vec;
}

// ============================================================================
// CPUOps 实现（CPU 向量运算）
// ============================================================================

namespace CPUOps {
    float dot(const std::vector<float>& x, const std::vector<float>& y) {
        float result = 0.0f;
        for (size_t i = 0; i < x.size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }

    void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y) {
        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    void scal(float alpha, std::vector<float>& x) {
        for (size_t i = 0; i < x.size(); i++) {
            x[i] *= alpha;
        }
    }

    void copy(const std::vector<float>& x, std::vector<float>& y) {
        y = x;
    }

    void spmv(int n, const std::vector<int>& row_ptr,
              const std::vector<int>& col_ind,
              const std::vector<float>& values,
              const std::vector<float>& x,
              std::vector<float>& y) {
        std::fill(y.begin(), y.end(), 0.0f);
        for (int i = 0; i < n; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                y[i] += values[j] * x[col_ind[j]];
            }
        }
    }
}
