#include "sparse_utils.h"
#include <stdio.h>

// ============================================================================
// CUBLASWrapper 模板实现（使用标签分发处理S/D后缀）
// ============================================================================

template<Precision P>
CUBLASWrapper<P>::CUBLASWrapper() {
    CHECK_CUBLAS(cublasCreate(&handle_));
}

template<Precision P>
CUBLASWrapper<P>::~CUBLASWrapper() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

// float 特化版本
template<>
float CUBLASWrapper<Precision::Float32>::dot(int n, const float* x, const float* y) {
    float result = 0.0f;
    CHECK_CUBLAS(cublasSdot(handle_, n, x, 1, y, 1, &result));
    return result;
}

template<>
void CUBLASWrapper<Precision::Float32>::axpy(int n, float alpha, const float* x, float* y) {
    CHECK_CUBLAS(cublasSaxpy(handle_, n, &alpha, x, 1, y, 1));
}

template<>
void CUBLASWrapper<Precision::Float32>::copy(int n, const float* x, float* y) {
    CHECK_CUBLAS(cublasScopy(handle_, n, x, 1, y, 1));
}

template<>
void CUBLASWrapper<Precision::Float32>::scal(int n, float alpha, float* x) {
    CHECK_CUBLAS(cublasSscal(handle_, n, &alpha, x, 1));
}

// double 特化版本
template<>
double CUBLASWrapper<Precision::Float64>::dot(int n, const double* x, const double* y) {
    double result = 0.0;
    CHECK_CUBLAS(cublasDdot(handle_, n, x, 1, y, 1, &result));
    return result;
}

template<>
void CUBLASWrapper<Precision::Float64>::axpy(int n, double alpha, const double* x, double* y) {
    CHECK_CUBLAS(cublasDaxpy(handle_, n, &alpha, x, 1, y, 1));
}

template<>
void CUBLASWrapper<Precision::Float64>::copy(int n, const double* x, double* y) {
    CHECK_CUBLAS(cublasDcopy(handle_, n, x, 1, y, 1));
}

template<>
void CUBLASWrapper<Precision::Float64>::scal(int n, double alpha, double* x) {
    CHECK_CUBLAS(cublasDscal(handle_, n, &alpha, x, 1));
}

// ============================================================================
// CUSparseWrapper 模板实现
// ============================================================================

template<Precision P>
CUSparseWrapper<P>::CUSparseWrapper() {
    CHECK_CUSPARSE(cusparseCreate(&handle_));
}

template<Precision P>
CUSparseWrapper<P>::~CUSparseWrapper() {
    if (handle_) {
        cusparseDestroy(handle_);
    }
}

template<Precision P>
void CUSparseWrapper<P>::spmv(cusparseSpMatDescr_t matA,
                               const cusparseDnVecDescr_t vecX,
                               cusparseDnVecDescr_t vecY,
                               typename CUSparseWrapper<P>::Scalar alpha,
                               typename CUSparseWrapper<P>::Scalar beta,
                               void* buffer) {
    CHECK_CUSPARSE(cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CudaDataType<P>::value, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
}

// float 版本的 ILU0
template<>
void CUSparseWrapper<Precision::Float32>::ilu0_setup(int n, int nz,
                                                     cusparseMatDescr_t mat_descr,
                                                     float* d_valsILU0,
                                                     const int* d_row_ptr,
                                                     const int* d_col_ind,
                                                     csrilu02Info_t ilu0_info,
                                                     void** d_buffer, int* buffer_size) {
    CHECK_CUSPARSE(cusparseScsrilu02_bufferSize(handle_, n, nz, mat_descr,
                                                 d_valsILU0, d_row_ptr, d_col_ind,
                                                 ilu0_info, buffer_size));
    if (*d_buffer == nullptr) {
        CHECK_CUDA(cudaMalloc(d_buffer, *buffer_size));
    }
    CHECK_CUSPARSE(cusparseScsrilu02_analysis(handle_, n, nz, mat_descr,
                                              d_valsILU0, d_row_ptr, d_col_ind,
                                              ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                              *d_buffer));
}

template<>
void CUSparseWrapper<Precision::Float32>::ilu0_compute(int n, int nz,
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

// double 版本的 ILU0
template<>
void CUSparseWrapper<Precision::Float64>::ilu0_setup(int n, int nz,
                                                     cusparseMatDescr_t mat_descr,
                                                     double* d_valsILU0,
                                                     const int* d_row_ptr,
                                                     const int* d_col_ind,
                                                     csrilu02Info_t ilu0_info,
                                                     void** d_buffer, int* buffer_size) {
    CHECK_CUSPARSE(cusparseDcsrilu02_bufferSize(handle_, n, nz, mat_descr,
                                                 d_valsILU0, d_row_ptr, d_col_ind,
                                                 ilu0_info, buffer_size));
    if (*d_buffer == nullptr) {
        CHECK_CUDA(cudaMalloc(d_buffer, *buffer_size));
    }
    CHECK_CUSPARSE(cusparseDcsrilu02_analysis(handle_, n, nz, mat_descr,
                                              d_valsILU0, d_row_ptr, d_col_ind,
                                              ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                              *d_buffer));
}

template<>
void CUSparseWrapper<Precision::Float64>::ilu0_compute(int n, int nz,
                                                       cusparseMatDescr_t mat_descr,
                                                       double* d_valsILU0,
                                                       const int* d_row_ptr,
                                                       const int* d_col_ind,
                                                       csrilu02Info_t ilu0_info,
                                                       void* d_buffer) {
    CHECK_CUSPARSE(cusparseDcsrilu02(handle_, n, nz, mat_descr,
                                     d_valsILU0, d_row_ptr, d_col_ind,
                                     ilu0_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                     d_buffer));
}

template<Precision P>
void CUSparseWrapper<P>::triangular_solve_setup(cusparseSpMatDescr_t matM,
                                                 cusparseSpSVDescr_t spsv_descr,
                                                 const cusparseDnVecDescr_t vecX,
                                                 const cusparseDnVecDescr_t vecY,
                                                 void** d_buffer, size_t* buffer_size) {
    Scalar one = ScalarConstants<P>::one();
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &one, matM, vecX, vecY,
                                          CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                          spsv_descr, buffer_size));
    if (*d_buffer == nullptr) {
        CHECK_CUDA(cudaMalloc(d_buffer, *buffer_size));
    }
    CHECK_CUSPARSE(cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matM, vecX, vecY,
                                        CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                        spsv_descr, *d_buffer));
}

template<Precision P>
void CUSparseWrapper<P>::triangular_solve(cusparseSpMatDescr_t matM,
                                          cusparseSpSVDescr_t spsv_descr,
                                          const cusparseDnVecDescr_t vecX,
                                          cusparseDnVecDescr_t vecY) {
    Scalar one = ScalarConstants<P>::one();
    CHECK_CUSPARSE(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &one, matM, vecX, vecY,
                                     CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                     spsv_descr));
}

// ============================================================================
// SparseMatrix 模板实现
// ============================================================================

template<Precision P>
SparseMatrix<P>::SparseMatrix(int rows, int cols, int nn)
    : rows(rows), cols(cols), nnz(nn),
      row_ptr(rows + 1), col_ind(nn), values(nn),
      d_row_ptr(nullptr), d_col_ind(nullptr), d_values(nullptr) {
}

template<Precision P>
SparseMatrix<P>::~SparseMatrix() {
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_ind) cudaFree(d_col_ind);
    if (d_values) cudaFree(d_values);
}

template<Precision P>
void SparseMatrix<P>::upload_to_gpu() {
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(typename SparseMatrix<P>::Scalar)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, col_ind.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, values.data(), nnz * sizeof(typename SparseMatrix<P>::Scalar),
                          cudaMemcpyHostToDevice));
}

template<Precision P>
cusparseSpMatDescr_t SparseMatrix<P>::create_sparse_descr() const {
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, rows, cols, nnz,
                                     d_row_ptr, d_col_ind, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    return matA;
}

// ============================================================================
// GPUVector 模板实现
// ============================================================================

template<Precision P>
GPUVector<P>::GPUVector(size_t n) : n(n), d_data(nullptr) {
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(typename GPUVector<P>::Scalar)));
}

template<Precision P>
GPUVector<P>::~GPUVector() {
    if (d_data) {
        cudaFree(d_data);
    }
}

// 移动构造函数
template<Precision P>
GPUVector<P>::GPUVector(GPUVector<P>&& other) noexcept
    : n(other.n), d_data(other.d_data) {
    other.n = 0;
    other.d_data = nullptr;
}

// 移动赋值运算符
template<Precision P>
GPUVector<P>& GPUVector<P>::operator=(GPUVector<P>&& other) noexcept {
    if (this != &other) {
        if (d_data) {
            cudaFree(d_data);
        }
        n = other.n;
        d_data = other.d_data;
        other.n = 0;
        other.d_data = nullptr;
    }
    return *this;
}

template<Precision P>
void GPUVector<P>::upload_from_host(const typename GPUVector<P>::Scalar* h_data) {
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(typename GPUVector<P>::Scalar), cudaMemcpyHostToDevice));
}

template<Precision P>
void GPUVector<P>::download_to_host(typename GPUVector<P>::Scalar* h_data) const {
    CHECK_CUDA(cudaMemcpy(h_data, d_data, n * sizeof(typename GPUVector<P>::Scalar), cudaMemcpyDeviceToHost));
}

template<Precision P>
cusparseDnVecDescr_t GPUVector<P>::create_dnvec_descr() const {
    cusparseDnVecDescr_t vec;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec, n, d_data, CudaDataType<P>::value));
    return vec;
}

// ============================================================================
// CPUOps 模板实现（CPU 向量运算）
// ============================================================================

namespace CPUOps {
    template<>
    float dot<Precision::Float32>(const std::vector<float>& x, const std::vector<float>& y) {
        float result = 0.0f;
        for (size_t i = 0; i < x.size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }

    template<>
    double dot<Precision::Float64>(const std::vector<double>& x, const std::vector<double>& y) {
        double result = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }

    template<>
    void axpy<Precision::Float32>(float alpha, const std::vector<float>& x, std::vector<float>& y) {
        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    template<>
    void axpy<Precision::Float64>(double alpha, const std::vector<double>& x, std::vector<double>& y) {
        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    template<>
    void scal<Precision::Float32>(float alpha, std::vector<float>& x) {
        for (size_t i = 0; i < x.size(); i++) {
            x[i] *= alpha;
        }
    }

    template<>
    void scal<Precision::Float64>(double alpha, std::vector<double>& x) {
        for (size_t i = 0; i < x.size(); i++) {
            x[i] *= alpha;
        }
    }

    template<>
    void copy<Precision::Float32>(const std::vector<float>& x, std::vector<float>& y) {
        y = x;
    }

    template<>
    void copy<Precision::Float64>(const std::vector<double>& x, std::vector<double>& y) {
        y = x;
    }

    template<>
    void spmv<Precision::Float32>(int n, const std::vector<int>& row_ptr,
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

    template<>
    void spmv<Precision::Float64>(int n, const std::vector<int>& row_ptr,
                                  const std::vector<int>& col_ind,
                                  const std::vector<double>& values,
                                  const std::vector<double>& x,
                                  std::vector<double>& y) {
        std::fill(y.begin(), y.end(), 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                y[i] += values[j] * x[col_ind[j]];
            }
        }
    }
}

// ============================================================================
// 显式模板实例化（防止链接错误）
// ============================================================================

// CUBLASWrapper
template class CUBLASWrapper<Precision::Float32>;
template class CUBLASWrapper<Precision::Float64>;

// CUSparseWrapper
template class CUSparseWrapper<Precision::Float32>;
template class CUSparseWrapper<Precision::Float64>;

// SparseMatrix
template class SparseMatrix<Precision::Float32>;
template class SparseMatrix<Precision::Float64>;

// GPUVector
template class GPUVector<Precision::Float32>;
template class GPUVector<Precision::Float64>;
