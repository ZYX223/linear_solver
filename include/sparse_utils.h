#ifndef SPARSE_UTILS_H
#define SPARSE_UTILS_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>

// 常量声明
extern const float floatone;
extern const float floatzero;

#define CHECK_CUDA(val) do { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(val) do { \
    cublasStatus_t err = (val); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS Error: %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(val) do { \
    cusparseStatus_t err = (val); \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        printf("CUSPARSE Error: %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// cuBLAS封装
class CUBLASWrapper {
public:
    CUBLASWrapper();
    ~CUBLASWrapper();

    cublasHandle_t handle() const { return handle_; }

    // 向量点积: return x^T * y
    float dot(int n, const float* x, const float* y);

    // 向量更新: y = alpha * x + y
    void axpy(int n, float alpha, const float* x, float* y);

    // 向量复制: y = x
    void copy(int n, const float* x, float* y);

    // 向量缩放: x = alpha * x
    void scal(int n, float alpha, float* x);

private:
    cublasHandle_t handle_;
};

// cuSPARSE封装
class CUSparseWrapper {
public:
    CUSparseWrapper();
    ~CUSparseWrapper();

    cusparseHandle_t handle() const { return handle_; }

    // 稀疏矩阵向量乘法: y = alpha * A * x + beta * y
    void spmv(cusparseSpMatDescr_t matA,
              const cusparseDnVecDescr_t vecX,
              cusparseDnVecDescr_t vecY,
              float alpha, float beta, void* buffer);

    // ILU(0)分解
    void ilu0_setup(int n, int nz,
                    cusparseMatDescr_t mat_descr,
                    float* d_valsILU0,
                    const int* d_row_ptr,
                    const int* d_col_ind,
                    csrilu02Info_t ilu0_info,
                    void** d_buffer, int* buffer_size);

    void ilu0_compute(int n, int nz,
                      cusparseMatDescr_t mat_descr,
                      float* d_valsILU0,
                      const int* d_row_ptr,
                      const int* d_col_ind,
                      csrilu02Info_t ilu0_info,
                      void* d_buffer);

    // 三角求解: y = alpha * M^(-1) * x (其中M是L或U)
    void triangular_solve_setup(cusparseSpMatDescr_t matM,
                                 cusparseSpSVDescr_t spsv_descr,
                                 const cusparseDnVecDescr_t vecX,
                                 const cusparseDnVecDescr_t vecY,
                                 void** d_buffer, size_t* buffer_size);

    void triangular_solve(cusparseSpMatDescr_t matM,
                          cusparseSpSVDescr_t spsv_descr,
                          const cusparseDnVecDescr_t vecX,
                          cusparseDnVecDescr_t vecY);

private:
    cusparseHandle_t handle_;
};

// 稀疏矩阵（CSR格式）
class SparseMatrix {
public:
    int rows, cols, nnz;
    std::vector<int> row_ptr;   // 行指针
    std::vector<int> col_ind;   // 列索引
    std::vector<float> values;  // 非零值

    // GPU内存
    int *d_row_ptr, *d_col_ind;
    float *d_values;

    SparseMatrix(int rows, int cols, int nn);
    ~SparseMatrix();

    // 上传到GPU
    void upload_to_gpu();

    // 创建cuSPARSE矩阵描述符
    cusparseSpMatDescr_t create_sparse_descr() const;
};

// GPU向量
class GPUVector {
public:
    size_t n;  // 使用 size_t 避免负数
    float* d_data;

    GPUVector(size_t n);
    ~GPUVector();

    // 禁止拷贝（CUDA 资源不应该被拷贝）
    GPUVector(const GPUVector&) = delete;
    GPUVector& operator=(const GPUVector&) = delete;

    // 支持移动（提高性能）
    GPUVector(GPUVector&& other) noexcept;
    GPUVector& operator=(GPUVector&& other) noexcept;

    // 从主机数据复制
    void upload_from_host(const float* h_data);
    void download_to_host(float* h_data) const;

    // 创建cuSPARSE向量描述符
    cusparseDnVecDescr_t create_dnvec_descr() const;
};

// ============================================================================
// CPU 向量运算（辅助函数）
// ============================================================================

namespace CPUOps {
    float dot(const std::vector<float>& x, const std::vector<float>& y);
    void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y);
    void scal(float alpha, std::vector<float>& x);
    void copy(const std::vector<float>& x, std::vector<float>& y);
    void spmv(int n, const std::vector<int>& row_ptr,
              const std::vector<int>& col_ind,
              const std::vector<float>& values,
              const std::vector<float>& x,
              std::vector<float>& y);
}

#endif // SPARSE_UTILS_H
