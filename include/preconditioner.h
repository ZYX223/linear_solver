#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "sparse_utils.h"
#include "precision_traits.h"
#include "amg_config.h"
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <memory>
#include <vector>
#include <unordered_map>

// 抑制 cuSPARSE 废弃 API 警告 (csrilu02Info_t 等)
// 这些宏定义在 sparse_utils.h 中

// amgcl 库头文件（在 AMG 预条件子类型定义之前）
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>  // CPU AMG
#include <amgcl/relaxation/damped_jacobi.hpp> // GPU AMG (Jacobi)

// ============================================================================
// 预条件子类型定义
// ============================================================================

/// 预条件子类型
enum class PreconditionerType {
    NONE,     // 无预条件（纯 CG）
    JACOBI,   // 对角预条件
    ILU0,     // 不完全 LU 分解
    IC0,      // 不完全 Cholesky 分解（对称正定矩阵专用）
    AMG       // 代数多重网格（自动选择 CPU 或 GPU 后端）
};

// ============================================================================
// 预处理器模板基类（统一 CPU/GPU 接口）
// ============================================================================
template<Precision P, typename VectorType>
class PreconditionerBase {
public:
    using Scalar = PRECISION_SCALAR(P);
    virtual ~PreconditionerBase() = default;

    // 初始化预处理器（如ILU分解）
    virtual void setup(const SparseMatrix<P>& A) = 0;

    // 应用预处理器: z = M^(-1) * r
    virtual void apply(const VectorType& r, VectorType& z) const = 0;
};

// ============================================================================
// GPU 预处理器
// ============================================================================

// ILU(0)预处理器（GPU，模板化）
template<Precision P>
class GPUILUPreconditioner : public PreconditionerBase<P, GPUVector<P>> {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Matrix = SparseMatrix<P>;
    using Vector = GPUVector<P>;

    GPUILUPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse);
    ~GPUILUPreconditioner();

    void setup(const Matrix& A) override;
    void apply(const Vector& r, Vector& z) const override;

private:
    std::shared_ptr<CUSparseWrapper<P>> sparse_;

    int rows_, nnz_;

    // ILU(0)分解的L和U因子（存储在同一个数组中）
    Scalar* d_valsILU0_;

    // 矩阵描述符
    cusparseMatDescr_t matLU_descr_;
    CUSPARSE_DEPRECATED_DISABLE_BEGIN
    csrilu02Info_t ilu0_info_;
    CUSPARSE_DEPRECATED_DISABLE_END

    // ILU(0) buffer
    void* d_bufferILU0_;
    int bufferILU0_size_;

    // L和U矩阵描述符（分别用于三角求解）
    cusparseSpMatDescr_t matL_;
    cusparseSpMatDescr_t matU_;

    // 三角求解描述符和buffer
    cusparseSpSVDescr_t spsvDescrL_;
    cusparseSpSVDescr_t spsvDescrU_;
    void* d_bufferL_;
    void* d_bufferU_;
    size_t bufferSizeL_;
    size_t bufferSizeU_;

    // 辅助向量（用于L和U之间的中间结果）
    Scalar* d_y_;

    bool is_setup_;
};

// ============================================================================
// GPU IC(0) 预处理器（不完全 Cholesky 分解）
// 实现：CPU 分解得到上三角 R + GPU 三角求解
// 使用 NVIDIA 推荐方式：R^T * R 分解，用 TRANSPOSE 操作避免显式构建 R^T
// ============================================================================

template<Precision P>
class GPUIPCPreconditioner : public PreconditionerBase<P, GPUVector<P>> {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Matrix = SparseMatrix<P>;
    using Vector = GPUVector<P>;

    GPUIPCPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse);
    ~GPUIPCPreconditioner();

    void setup(const Matrix& A) override;
    void apply(const Vector& r, Vector& z) const override;

private:
    std::shared_ptr<CUSparseWrapper<P>> sparse_;

    int rows_;
    int nnz_;

    // 上三角 R 因子（CSR 格式，CPU 端用于分解）
    std::vector<int> row_ptr_;
    std::vector<int> col_ind_;
    std::vector<Scalar> values_;

    // GPU 端数据（只有一个 R 矩阵）
    int* d_row_ptr_;
    int* d_col_ind_;
    Scalar* d_values_;

    // R 矩阵描述符（用于 R^T 和 R 的三角求解）
    cusparseSpMatDescr_t matR_;

    // 三角求解描述符和 buffer
    cusparseSpSVDescr_t spsvDescrRt_;  // R^T 求解（TRANSPOSE）
    cusparseSpSVDescr_t spsvDescrR_;   // R 求解（NON_TRANSPOSE）
    void* d_bufferRt_;
    void* d_bufferR_;
    size_t bufferSizeRt_;
    size_t bufferSizeR_;

    // 辅助向量
    Scalar* d_t_;

    bool is_setup_;
};

// ============================================================================
// CPU ILU(0) 预处理器
// ============================================================================

template<Precision P>
class CPUILUPreconditioner : public PreconditionerBase<P, std::vector<PRECISION_SCALAR(P)>> {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Matrix = SparseMatrix<P>;
    using Vector = std::vector<Scalar>;

    CPUILUPreconditioner();
    ~CPUILUPreconditioner() override;

    // 实现基类接口
    void setup(const Matrix& A) override;
    void apply(const Vector& r, Vector& z) const override;

    // 重载版本（保持向后兼容）
    void setup(int n, const std::vector<int>& row_ptr,
              const std::vector<int>& col_ind,
              const std::vector<Scalar>& values);

private:
    int n_;
    std::vector<int> row_ptr_;
    std::vector<int> col_ind_;
    std::vector<Scalar> values_;  // ILU(0) 分解后的值

    void forward_substitute(const Vector& b, Vector& y) const;
    void backward_substitute(const Vector& y, Vector& x) const;
};

// ============================================================================
// CPU IC(0) 预处理器（不完全 Cholesky 分解，对称正定矩阵专用）
// ============================================================================

template<Precision P>
class CPUIPCPreconditioner : public PreconditionerBase<P, std::vector<PRECISION_SCALAR(P)>> {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Matrix = SparseMatrix<P>;
    using Vector = std::vector<Scalar>;

    CPUIPCPreconditioner();
    ~CPUIPCPreconditioner() override = default;

    void setup(const Matrix& A) override;
    void apply(const Vector& r, Vector& z) const override;

private:
    int n_;
    // R 因子（上三角，CSR 格式）- 用于后向代入
    std::vector<int> row_ptr_;
    std::vector<int> col_ind_;
    std::vector<Scalar> values_;

    // L = R^T 因子（下三角，CSR 格式）- 用于前向代入
    std::vector<int> l_row_ptr_;
    std::vector<int> l_col_ind_;
    std::vector<Scalar> l_values_;

    // 对角线元素（单独存储，加速访问）
    std::vector<Scalar> diag_;

    void forward_substitute(const Vector& b, Vector& y) const;
    void backward_substitute(const Vector& y, Vector& x) const;
};

// ============================================================================
// CPU AMG 预条件子
// ============================================================================

template<Precision P>
class CPUAMGPreconditioner : public PreconditionerBase<P, std::vector<PRECISION_SCALAR(P)>> {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Vector = std::vector<Scalar>;
    using Matrix = SparseMatrix<P>;

    explicit CPUAMGPreconditioner(std::shared_ptr<AMGConfig> config);
    ~CPUAMGPreconditioner() override = default;

    void setup(const Matrix& A) override;
    void apply(const Vector& r, Vector& z) const override;

private:
    std::shared_ptr<AMGConfig> config_;

    // amgcl 类型定义（使用编译时类型，无需 Boost）
    typedef amgcl::backend::builtin<Scalar> Backend;
    typedef amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel
    > AMGPreconditioner;

    std::shared_ptr<typename Backend::matrix> A_amgcl_;
    std::shared_ptr<AMGPreconditioner> amg_prec_;

    // 适配后的矩阵数据（amgcl 使用 ptrdiff_t）
    std::vector<ptrdiff_t> ptr_;
    std::vector<ptrdiff_t> col_;
    std::vector<Scalar> val_;

    bool is_setup_;
};

// ============================================================================
// GPU AMG 预条件子
// 使用 AMGCL 原生 CUDA 后端 + Damped Jacobi 松弛
// ============================================================================

#ifndef __CPU_ONLY__

template<Precision P>
class GPUAMGPreconditioner : public PreconditionerBase<P, GPUVector<P>> {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Vector = GPUVector<P>;
    using Matrix = SparseMatrix<P>;

    GPUAMGPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse,
                          std::shared_ptr<AMGConfig> config);
    ~GPUAMGPreconditioner() override;

    void setup(const Matrix& A) override;
    void apply(const Vector& r, Vector& z) const override;

private:
    std::shared_ptr<CUSparseWrapper<P>> sparse_;
    std::shared_ptr<AMGConfig> config_;

    // 使用 CUDA backend + Damped Jacobi（GPU 友好的松弛方法）
    typedef amgcl::backend::cuda<Scalar> Backend;
    typedef amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::damped_jacobi
    > AMGPreconditioner;

    // CPU端矩阵用于构建层级（amgcl在构建时使用builtin格式）
    typedef amgcl::backend::builtin<Scalar> BuildBackend;
    std::shared_ptr<typename BuildBackend::matrix> A_build_;
    std::shared_ptr<AMGPreconditioner> amg_prec_;

    // Backend 参数（cusparse handle）
    typename Backend::params backend_params_;

    // 矩阵维度
    int n_ = 0;
    int nnz_ = 0;

    // 适配后的矩阵数据（用于构建阶段）
    std::vector<ptrdiff_t> ptr_;
    std::vector<ptrdiff_t> col_;
    std::vector<Scalar> val_;

    // GPU 端向量（thrust::device_vector，用于 amgcl CUDA 后端）
    mutable thrust::device_vector<Scalar> r_dev_;
    mutable thrust::device_vector<Scalar> z_dev_;

    bool is_setup_;
};

#endif // __CPU_ONLY__

#endif // PRECONDITIONER_H
