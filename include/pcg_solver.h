#ifndef PCG_SOLVER_H
#define PCG_SOLVER_H

#include "precision_traits.h"
#include "sparse_utils.h"
#include "preconditioner.h"
#include "solve_stats.h"
#include <memory>
#include <vector>

// ============================================================================
// PCG 求解器配置
// ============================================================================

struct PCGConfig {
    int max_iterations = 1000;
    double tolerance = 1e-8;         // 使用double
    bool use_preconditioner = true;  // 默认启用预处理
    PreconditionerType preconditioner_type = PreconditionerType::ILU0;  // 默认使用ILU0预条件子
    Backend backend = BACKEND_GPU;    // 默认使用 GPU
    Precision precision = Precision::Float64;  // 精度选择
    std::shared_ptr<AMGConfig> amg_config = nullptr;  // AMG 预条件子配置
};

// ============================================================================
// PCG 求解器
// ============================================================================

template<Precision P>
class PCGSolver {
public:
    using Scalar = ScalarT<P>;
    using Vector = std::vector<Scalar>;
    using Matrix = SparseMatrix<P>;
    using GPUVectorType = GPUVector<P>;
    using GPUPrec = PreconditionerBase<P, GPUVector<P>>;
    using CPUPrec = PreconditionerBase<P, std::vector<Scalar>>;

    PCGSolver(const PCGConfig& config);
    ~PCGSolver();

    // 求解 Ax = b
    SolveStats solve(const Matrix& A, const Vector& b, Vector& x);

private:
    struct CudaBufferDeleter {
        void operator()(void* ptr) const {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    PCGConfig config_;
    Backend backend_;

    // GPU 后端成员
    std::unique_ptr<CUBLASWrapper<P>> blas_;
    std::shared_ptr<CUSparseWrapper<P>> sparse_;
    std::shared_ptr<GPUPrec> gpu_preconditioner_;  // GPU 预条件子
    std::unique_ptr<GPUVector<P>> d_r_;
    std::unique_ptr<GPUVector<P>> d_z_;
    std::unique_ptr<GPUVector<P>> d_rm2_;
    std::unique_ptr<GPUVector<P>> d_zm2_;
    std::unique_ptr<GPUVector<P>> d_p_;
    std::unique_ptr<GPUVector<P>> d_Ap_;
    std::unique_ptr<void, CudaBufferDeleter> d_buffer_spMV_;
    size_t buffer_spMV_size_;

    // CPU 后端成员
    std::unique_ptr<CPUPrec> cpu_preconditioner_;  // CPU 预条件子

    // 方法
    void allocate_workspace(int n);
    void free_workspace();

    // 统一预条件子管理
    void setup_gpu_preconditioner(const Matrix& A);
    void setup_cpu_preconditioner(const Matrix& A);

    SolveStats solve_gpu(const Matrix& A, const Vector& b, Vector& x);
    SolveStats solve_cpu(const Matrix& A, const Vector& b, Vector& x);

    // ============================================================================
    // PCG 核心算法
    // ============================================================================

    // GPU 版本的核心算法
    SolveStats solve_core_gpu(
        int n,
        const Matrix& A,
        const Vector& b,
        Vector& x,
        const cusparseSpMatDescr_t matA,
        void* spmv_buffer);

    // CPU 版本的核心算法
    SolveStats solve_core_cpu(
        int n,
        const Matrix& A,
        const Vector& b,
        Vector& x);
};

// ============================================================================
// 向后兼容的类型别名
// ============================================================================

using SparseMatrixFloat = SparseMatrix<Precision::Float32>;
using SparseMatrixDouble = SparseMatrix<Precision::Float64>;
using GPUVectorFloat = GPUVector<Precision::Float32>;
using GPUVectorDouble = GPUVector<Precision::Float64>;
using PCGSolverFloat = PCGSolver<Precision::Float32>;
using PCGSolverDouble = PCGSolver<Precision::Float64>;
using CUBLASWrapperFloat = CUBLASWrapper<Precision::Float32>;
using CUBLASWrapperDouble = CUBLASWrapper<Precision::Float64>;
using CUSparseWrapperFloat = CUSparseWrapper<Precision::Float32>;
using CUSparseWrapperDouble = CUSparseWrapper<Precision::Float64>;
using GPUILUPreconditionerFloat = GPUILUPreconditioner<Precision::Float32>;
using GPUILUPreconditionerDouble = GPUILUPreconditioner<Precision::Float64>;
using CPUILUPreconditionerFloat = CPUILUPreconditioner<Precision::Float32>;
using CPUILUPreconditionerDouble = CPUILUPreconditioner<Precision::Float64>;

#endif // PCG_SOLVER_H
