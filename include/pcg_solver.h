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
    double tolerance = 1e-12;         // 使用double
    bool use_preconditioner = false;  // 默认关闭预处理
    Backend backend = BACKEND_GPU;    // 默认使用 GPU
    Precision precision = Precision::Float32;  // 新增：精度选择
};

// ============================================================================
// 统一 PCG 求解器（支持 CPU 和 GPU，模板化）
// ============================================================================

template<Precision P>
class PCGSolver {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Vector = std::vector<Scalar>;
    using Matrix = SparseMatrix<P>;
    using GPUVectorType = GPUVector<P>;
    using GPUPrec = PreconditionerBase<P, GPUVector<P>>;

    PCGSolver(const PCGConfig& config);
    ~PCGSolver();

    // 设置预处理器（仅 GPU）
    void set_preconditioner(std::shared_ptr<GPUPrec> prec);

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
    std::shared_ptr<GPUPrec> preconditioner_;
    std::unique_ptr<GPUVector<P>> d_r_;
    std::unique_ptr<GPUVector<P>> d_z_;
    std::unique_ptr<GPUVector<P>> d_rm2_;
    std::unique_ptr<GPUVector<P>> d_zm2_;
    std::unique_ptr<GPUVector<P>> d_p_;
    std::unique_ptr<GPUVector<P>> d_Ap_;
    std::unique_ptr<void, CudaBufferDeleter> d_buffer_spMV_;
    size_t buffer_spMV_size_;

    // CPU 后端成员
    std::unique_ptr<CPUILUPreconditioner<P>> cpu_ilu_prec_;

    // 方法
    void allocate_workspace(int n);
    void free_workspace();

    SolveStats solve_gpu(const Matrix& A, const Vector& b, Vector& x);
    SolveStats solve_cpu(const Matrix& A, const Vector& b, Vector& x);
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
