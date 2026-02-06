#ifndef PCG_SOLVER_H
#define PCG_SOLVER_H

#include "sparse_utils.h"
#include "preconditioner.h"
#include <memory>
#include <vector>

// ============================================================================
// PCG 求解器统计信息
// ============================================================================

struct SolveStats {
    int iterations;
    float final_residual;
    bool converged;
};

// ============================================================================
// 后端类型
// ============================================================================

enum Backend {
    BACKEND_GPU,
    BACKEND_CPU
};

// ============================================================================
// PCG 求解器配置
// ============================================================================

struct PCGConfig {
    int max_iterations = 1000;
    float tolerance = 1e-12f;
    bool use_preconditioner = false;  // 默认关闭预处理
    Backend backend = BACKEND_GPU;    // 默认使用 GPU
};

// ============================================================================
// CPU 向量运算（辅助函数）- 声明
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

// ============================================================================
// CPU ILU(0) 预处理器
// ============================================================================

class CPUIILUPreconditioner {
public:
    CPUIILUPreconditioner();
    ~CPUIILUPreconditioner();

    void setup(int n, const std::vector<int>& row_ptr,
              const std::vector<int>& col_ind,
              const std::vector<float>& values);

    void apply(const std::vector<float>& r, std::vector<float>& z) const;

private:
    int n_;
    std::vector<int> row_ptr_;
    std::vector<int> col_ind_;
    std::vector<float> values_;  // ILU(0) 分解后的值

    void forward_substitute(const std::vector<float>& b, std::vector<float>& y) const;
    void backward_substitute(const std::vector<float>& y, std::vector<float>& x) const;
};

// ============================================================================
// 统一 PCG 求解器（支持 CPU 和 GPU）
// ============================================================================

class PCGSolver {
public:
    PCGSolver(const PCGConfig& config);
    ~PCGSolver();

    // 设置预处理器（仅 GPU）
    void set_preconditioner(std::shared_ptr<Preconditioner> prec);

    // 求解 Ax = b
    SolveStats solve(const SparseMatrix& A,
                    const std::vector<float>& b,
                    std::vector<float>& x);

private:
    PCGConfig config_;
    Backend backend_;

    // GPU 后端成员
    std::shared_ptr<CUBLASWrapper> blas_;
    std::shared_ptr<CUSparseWrapper> sparse_;
    std::shared_ptr<Preconditioner> preconditioner_;
    GPUVector* d_r_;
    GPUVector* d_z_;
    GPUVector* d_rm2_;
    GPUVector* d_zm2_;
    GPUVector* d_p_;
    GPUVector* d_Ap_;
    void* d_buffer_spMV_;
    size_t buffer_spMV_size_;

    // CPU 后端成员
    std::shared_ptr<CPUIILUPreconditioner> cpu_ilu_prec_;

    // 方法
    void allocate_workspace(int n);
    void free_workspace();
    SolveStats solve_gpu(const SparseMatrix& A,
                        const std::vector<float>& b,
                        std::vector<float>& x);
    SolveStats solve_cpu(const SparseMatrix& A,
                        const std::vector<float>& b,
                        std::vector<float>& x);
};

#endif // PCG_SOLVER_H
