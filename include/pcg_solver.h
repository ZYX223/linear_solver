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
    float solve_time;  // 求解时间（秒）
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
// 统一 PCG 求解器（支持 CPU 和 GPU）
// ============================================================================

class PCGSolver {
public:
    PCGSolver(const PCGConfig& config);
    ~PCGSolver();

    // 设置预处理器（仅 GPU）
    void set_preconditioner(std::shared_ptr<PreconditionerBase<GPUVector>> prec);

    // 求解 Ax = b
    SolveStats solve(const SparseMatrix& A,
                    const std::vector<float>& b,
                    std::vector<float>& x);

private:
    // CUDA buffer 自定义删除器
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
    std::unique_ptr<CUBLASWrapper> blas_;
    std::shared_ptr<CUSparseWrapper> sparse_;
    std::shared_ptr<PreconditionerBase<GPUVector>> preconditioner_;
    std::unique_ptr<GPUVector> d_r_;
    std::unique_ptr<GPUVector> d_z_;
    std::unique_ptr<GPUVector> d_rm2_;
    std::unique_ptr<GPUVector> d_zm2_;
    std::unique_ptr<GPUVector> d_p_;
    std::unique_ptr<GPUVector> d_Ap_;
    std::unique_ptr<void, CudaBufferDeleter> d_buffer_spMV_;
    size_t buffer_spMV_size_;

    // CPU 后端成员
    std::unique_ptr<CPUILUPreconditioner> cpu_ilu_prec_;

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
