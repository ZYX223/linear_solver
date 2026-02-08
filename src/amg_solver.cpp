#include "amg_solver.h"
#include "amg_hierarchy.h"
#include "sparse_utils.h"
#include <chrono>
#include <cmath>

// ============================================================================
// AMGSolver 模板实现
// ============================================================================

template<Precision P>
AMGSolver<P>::AMGSolver(const AMGConfig& config)
    : config_(config), hierarchy_(nullptr) {
}

template<Precision P>
AMGSolver<P>::~AMGSolver() {
    // 智能指针自动清理
}

template<Precision P>
SolveStats AMGSolver<P>::solve(const SparseMatrix<P>& A,
                                const std::vector<Scalar>& b,
                                std::vector<Scalar>& x) {
    return solve_cpu(A, b, x);
}

template<Precision P>
void AMGSolver<P>::compute_residual(const Matrix& A, const Vector& x,
                                     const Vector& b, Vector& r) {
    CPUOps::spmv<P>(A.rows, A.row_ptr, A.col_ind, A.values, x, r);
    for (int i = 0; i < A.rows; ++i) {
        r[i] = b[i] - r[i];
    }
}

template<Precision P>
SolveStats AMGSolver<P>::solve_cpu(const Matrix& A, const Vector& b, Vector& x) {
    int n = A.rows;

    // 构建层次结构
    if (!hierarchy_) {
        hierarchy_ = std::make_unique<AMGHierarchy<P>>();
        hierarchy_->setup(A, config_);
    }

    // 初始化解向量
    if (x.size() != static_cast<size_t>(n)) {
        x.resize(n, Scalar(0));
    }

    // 计算初始残差
    Vector r(n);
    compute_residual(A, x, b, r);
    double r0_norm = std::sqrt(CPUOps::dot<P>(r, r));

    int k = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // 主迭代循环
    while (k < config_.max_iterations) {
        // 执行V-cycle
        hierarchy_->cycle(b, x, 0, config_);

        // 计算残差
        compute_residual(A, x, b, r);
        double r_norm = std::sqrt(CPUOps::dot<P>(r, r));

        // 检查收敛
        if (r_norm / r0_norm < config_.tolerance) {
            break;
        }

        k++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    // 准备统计信息
    SolveStats stats;
    stats.iterations = k;
    stats.final_residual = std::sqrt(CPUOps::dot<P>(r, r)) / r0_norm;
    stats.converged = (k <= config_.max_iterations);
    stats.solve_time = solve_time;

    return stats;
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class AMGSolver<Precision::Float32>;
template class AMGSolver<Precision::Float64>;
