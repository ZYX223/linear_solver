#ifndef AMG_SOLVER_H
#define AMG_SOLVER_H

#include "precision_traits.h"
#include "sparse_utils.h"
#include "solve_stats.h"
#include <memory>
#include <vector>

// ============================================================================
// AMG 求解器配置
// ============================================================================

struct AMGConfig {
    int max_iterations = 100;           // AMG通常需要更少迭代
    double tolerance = 1e-12;

    // AMG特定参数
    int max_levels = 10;                // 最大层数
    int coarse_grid_size = 50;          // 最粗网格大小阈值
    int pre_smooth_steps = 1;           // 前光滑步数
    int post_smooth_steps = 1;          // 后光滑步数
    double aggregation_eps = 0.001;     // 聚合阈值

    Precision precision = Precision::Float32;
};

// ============================================================================
// AMG 求解器（前向声明）
// ============================================================================

template<Precision P>
class AMGHierarchy;

template<Precision P>
class AMGSolver {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Vector = std::vector<Scalar>;
    using Matrix = SparseMatrix<P>;

    AMGSolver(const AMGConfig& config);
    ~AMGSolver();

    // 主求解接口（与PCG一致）
    SolveStats solve(const Matrix& A, const Vector& b, Vector& x);

private:
    AMGConfig config_;
    std::unique_ptr<AMGHierarchy<P>> hierarchy_;

    SolveStats solve_cpu(const Matrix& A, const Vector& b, Vector& x);
    void compute_residual(const Matrix& A, const Vector& x,
                         const Vector& b, Vector& r);
};

// ============================================================================
// 类型别名
// ============================================================================

using AMGSolverFloat = AMGSolver<Precision::Float32>;
using AMGSolverDouble = AMGSolver<Precision::Float64>;

#endif // AMG_SOLVER_H
