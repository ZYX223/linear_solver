#ifndef AMG_SOLVER_H
#define AMG_SOLVER_H

#include "amg_config.h"
#include "precision_traits.h"
#include "sparse_utils.h"
#include "solve_stats.h"
#include <memory>
#include <vector>

// ============================================================================
// AMG 求解器（使用内嵌的 amgcl 库）
// ============================================================================

// 不透明接口类
class AMGSolverImplBase {
public:
    virtual ~AMGSolverImplBase() = default;
    virtual void setup(const void* A, Precision precision) = 0;
    virtual std::pair<int, double> solve(const void* b, void* x, Precision precision) = 0;
};

template<Precision P>
class AMGSolver {
public:
    using Scalar = ScalarT<P>;
    using Vector = std::vector<Scalar>;
    using Matrix = SparseMatrix<P>;

    AMGSolver(const AMGConfig& config);
    ~AMGSolver();

    // 禁止拷贝
    AMGSolver(const AMGSolver&) = delete;
    AMGSolver& operator=(const AMGSolver&) = delete;

    // 主求解接口（与其他求解器一致）
    SolveStats solve(const Matrix& A, const Vector& b, Vector& x);

private:
    AMGConfig config_;
    std::unique_ptr<AMGSolverImplBase> impl_;
};

// ============================================================================
// 类型别名
// ============================================================================

using AMGSolverFloat = AMGSolver<Precision::Float32>;
using AMGSolverDouble = AMGSolver<Precision::Float64>;

#endif // AMG_SOLVER_H
