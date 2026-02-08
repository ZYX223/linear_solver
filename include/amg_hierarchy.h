#ifndef AMG_HIERARCHY_H
#define AMG_HIERARCHY_H

#include "precision_traits.h"
#include "sparse_utils.h"
#include <vector>
#include <memory>

// 前向声明
struct AMGConfig;
struct SolveStats;

// ============================================================================
// AMG 层次结构
// ============================================================================

template<Precision P>
class AMGLevel {
public:
    using Scalar = PRECISION_SCALAR(P);

    int n;
    std::vector<int> row_ptr, col_ind;
    std::vector<Scalar> values;         // 当前层矩阵 A_l

    std::vector<int> P_row_ptr, P_col_ind;
    std::vector<Scalar> P_values;       // 插值算子 P (coarse to fine)

    std::vector<int> R_row_ptr, R_col_ind;
    std::vector<Scalar> R_values;       // 限制算子 R = P^T

    std::vector<Scalar> diag_inv;       // 对角线逆（用于松弛）

    AMGLevel(int size) : n(size) {}
};

template<Precision P>
class AMGHierarchy {
public:
    using Scalar = PRECISION_SCALAR(P);
    using Vector = std::vector<Scalar>;

    std::vector<std::unique_ptr<AMGLevel<P>>> levels;

    void setup(const SparseMatrix<P>& A, const AMGConfig& config);
    void cycle(const Vector& b, Vector& x, int level, const AMGConfig& config);

private:
    void solve_coarse_grid(const Vector& b, Vector& x);
};

#endif // AMG_HIERARCHY_H
