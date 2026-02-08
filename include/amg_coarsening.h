#ifndef AMG_COARSENING_H
#define AMG_COARSENING_H

#include "precision_traits.h"
#include "sparse_utils.h"
#include <vector>

// ============================================================================
// Smoothed Aggregation 粗化策略
// ============================================================================

template<Precision P>
class SmoothedAggregation {
public:
    using Scalar = PRECISION_SCALAR(P);

    // 构建聚合（基于强连接）
    static std::vector<int> build_aggregates(
        const SparseMatrix<P>& A,
        double eps
    );

    // 构建常数插值算子
    static void build_constant_interpolation(
        const SparseMatrix<P>& A,
        const std::vector<int>& aggregates,
        std::vector<int>& P_row_ptr,
        std::vector<int>& P_col_ind,
        std::vector<Scalar>& P_values
    );

    // 平滑插值算子: P_smooth = (I - ω D^{-1} A) * P
    static void smooth_interpolation(
        const SparseMatrix<P>& A,
        std::vector<int>& P_row_ptr,
        std::vector<int>& P_col_ind,
        std::vector<Scalar>& P_values,
        Scalar omega = Scalar(2.0/3.0)
    );
};

#endif // AMG_COARSENING_H
