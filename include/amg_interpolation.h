#ifndef AMG_INTERPOLATION_H
#define AMG_INTERPOLATION_H

#include "precision_traits.h"
#include <vector>

// ============================================================================
// AMG 插值和限制算子辅助函数
// ============================================================================

template<Precision P>
class AMGInterpolation {
public:
    using Scalar = PRECISION_SCALAR(P);

    // 矩阵转置（CSR格式）：R = P^T
    static void transpose_csr(
        int rows, int cols,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        const std::vector<Scalar>& values,
        std::vector<int>& T_row_ptr,
        std::vector<int>& T_col_ind,
        std::vector<Scalar>& T_values
    );

    // Galerkin乘积: A_coarse = R * A * P
    static void galerkin_product(
        int n_fine, int n_coarse,
        const std::vector<int>& R_row_ptr,
        const std::vector<int>& R_col_ind,
        const std::vector<Scalar>& R_values,
        const std::vector<int>& A_row_ptr,
        const std::vector<int>& A_col_ind,
        const std::vector<Scalar>& A_values,
        const std::vector<int>& P_row_ptr,
        const std::vector<int>& P_col_ind,
        const std::vector<Scalar>& P_values,
        std::vector<int>& Ac_row_ptr,
        std::vector<int>& Ac_col_ind,
        std::vector<Scalar>& Ac_values
    );

    // 计算对角线逆矩阵
    static void compute_diag_inv(
        int n,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        const std::vector<Scalar>& values,
        std::vector<Scalar>& diag_inv
    );
};

#endif // AMG_INTERPOLATION_H
