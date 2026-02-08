#include "amg_interpolation.h"
#include <algorithm>
#include <map>

// ============================================================================
// AMGInterpolation 模板实现
// ============================================================================

template<Precision P>
void AMGInterpolation<P>::transpose_csr(
    int rows, int cols,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<Scalar>& values,
    std::vector<int>& T_row_ptr,
    std::vector<int>& T_col_ind,
    std::vector<Scalar>& T_values
) {
    // 计算转置矩阵每行的非零元素数
    std::vector<int> T_row_nnz(cols, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int col = col_ind[j];
            T_row_nnz[col]++;
        }
    }

    // 构建转置矩阵的行指针
    T_row_ptr.resize(cols + 1);
    T_row_ptr[0] = 0;
    for (int i = 0; i < cols; ++i) {
        T_row_ptr[i + 1] = T_row_ptr[i] + T_row_nnz[i];
    }

    int total_nnz = T_row_ptr[cols];
    T_col_ind.resize(total_nnz);
    T_values.resize(total_nnz);

    // 临时计数器
    std::vector<int> current_row_count(cols, 0);

    // 填充转置矩阵
    for (int i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int col = col_ind[j];
            int idx = T_row_ptr[col] + current_row_count[col];
            T_col_ind[idx] = i;
            T_values[idx] = values[j];
            current_row_count[col]++;
        }
    }
}

template<Precision P>
void AMGInterpolation<P>::galerkin_product(
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
) {
    // Galerkin乘积: A_coarse = R * A * P
    // 分两步: 1) temp = A * P, 2) A_coarse = R * temp

    // 第一步：计算 temp = A * P
    // temp 的大小: n_fine x n_coarse
    std::vector<std::map<int, Scalar>> temp(n_fine);

    for (int i = 0; i < n_fine; ++i) {
        for (int j = A_row_ptr[i]; j < A_row_ptr[i + 1]; ++j) {
            int k = A_col_ind[j];
            Scalar A_ik = A_values[j];

            // A[i,k] * P[k,:]
            for (int p = P_row_ptr[k]; p < P_row_ptr[k + 1]; ++p) {
                int col = P_col_ind[p];
                temp[i][col] += A_ik * P_values[p];
            }
        }
    }

    // 第二步：计算 A_coarse = R * temp
    std::vector<std::map<int, Scalar>> A_coarse_map(n_coarse);

    for (int i = 0; i < n_coarse; ++i) {
        for (int j = R_row_ptr[i]; j < R_row_ptr[i + 1]; ++j) {
            int k = R_col_ind[j];
            Scalar R_ik = R_values[j];

            // R[i,k] * temp[k,:]
            for (const auto& entry : temp[k]) {
                A_coarse_map[i][entry.first] += R_ik * entry.second;
            }
        }
    }

    // 转换为CSR格式
    // 计算非零元素数
    std::vector<int> row_nnz(n_coarse);
    for (int i = 0; i < n_coarse; ++i) {
        row_nnz[i] = A_coarse_map[i].size();
    }

    Ac_row_ptr.resize(n_coarse + 1);
    Ac_row_ptr[0] = 0;
    for (int i = 0; i < n_coarse; ++i) {
        Ac_row_ptr[i + 1] = Ac_row_ptr[i] + row_nnz[i];
    }

    int total_nnz = Ac_row_ptr[n_coarse];
    Ac_col_ind.resize(total_nnz);
    Ac_values.resize(total_nnz);

    // 填充数据
    int idx = 0;
    for (int i = 0; i < n_coarse; ++i) {
        for (const auto& entry : A_coarse_map[i]) {
            Ac_col_ind[idx] = entry.first;
            Ac_values[idx] = entry.second;
            idx++;
        }
    }
}

template<Precision P>
void AMGInterpolation<P>::compute_diag_inv(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<Scalar>& values,
    std::vector<Scalar>& diag_inv
) {
    diag_inv.resize(n);

    for (int i = 0; i < n; ++i) {
        Scalar diag = Scalar(0);
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_ind[j] == i) {
                diag = values[j];
                break;
            }
        }
        diag_inv[i] = (diag != Scalar(0)) ? Scalar(1) / diag : Scalar(0);
    }
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class AMGInterpolation<Precision::Float32>;
template class AMGInterpolation<Precision::Float64>;
