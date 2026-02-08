#include "amg_coarsening.h"
#include "amg_interpolation.h"
#include <algorithm>
#include <cmath>
#include <set>

// ============================================================================
// SmoothedAggregation 模板实现
// ============================================================================

template<Precision P>
std::vector<int> SmoothedAggregation<P>::build_aggregates(
    const SparseMatrix<P>& A,
    double eps
) {
    using Scalar = typename ScalarType<P>::type;
    int n = A.rows;

    std::vector<int> aggregates(n, -1);  // -1表示未分配
    int num_aggregates = 0;

    // 第一步：标记强连接
    // 对于对称正定矩阵，使用对角占度判断强连接
    std::vector<std::vector<int>> strong_neighbors(n);

    for (int i = 0; i < n; ++i) {
        Scalar diag = Scalar(0);
        Scalar max_off_diag = Scalar(0);

        // 找对角元素和非对角元素最大值
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_ind[j];
            if (col == i) {
                diag = A.values[j];
            } else {
                max_off_diag = std::max(max_off_diag,
                                       std::abs(A.values[j]));
            }
        }

        // 判断强连接阈值
        Scalar threshold = Scalar(eps * diag);

        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int col = A.col_ind[j];
            if (col != i && std::abs(A.values[j]) > threshold) {
                strong_neighbors[i].push_back(col);
            }
        }
    }

    // 第二步：基于强连接构建聚合
    std::vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            // 创建新聚合
            aggregates[i] = num_aggregates;
            visited[i] = true;

            // 将强连接邻居加入同一聚合
            for (int neighbor : strong_neighbors[i]) {
                if (!visited[neighbor]) {
                    aggregates[neighbor] = num_aggregates;
                    visited[neighbor] = true;
                }
            }

            num_aggregates++;
        }
    }

    // 确保所有点都被分配（即使是孤立的点）
    for (int i = 0; i < n; ++i) {
        if (aggregates[i] == -1) {
            aggregates[i] = num_aggregates;
            num_aggregates++;
        }
    }

    return aggregates;
}

template<Precision P>
void SmoothedAggregation<P>::build_constant_interpolation(
    const SparseMatrix<P>& A,
    const std::vector<int>& aggregates,
    std::vector<int>& P_row_ptr,
    std::vector<int>& P_col_ind,
    std::vector<Scalar>& P_values
) {
    int n_fine = A.rows;
    int n_coarse = 0;

    // 计算粗网格大小
    for (int agg : aggregates) {
        n_coarse = std::max(n_coarse, agg + 1);
    }

    // 计算每行的非零元素数
    std::vector<int> row_nnz(n_fine, 1);
    P_row_ptr.resize(n_fine + 1);
    P_row_ptr[0] = 0;

    for (int i = 0; i < n_fine; ++i) {
        P_row_ptr[i + 1] = P_row_ptr[i] + row_nnz[i];
    }

    int total_nnz = P_row_ptr[n_fine];
    P_col_ind.resize(total_nnz);
    P_values.resize(total_nnz, Scalar(1));

    // 构建分段常数插值算子：每个细网格点对应到其聚合中心
    for (int i = 0; i < n_fine; ++i) {
        int idx = P_row_ptr[i];
        P_col_ind[idx] = aggregates[i];
        P_values[idx] = Scalar(1);
    }
}

template<Precision P>
void SmoothedAggregation<P>::smooth_interpolation(
    const SparseMatrix<P>& A,
    std::vector<int>& P_row_ptr,
    std::vector<int>& P_col_ind,
    std::vector<Scalar>& P_values,
    Scalar omega
) {
    // 平滑插值算子: P_smooth = (I - ω D^{-1} A) * P
    // 其中 D 是 A 的对角矩阵

    int n_fine = A.rows;
    int n_coarse = 0;

    // 计算粗网格大小
    for (int j = 0; j < P_col_ind.size(); ++j) {
        n_coarse = std::max(n_coarse, P_col_ind[j] + 1);
    }

    // 计算 D^{-1} A
    std::vector<Scalar> diag_inv(n_fine);
    AMGInterpolation<P>::compute_diag_inv(n_fine, A.row_ptr, A.col_ind,
                                          A.values, diag_inv);

    // 构建临时矩阵 temp = -ω D^{-1} A * P
    std::vector<Scalar> temp(n_fine * n_coarse, Scalar(0));

    // 计算 D^{-1} A * P
    for (int i = 0; i < n_fine; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int k = A.col_ind[j];
            Scalar val = omega * diag_inv[i] * A.values[j];

            // 将 (D^{-1} A) 的第 i 行第 k 列乘以 P 的第 k 行
            for (int p = P_row_ptr[k]; p < P_row_ptr[k + 1]; ++p) {
                int col = P_col_ind[p];
                temp[i * n_coarse + col] -= val * P_values[p];
            }
        }
    }

    // P_smooth = P + temp (密集格式)
    // 需要转换回稀疏格式
    std::vector<std::vector<int>> new_cols(n_fine);
    std::vector<std::vector<Scalar>> new_vals(n_fine);

    for (int i = 0; i < n_fine; ++i) {
        // 添加单位矩阵的贡献 (I * P)
        for (int p = P_row_ptr[i]; p < P_row_ptr[i + 1]; ++p) {
            new_cols[i].push_back(P_col_ind[p]);
            new_vals[i].push_back(P_values[p]);
        }

        // 添加 temp 的贡献
        for (int j = 0; j < n_coarse; ++j) {
            if (temp[i * n_coarse + j] != Scalar(0)) {
                // 检查是否已存在
                bool found = false;
                for (size_t k = 0; k < new_cols[i].size(); ++k) {
                    if (new_cols[i][k] == j) {
                        new_vals[i][k] += temp[i * n_coarse + j];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    new_cols[i].push_back(j);
                    new_vals[i].push_back(temp[i * n_coarse + j]);
                }
            }
        }
    }

    // 转换回CSR格式
    P_row_ptr[0] = 0;
    for (int i = 0; i < n_fine; ++i) {
        P_row_ptr[i + 1] = P_row_ptr[i] + new_cols[i].size();
    }

    P_col_ind.clear();
    P_values.clear();

    for (int i = 0; i < n_fine; ++i) {
        for (size_t j = 0; j < new_cols[i].size(); ++j) {
            P_col_ind.push_back(new_cols[i][j]);
            P_values.push_back(new_vals[i][j]);
        }
    }
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class SmoothedAggregation<Precision::Float32>;
template class SmoothedAggregation<Precision::Float64>;
