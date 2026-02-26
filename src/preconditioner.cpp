#include "preconditioner.h"
#include <stdio.h>
#include <unordered_map>

// ============================================================================
// CPUILUPreconditioner 模板实现
// ============================================================================

template<Precision P>
CPUILUPreconditioner<P>::CPUILUPreconditioner() : n_(0) {}

template<Precision P>
CPUILUPreconditioner<P>::~CPUILUPreconditioner() {}

template<Precision P>
void CPUILUPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    setup(A.rows, A.row_ptr, A.col_ind, A.values);
}

template<Precision P>
void CPUILUPreconditioner<P>::setup(int n, const std::vector<int>& row_ptr,
                                     const std::vector<int>& col_ind,
                                     const std::vector<Scalar>& values) {
    n_ = n;
    row_ptr_ = row_ptr;
    col_ind_ = col_ind;
    values_ = values;

    for (int i = 0; i < n_; i++) {
        for (int k_idx = row_ptr_[i]; k_idx < row_ptr_[i + 1]; k_idx++) {
            int k = col_ind_[k_idx];
            if (k >= i) break;

            Scalar a_ik = values_[k_idx];

            Scalar a_kk = ScalarConstants<P>::zero();
            for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                if (col_ind_[p] == k) {
                    a_kk = values_[p];
                    break;
                }
            }

            if (a_kk != ScalarConstants<P>::zero()) {
                a_ik /= a_kk;
                values_[k_idx] = a_ik;

                for (int j_idx = k_idx + 1; j_idx < row_ptr_[i + 1]; j_idx++) {
                    int j = col_ind_[j_idx];
                    if (j <= k) continue;

                    Scalar a_kj = ScalarConstants<P>::zero();
                    for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                        if (col_ind_[p] == j) {
                            a_kj = values_[p];
                            break;
                        }
                    }

                    if (a_kj != ScalarConstants<P>::zero()) {
                        values_[j_idx] -= a_ik * a_kj;
                    }
                }
            }
        }
    }
}

template<Precision P>
void CPUILUPreconditioner<P>::apply(const std::vector<Scalar>& r,
                                     std::vector<Scalar>& z) const {
    std::vector<Scalar> y(n_);
    forward_substitute(r, y);
    backward_substitute(y, z);
}

template<Precision P>
void CPUILUPreconditioner<P>::forward_substitute(const std::vector<Scalar>& b,
                                                  std::vector<Scalar>& y) const {
    for (int i = 0; i < n_; i++) {
        Scalar sum = b[i];
        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            int col = col_ind_[j];
            if (col < i) {
                sum -= values_[j] * y[col];
            }
        }
        y[i] = sum;
    }
}

template<Precision P>
void CPUILUPreconditioner<P>::backward_substitute(const std::vector<Scalar>& y,
                                                   std::vector<Scalar>& x) const {
    for (int i = n_ - 1; i >= 0; i--) {
        Scalar sum = y[i];
        Scalar diag = ScalarConstants<P>::one();

        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            int col = col_ind_[j];
            if (col > i) {
                sum -= values_[j] * x[col];
            } else if (col == i) {
                diag = values_[j];
            }
        }

        x[i] = sum / diag;
    }
}

// ============================================================================
// CPUIPCPreconditioner 模板实现（不完全 Cholesky 分解）
// 实现：A ≈ L * L^T，其中 L 是下三角矩阵（直接存储，无需转置）
// ============================================================================

template<Precision P>
CPUIPCPreconditioner<P>::CPUIPCPreconditioner() : n_(0) {}

template<Precision P>
void CPUIPCPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    n_ = A.rows;

    using Scalar = Scalar;

    // ========================================================================
    // Step 1: 提取下三角部分（包括对角线）并构建 L 的 CSR 结构
    // ========================================================================
    l_row_ptr_.resize(n_ + 1, 0);

    // 统计每行下三角元素数量
    for (int i = 0; i < n_; i++) {
        int count = 0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] <= i) count++;  // 下三角：col <= row
        }
        l_row_ptr_[i + 1] = l_row_ptr_[i] + count;
    }

    int l_nnz = l_row_ptr_[n_];
    l_col_ind_.resize(l_nnz);
    l_values_.resize(l_nnz);

    // 填充下三角部分
    for (int i = 0; i < n_; i++) {
        int idx = l_row_ptr_[i];
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] <= i) {
                l_col_ind_[idx] = A.col_ind[j];
                l_values_[idx] = A.values[j];
                idx++;
            }
        }
    }

    // ========================================================================
    // Step 2: 构建列索引映射（用于 IC 分解）
    // ========================================================================
    std::vector<std::unordered_map<int, int>> col_map(n_);
    for (int row = 0; row < n_; row++) {
        for (int j = l_row_ptr_[row]; j < l_row_ptr_[row + 1]; j++) {
            col_map[row][l_col_ind_[j]] = j;
        }
    }

    // ========================================================================
    // Step 3: 构建列到行的反向映射
    // ========================================================================
    std::vector<std::vector<int>> col_rows(n_);
    for (int k = 0; k < n_; k++) {
        for (int j = l_row_ptr_[k]; j < l_row_ptr_[k + 1]; j++) {
            int col = l_col_ind_[j];
            col_rows[col].push_back(k);
        }
    }

    // ========================================================================
    // Step 4: IC(0) 分解: A ≈ L * L^T
    // L[row,col] 对于 col <= row:
    //   L[row,col] = (A[row,col] - sum_{k<col} L[row,k] * L[col,k]) / L[col,col]  (col < row)
    //   L[row,row] = sqrt(A[row,row] - sum_{k<row} L[row,k]^2)                    (col == row)
    // ========================================================================
    diag_.resize(n_);

    for (int row = 0; row < n_; row++) {
        // 找到 L[row,row] 的索引
        auto it_diag = col_map[row].find(row);
        if (it_diag == col_map[row].end()) continue;
        int diag_idx = it_diag->second;

        // 计算非对角线元素 L[row,col] (col < row)
        for (int j_idx = l_row_ptr_[row]; j_idx < diag_idx; j_idx++) {
            int col = l_col_ind_[j_idx];

            Scalar sum_prod = ScalarConstants<P>::zero();
            // sum_{k<col} L[row,k] * L[col,k]
            const auto& rows_row = col_rows[row];
            const auto& rows_col = col_rows[col];

            size_t i1 = 0, i2 = 0;
            while (i1 < rows_row.size() && i2 < rows_col.size()) {
                int k1 = rows_row[i1];
                int k2 = rows_col[i2];
                if (k1 >= col || k2 >= col) break;
                if (k1 == k2) {
                    auto it_row = col_map[row].find(k1);
                    auto it_col = col_map[col].find(k1);
                    if (it_row != col_map[row].end() && it_col != col_map[col].end()) {
                        sum_prod += l_values_[it_row->second] * l_values_[it_col->second];
                    }
                    i1++; i2++;
                } else if (k1 < k2) {
                    i1++;
                } else {
                    i2++;
                }
            }

            // L[col,col]
            auto it_col_diag = col_map[col].find(col);
            if (it_col_diag != col_map[col].end()) {
                Scalar l_col_col = l_values_[it_col_diag->second];
                if (l_col_col != ScalarConstants<P>::zero()) {
                    l_values_[j_idx] = (l_values_[j_idx] - sum_prod) / l_col_col;
                }
            }
        }

        // 计算对角线 L[row,row] = sqrt(A[row,row] - sum_{k<row} L[row,k]^2)
        Scalar sum_sq = ScalarConstants<P>::zero();
        for (int k : col_rows[row]) {
            if (k < row) {
                auto it = col_map[row].find(k);
                if (it != col_map[row].end()) {
                    Scalar l_row_k = l_values_[it->second];
                    sum_sq += l_row_k * l_row_k;
                }
            }
        }

        Scalar diag_val = l_values_[diag_idx] - sum_sq;
        if (diag_val > ScalarConstants<P>::zero()) {
            l_values_[diag_idx] = std::sqrt(diag_val);
        } else {
            l_values_[diag_idx] = std::sqrt(std::fabs(diag_val) + 1e-12);
        }
        diag_[row] = l_values_[diag_idx];
    }

    // ========================================================================
    // Step 5: 构建上三角 R = L^T 的 CSR 结构（用于后向代入）
    // ========================================================================
    // 统计每行上三角元素数量
    row_ptr_.resize(n_ + 1, 0);
    for (int col = 0; col < n_; col++) {
        for (int k : col_rows[col]) {
            if (k > col) {  // L[k,col] 存在且 k > col，对应 R[col,k]
                row_ptr_[col + 1]++;
            }
        }
        row_ptr_[col + 1]++;  // 对角线
    }
    for (int i = 0; i < n_; i++) {
        row_ptr_[i + 1] += row_ptr_[i];
    }

    int r_nnz = row_ptr_[n_];
    col_ind_.resize(r_nnz);
    values_.resize(r_nnz);

    // 填充 R（按行）
    std::vector<int> cur_pos(n_, 0);
    for (int row = 0; row < n_; row++) {
        // 先填对角线
        int diag_pos = row_ptr_[row];
        col_ind_[diag_pos] = row;
        values_[diag_pos] = diag_[row];
        cur_pos[row] = diag_pos + 1;

        // 填非对角线：R[row,col] = L[col,row] for col > row
        for (int k : col_rows[row]) {
            if (k > row) {
                auto it = col_map[k].find(row);
                if (it != col_map[k].end()) {
                    int pos = cur_pos[row]++;
                    col_ind_[pos] = k;
                    values_[pos] = l_values_[it->second];
                }
            }
        }
    }
}

template<Precision P>
void CPUIPCPreconditioner<P>::apply(const std::vector<Scalar>& r,
                                     std::vector<Scalar>& z) const {
    // M z = r, M = L * L^T
    // 先解 L * y = r（前向代入）
    // 再解 L^T * z = y（后向代入）
    std::vector<Scalar> y(n_);
    forward_substitute(r, y);
    backward_substitute(y, z);
}

template<Precision P>
void CPUIPCPreconditioner<P>::forward_substitute(const std::vector<Scalar>& b,
                                                  std::vector<Scalar>& y) const {
    // L * y = b，L 是下三角（CSR 格式）
    // y[row] = (b[row] - sum_{col<row} L[row,col] * y[col]) / L[row,row]
    using Scalar = Scalar;

    for (int row = 0; row < n_; row++) {
        Scalar sum = b[row];

        // 遍历第 row 行的 col < row 元素
        for (int j = l_row_ptr_[row]; j < l_row_ptr_[row + 1] - 1; j++) {  // -1 跳过对角线
            int col = l_col_ind_[j];
            sum -= l_values_[j] * y[col];
        }

        y[row] = sum / diag_[row];
    }
}

template<Precision P>
void CPUIPCPreconditioner<P>::backward_substitute(const std::vector<Scalar>& y,
                                                   std::vector<Scalar>& z) const {
    // L^T * z = y，L^T 是上三角（CSR 格式，存储在 row_ptr_, col_ind_, values_）
    // z[row] = (y[row] - sum_{col>row} L^T[row,col] * z[col]) / L^T[row,row]
    using Scalar = Scalar;

    for (int row = n_ - 1; row >= 0; row--) {
        Scalar sum = y[row];

        // 遍历第 row 行的 col > row 元素
        for (int j = row_ptr_[row] + 1; j < row_ptr_[row + 1]; j++) {  // +1 跳过对角线
            int col = col_ind_[j];
            sum -= values_[j] * z[col];
        }

        z[row] = sum / diag_[row];
    }
}

// ============================================================================
// CPU AMG 预条件子实现
// ============================================================================

template<Precision P>
CPUAMGPreconditioner<P>::CPUAMGPreconditioner(std::shared_ptr<AMGConfig> config)
    : config_(config), is_setup_(false) {
    if (!config_) {
        config_ = std::make_shared<AMGConfig>();
    }
}

template<Precision P>
void CPUAMGPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    // 转换矩阵到 amgcl 格式（参考 amg_solver.cpp）
    size_t n = A.rows;
    size_t nnz = A.nnz;

    ptr_.resize(n + 1);
    col_.resize(nnz);
    val_.resize(nnz);

    for (size_t i = 0; i <= n; ++i) {
        ptr_[i] = static_cast<ptrdiff_t>(A.row_ptr[i]);
    }
    for (size_t i = 0; i < nnz; ++i) {
        col_[i] = static_cast<ptrdiff_t>(A.col_ind[i]);
        val_[i] = static_cast<Scalar>(A.values[i]);
    }

    // 创建 amgcl 后端矩阵
    A_amgcl_ = std::make_shared<typename Backend::matrix>(n, n, ptr_, col_, val_);

    // 配置 AMG 参数
    typename AMGPreconditioner::params prm;
    prm.coarsening.aggr.eps_strong = static_cast<float>(config_->aggregation_eps);
    prm.max_levels = config_->max_levels;
    prm.coarse_enough = config_->coarse_grid_size;
    prm.npre = config_->pre_smooth_steps;
    prm.npost = config_->post_smooth_steps;

    // 创建 AMG 预条件子
    amg_prec_ = std::make_shared<AMGPreconditioner>(*A_amgcl_, prm);

    is_setup_ = true;
}

template<Precision P>
void CPUAMGPreconditioner<P>::apply(const std::vector<ScalarT<P>>& r,
                                      std::vector<ScalarT<P>>& z) const {
    if (!is_setup_) {
        printf("Error: CPUAMGPreconditioner not setup!\n");
        exit(1);
    }

    // 调用 amgcl 预条件子 (AMG V-cycle)
    amg_prec_->apply(r, z);
}

// ============================================================================
// 显式模板实例化（仅 CPU 预条件子）
// 注意：GPU 预条件子的模板实例化在 preconditioner_cuda.cu 中
// ============================================================================

template class CPUILUPreconditioner<Precision::Float32>;
template class CPUILUPreconditioner<Precision::Float64>;

template class CPUIPCPreconditioner<Precision::Float32>;
template class CPUIPCPreconditioner<Precision::Float64>;

template class CPUAMGPreconditioner<Precision::Float32>;
template class CPUAMGPreconditioner<Precision::Float64>;

// 注意：GPU 预条件子（GPUILUPreconditioner, GPUIPCPreconditioner, GPUAMGPreconditioner）
// 的实现和模板实例化已移到 src/preconditioner_cuda.cu
