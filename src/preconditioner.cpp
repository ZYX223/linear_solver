#include "preconditioner.h"
#include <stdio.h>

// 注意: GPUILUPreconditioner 的实现已移至 preconditioner_cuda.cu

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
                                     const std::vector<typename CPUILUPreconditioner<P>::Scalar>& values) {
    n_ = n;
    row_ptr_ = row_ptr;
    col_ind_ = col_ind;
    values_ = values;

    for (int i = 0; i < n_; i++) {
        for (int k_idx = row_ptr_[i]; k_idx < row_ptr_[i + 1]; k_idx++) {
            int k = col_ind_[k_idx];
            if (k >= i) break;

            typename CPUILUPreconditioner<P>::Scalar a_ik = values_[k_idx];

            typename CPUILUPreconditioner<P>::Scalar a_kk = ScalarConstants<P>::zero();
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

                    typename CPUILUPreconditioner<P>::Scalar a_kj = ScalarConstants<P>::zero();
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
void CPUILUPreconditioner<P>::apply(const std::vector<typename CPUILUPreconditioner<P>::Scalar>& r,
                                     std::vector<typename CPUILUPreconditioner<P>::Scalar>& z) const {
    std::vector<typename CPUILUPreconditioner<P>::Scalar> y(n_);
    forward_substitute(r, y);
    backward_substitute(y, z);
}

template<Precision P>
void CPUILUPreconditioner<P>::forward_substitute(const std::vector<typename CPUILUPreconditioner<P>::Scalar>& b,
                                                  std::vector<typename CPUILUPreconditioner<P>::Scalar>& y) const {
    for (int i = 0; i < n_; i++) {
        typename CPUILUPreconditioner<P>::Scalar sum = b[i];
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
void CPUILUPreconditioner<P>::backward_substitute(const std::vector<typename CPUILUPreconditioner<P>::Scalar>& y,
                                                   std::vector<typename CPUILUPreconditioner<P>::Scalar>& x) const {
    for (int i = n_ - 1; i >= 0; i--) {
        typename CPUILUPreconditioner<P>::Scalar sum = y[i];
        typename CPUILUPreconditioner<P>::Scalar diag = ScalarConstants<P>::one();

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
// 实现：A ≈ R^T * R，其中 R 是上三角矩阵
// 参考：NVIDIA cuSPARSE White Paper
// ============================================================================

template<Precision P>
CPUIPCPreconditioner<P>::CPUIPCPreconditioner() : n_(0) {}

template<Precision P>
void CPUIPCPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    n_ = A.rows;

    using Scalar = typename CPUIPCPreconditioner<P>::Scalar;

    // ========================================================================
    // Step 1: 提取上三角部分
    // ========================================================================
    row_ptr_.resize(n_ + 1, 0);

    // 统计每行上三角元素数量（包括对角线）
    for (int i = 0; i < n_; i++) {
        int count = 0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] >= i) count++;  // 上三角：col >= row
        }
        row_ptr_[i + 1] = row_ptr_[i] + count;
    }

    int nnz = row_ptr_[n_];
    col_ind_.resize(nnz);
    values_.resize(nnz);

    // 填充上三角部分
    for (int i = 0; i < n_; i++) {
        int idx = row_ptr_[i];
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] >= i) {
                col_ind_[idx] = A.col_ind[j];
                values_[idx] = A.values[j];
                idx++;
            }
        }
    }

    // ========================================================================
    // Step 2: IC(0) 分解: A ≈ R^T * R
    // R 是上三角，按行计算
    // 对于 col >= row 的元素 R[row,col]:
    //   R[row,col] = (A[row,col] - sum_{k<row} R[k,row] * R[k,col]) / R[row,row]  (row < col)
    //   R[row,row] = sqrt(A[row,row] - sum_{k<row} R[k,row]^2)                    (row == col)
    // ========================================================================
    for (int row = 0; row < n_; row++) {
        // 找到 R[row,row] 在 values_ 中的索引
        int diag_idx = -1;
        for (int j = row_ptr_[row]; j < row_ptr_[row + 1]; j++) {
            if (col_ind_[j] == row) {
                diag_idx = j;
                break;
            }
        }

        // 先计算对角线 R[row,row]
        Scalar sum_sq = ScalarConstants<P>::zero();
        for (int k = 0; k < row; k++) {
            // 找 R[k,row]
            Scalar r_k_row = ScalarConstants<P>::zero();
            for (int j = row_ptr_[k]; j < row_ptr_[k + 1]; j++) {
                if (col_ind_[j] == row) {
                    r_k_row = values_[j];
                    break;
                }
            }
            sum_sq += r_k_row * r_k_row;
        }

        Scalar diag_val = values_[diag_idx] - sum_sq;
        if (diag_val > ScalarConstants<P>::zero()) {
            values_[diag_idx] = std::sqrt(diag_val);
        } else {
            // 数值保护
            values_[diag_idx] = std::sqrt(std::fabs(diag_val) + 1e-12);
        }

        Scalar r_row_row = values_[diag_idx];

        // 计算非对角线元素 R[row,col] (col > row)
        for (int j_idx = diag_idx + 1; j_idx < row_ptr_[row + 1]; j_idx++) {
            int col = col_ind_[j_idx];

            // sum_{k<row} R[k,row] * R[k,col]
            Scalar sum_prod = ScalarConstants<P>::zero();
            for (int k = 0; k < row; k++) {
                // 找 R[k,row]
                Scalar r_k_row = ScalarConstants<P>::zero();
                for (int j = row_ptr_[k]; j < row_ptr_[k + 1]; j++) {
                    if (col_ind_[j] == row) {
                        r_k_row = values_[j];
                        break;
                    }
                }

                // 找 R[k,col]
                Scalar r_k_col = ScalarConstants<P>::zero();
                for (int j = row_ptr_[k]; j < row_ptr_[k + 1]; j++) {
                    if (col_ind_[j] == col) {
                        r_k_col = values_[j];
                        break;
                    }
                }

                sum_prod += r_k_row * r_k_col;
            }

            if (r_row_row != ScalarConstants<P>::zero()) {
                values_[j_idx] = (values_[j_idx] - sum_prod) / r_row_row;
            }
        }
    }
}

template<Precision P>
void CPUIPCPreconditioner<P>::apply(const std::vector<typename CPUIPCPreconditioner<P>::Scalar>& r,
                                     std::vector<typename CPUIPCPreconditioner<P>::Scalar>& z) const {
    // M z = r, M = R^T * R
    // 先解 R^T * t = r（前向代入，R^T 是下三角）
    // 再解 R * z = t（后向代入，R 是上三角）
    std::vector<typename CPUIPCPreconditioner<P>::Scalar> t(n_);
    forward_substitute(r, t);   // R^T * t = r
    backward_substitute(t, z);  // R * z = t
}

template<Precision P>
void CPUIPCPreconditioner<P>::forward_substitute(const std::vector<typename CPUIPCPreconditioner<P>::Scalar>& b,
                                                  std::vector<typename CPUIPCPreconditioner<P>::Scalar>& t) const {
    // R^T * t = b
    // R^T 是下三角，第 row 行非零元素对应 R[k,row] (k <= row)
    // t[row] = (b[row] - sum_{k<row} R[k,row] * t[k]) / R[row,row]
    using Scalar = typename CPUIPCPreconditioner<P>::Scalar;

    for (int row = 0; row < n_; row++) {
        Scalar sum = ScalarConstants<P>::zero();
        Scalar r_row_row = ScalarConstants<P>::one();

        // 遍历 R 的第 row 列，找 R[k,row] (k <= row)
        // 由于 R 是 CSR 按行存储，需要在所有 k <= row 的行中找 col_ind == row
        for (int k = 0; k < row; k++) {
            // 在第 k 行找 col == row 的元素
            for (int j = row_ptr_[k]; j < row_ptr_[k + 1]; j++) {
                if (col_ind_[j] == row) {
                    sum += values_[j] * t[k];
                    break;
                }
            }
        }

        // 找对角线 R[row,row]
        for (int j = row_ptr_[row]; j < row_ptr_[row + 1]; j++) {
            if (col_ind_[j] == row) {
                r_row_row = values_[j];
                break;
            }
        }

        t[row] = (b[row] - sum) / r_row_row;
    }
}

template<Precision P>
void CPUIPCPreconditioner<P>::backward_substitute(const std::vector<typename CPUIPCPreconditioner<P>::Scalar>& t,
                                                   std::vector<typename CPUIPCPreconditioner<P>::Scalar>& x) const {
    // R * x = t
    // R 是上三角，第 row 行非零元素对应 R[row,col] (col >= row)
    // x[row] = (t[row] - sum_{col>row} R[row,col] * x[col]) / R[row,row]
    using Scalar = typename CPUIPCPreconditioner<P>::Scalar;

    for (int row = n_ - 1; row >= 0; row--) {
        Scalar sum = t[row];
        Scalar r_row_row = ScalarConstants<P>::one();

        // 遍历第 row 行，找 col > row 的元素
        bool found_diag = false;
        for (int j = row_ptr_[row]; j < row_ptr_[row + 1]; j++) {
            int col = col_ind_[j];
            if (col == row) {
                r_row_row = values_[j];
                found_diag = true;
            } else if (col > row) {
                sum -= values_[j] * x[col];
            }
        }

        x[row] = sum / r_row_row;
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
void CPUAMGPreconditioner<P>::apply(const std::vector<PRECISION_SCALAR(P)>& r,
                                      std::vector<PRECISION_SCALAR(P)>& z) const {
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
// 该文件由 NVCC 编译，解决了 Thrust/cuSPARSE 兼容性问题
