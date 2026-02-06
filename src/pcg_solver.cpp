#include "pcg_solver.h"
#include <algorithm>
#include <iostream>

// ============================================================================
// CPUOps 实现
// ============================================================================

namespace CPUOps {
    float dot(const std::vector<float>& x, const std::vector<float>& y) {
        float result = 0.0f;
        for (size_t i = 0; i < x.size(); i++) {
            result += x[i] * y[i];
        }
        return result;
    }

    void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y) {
        for (size_t i = 0; i < x.size(); i++) {
            y[i] += alpha * x[i];
        }
    }

    void scal(float alpha, std::vector<float>& x) {
        for (size_t i = 0; i < x.size(); i++) {
            x[i] *= alpha;
        }
    }

    void copy(const std::vector<float>& x, std::vector<float>& y) {
        y = x;
    }

    void spmv(int n, const std::vector<int>& row_ptr,
              const std::vector<int>& col_ind,
              const std::vector<float>& values,
              const std::vector<float>& x,
              std::vector<float>& y) {
        std::fill(y.begin(), y.end(), 0.0f);
        for (int i = 0; i < n; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                y[i] += values[j] * x[col_ind[j]];
            }
        }
    }
}

// ============================================================================
// CPUIILUPreconditioner 实现
// ============================================================================

CPUIILUPreconditioner::CPUIILUPreconditioner() : n_(0) {}

CPUIILUPreconditioner::~CPUIILUPreconditioner() {}

void CPUIILUPreconditioner::setup(int n, const std::vector<int>& row_ptr,
                                  const std::vector<int>& col_ind,
                                  const std::vector<float>& values) {
    n_ = n;
    row_ptr_ = row_ptr;
    col_ind_ = col_ind;
    values_ = values;

    // ILU(0) 分解（基于 IKJ 版本的 Gaussian 消去法）
    for (int i = 0; i < n_; i++) {
        for (int k_idx = row_ptr_[i]; k_idx < row_ptr_[i + 1]; k_idx++) {
            int k = col_ind_[k_idx];

            if (k >= i) break;

            float a_ik = values_[k_idx];

            float a_kk = 0.0f;
            for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                if (col_ind_[p] == k) {
                    a_kk = values_[p];
                    break;
                }
            }

            if (a_kk != 0.0f) {
                a_ik /= a_kk;
                values_[k_idx] = a_ik;

                for (int j_idx = k_idx + 1; j_idx < row_ptr_[i + 1]; j_idx++) {
                    int j = col_ind_[j_idx];
                    if (j <= k) continue;

                    float a_kj = 0.0f;
                    for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                        if (col_ind_[p] == j) {
                            a_kj = values_[p];
                            break;
                        }
                    }

                    if (a_kj != 0.0f) {
                        values_[j_idx] -= a_ik * a_kj;
                    }
                }
            }
        }
    }
}

void CPUIILUPreconditioner::apply(const std::vector<float>& r, std::vector<float>& z) const {
    std::vector<float> y(n_);
    forward_substitute(r, y);
    backward_substitute(y, z);
}

void CPUIILUPreconditioner::forward_substitute(const std::vector<float>& b, std::vector<float>& y) const {
    for (int i = 0; i < n_; i++) {
        float sum = b[i];
        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            int col = col_ind_[j];
            if (col < i) {
                sum -= values_[j] * y[col];
            }
        }
        y[i] = sum;
    }
}

void CPUIILUPreconditioner::backward_substitute(const std::vector<float>& y, std::vector<float>& x) const {
    for (int i = n_ - 1; i >= 0; i--) {
        float sum = y[i];
        float diag = 1.0f;

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
// PCGSolver 实现
// ============================================================================

PCGSolver::PCGSolver(const PCGConfig& config)
    : config_(config),
      d_r_(nullptr), d_z_(nullptr), d_rm2_(nullptr), d_zm2_(nullptr),
      d_p_(nullptr), d_Ap_(nullptr),
      d_buffer_spMV_(nullptr), buffer_spMV_size_(0) {

    backend_ = config.backend;

    if (backend_ == BACKEND_GPU) {
        blas_ = std::make_shared<CUBLASWrapper>();
        sparse_ = std::make_shared<CUSparseWrapper>();

        if (!config_.use_preconditioner) {
            preconditioner_ = std::make_shared<NonePreconditioner>();
        }
    }
}

PCGSolver::~PCGSolver() {
    free_workspace();
}

void PCGSolver::set_preconditioner(std::shared_ptr<Preconditioner> prec) {
    if (backend_ == BACKEND_GPU) {
        preconditioner_ = prec;
    }
}

void PCGSolver::allocate_workspace(int n) {
    if (backend_ == BACKEND_GPU) {
        d_r_ = new GPUVector(n);
        d_z_ = new GPUVector(n);
        d_rm2_ = new GPUVector(n);
        d_zm2_ = new GPUVector(n);
        d_p_ = new GPUVector(n);
        d_Ap_ = new GPUVector(n);
    }
    // CPU 不需要预分配
}

void PCGSolver::free_workspace() {
    if (backend_ == BACKEND_GPU) {
        if (d_r_) { delete d_r_; d_r_ = nullptr; }
        if (d_z_) { delete d_z_; d_z_ = nullptr; }
        if (d_rm2_) { delete d_rm2_; d_rm2_ = nullptr; }
        if (d_zm2_) { delete d_zm2_; d_zm2_ = nullptr; }
        if (d_p_) { delete d_p_; d_p_ = nullptr; }
        if (d_Ap_) { delete d_Ap_; d_Ap_ = nullptr; }
        if (d_buffer_spMV_) { cudaFree(d_buffer_spMV_); d_buffer_spMV_ = nullptr; }
    }
}

SolveStats PCGSolver::solve(const SparseMatrix& A,
                            const std::vector<float>& b,
                            std::vector<float>& x) {
    if (backend_ == BACKEND_GPU) {
        return solve_gpu(A, b, x);
    } else {
        return solve_cpu(A, b, x);
    }
}

// ============================================================================
// GPU 求解实现
// ============================================================================

SolveStats PCGSolver::solve_gpu(const SparseMatrix& A,
                               const std::vector<float>& b,
                               std::vector<float>& x) {
    int n = A.rows;
    allocate_workspace(n);

    auto vecp = d_p_->create_dnvec_descr();
    auto vecAp = d_Ap_->create_dnvec_descr();
    auto vecR = d_r_->create_dnvec_descr();
    auto vecZ = d_z_->create_dnvec_descr();
    auto matA = A.create_sparse_descr();

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(sparse_->handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &floatone, matA, vecp, &floatzero, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buffer_spMV_size_));
    CHECK_CUDA(cudaMalloc(&d_buffer_spMV_, buffer_spMV_size_));

    if (preconditioner_) {
        preconditioner_->setup(A);
    }

    GPUVector d_x(n);
    d_x.upload_from_host(x.data());
    d_r_->upload_from_host(b.data());

    float r1 = blas_->dot(n, d_r_->d_data, d_r_->d_data);

    int k = 0;
    const float tol = config_.tolerance;
    const int max_iter = config_.max_iterations;

    while (r1 > tol * tol && k < max_iter) {
        if (preconditioner_) {
            preconditioner_->apply(*d_r_, *d_z_);
        }

        k++;

        if (k == 1) {
            blas_->copy(n, d_z_->d_data, d_p_->d_data);
        } else {
            float numerator = blas_->dot(n, d_r_->d_data, d_z_->d_data);
            float denominator = blas_->dot(n, d_rm2_->d_data, d_zm2_->d_data);
            float beta = numerator / denominator;

            blas_->scal(n, beta, d_p_->d_data);
            blas_->axpy(n, floatone, d_z_->d_data, d_p_->d_data);
        }

        sparse_->spmv(matA, vecp, vecAp, floatone, floatzero, d_buffer_spMV_);

        float numerator = blas_->dot(n, d_r_->d_data, d_z_->d_data);
        float denominator = blas_->dot(n, d_p_->d_data, d_Ap_->d_data);
        float alpha = numerator / denominator;

        blas_->axpy(n, alpha, d_p_->d_data, d_x.d_data);

        blas_->copy(n, d_r_->d_data, d_rm2_->d_data);
        blas_->copy(n, d_z_->d_data, d_zm2_->d_data);

        float nalpha = -alpha;
        blas_->axpy(n, nalpha, d_Ap_->d_data, d_r_->d_data);

        r1 = blas_->dot(n, d_r_->d_data, d_r_->d_data);
    }

    d_x.download_to_host(x.data());

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));

    SolveStats stats;
    stats.iterations = k;
    stats.final_residual = sqrt(r1);
    stats.converged = (k <= max_iter);

    return stats;
}

// ============================================================================
// CPU 求解实现
// ============================================================================

SolveStats PCGSolver::solve_cpu(const SparseMatrix& A,
                               const std::vector<float>& b,
                               std::vector<float>& x) {
    int n = A.rows;
    int k = 0;
    const float tol = config_.tolerance;
    const int max_iter = config_.max_iterations;

    std::vector<float> r(n);
    std::vector<float> z(n);
    std::vector<float> p(n);
    std::vector<float> Ap(n);
    std::vector<float> rm2(n);
    std::vector<float> zm2(n);

    if (config_.use_preconditioner) {
        cpu_ilu_prec_ = std::make_shared<CPUIILUPreconditioner>();
        cpu_ilu_prec_->setup(n, A.row_ptr, A.col_ind, A.values);
    }

    r = b;
    float r1 = CPUOps::dot(r, r);

    while (r1 > tol * tol && k < max_iter) {
        if (config_.use_preconditioner && cpu_ilu_prec_) {
            cpu_ilu_prec_->apply(r, z);
        } else {
            z = r;
        }

        k++;

        if (k == 1) {
            p = z;
        } else {
            float numerator = CPUOps::dot(r, z);
            float denominator = CPUOps::dot(rm2, zm2);
            float beta = numerator / denominator;

            CPUOps::scal(beta, p);
            CPUOps::axpy(1.0f, z, p);
        }

        CPUOps::spmv(n, A.row_ptr, A.col_ind, A.values, p, Ap);

        float numerator = CPUOps::dot(r, z);
        float denominator = CPUOps::dot(p, Ap);
        float alpha = numerator / denominator;

        CPUOps::axpy(alpha, p, x);

        rm2 = r;
        zm2 = z;

        CPUOps::axpy(-alpha, Ap, r);

        r1 = CPUOps::dot(r, r);
    }

    SolveStats stats;
    stats.iterations = k;
    stats.final_residual = sqrt(r1);
    stats.converged = (k <= max_iter);

    return stats;
}
