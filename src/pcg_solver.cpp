#include "pcg_solver.h"
#include "preconditioner.h"
#include <algorithm>
#include <iostream>
#include <chrono>

// ============================================================================
// PCGSolver 实现
// ============================================================================

PCGSolver::PCGSolver(const PCGConfig& config)
    : config_(config),
      d_buffer_spMV_(nullptr), buffer_spMV_size_(0) {

    backend_ = config.backend;

    if (backend_ == BACKEND_GPU) {
        blas_ = std::make_unique<CUBLASWrapper>();
        sparse_ = std::make_shared<CUSparseWrapper>();
    }
}

PCGSolver::~PCGSolver() {
    free_workspace();
}

void PCGSolver::set_preconditioner(std::shared_ptr<PreconditionerBase<GPUVector>> prec) {
    if (backend_ == BACKEND_GPU) {
        preconditioner_ = prec;
    }
}

void PCGSolver::allocate_workspace(int n) {
    if (backend_ == BACKEND_GPU) {
        d_r_ = std::make_unique<GPUVector>(n);
        d_z_ = std::make_unique<GPUVector>(n);
        d_rm2_ = std::make_unique<GPUVector>(n);
        d_zm2_ = std::make_unique<GPUVector>(n);
        d_p_ = std::make_unique<GPUVector>(n);
        d_Ap_ = std::make_unique<GPUVector>(n);
    }
    // CPU 不需要预分配
}

void PCGSolver::free_workspace() {
    if (backend_ == BACKEND_GPU) {
        // unique_ptr 会自动调用 GPUVector 的析构函数释放 GPU 内存
        d_r_.reset();
        d_z_.reset();
        d_rm2_.reset();
        d_zm2_.reset();
        d_p_.reset();
        d_Ap_.reset();
        d_buffer_spMV_.reset();  // 自动调用 cudaFree
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
// 辅助函数：创建统计信息
// ============================================================================
inline SolveStats create_stats(int k, float r1, bool converged, float time) {
    SolveStats stats;
    stats.iterations = k;
    stats.final_residual = sqrt(r1);
    stats.converged = converged;
    stats.solve_time = time;
    return stats;
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
    void* buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer, buffer_spMV_size_));
    d_buffer_spMV_.reset(buffer);

    // 延迟初始化预处理器（如果需要）
    if (config_.use_preconditioner && !preconditioner_) {
        preconditioner_ = std::make_shared<GPUILUPreconditioner>(sparse_);
    }

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

    auto start = std::chrono::high_resolution_clock::now();

    while (r1 > tol * tol && k < max_iter) {
        if (preconditioner_) {
            preconditioner_->apply(*d_r_, *d_z_);
        } else {
            // 无预处理：z = r
            blas_->copy(n, d_r_->d_data, d_z_->d_data);
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

        sparse_->spmv(matA, vecp, vecAp, floatone, floatzero, d_buffer_spMV_.get());

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

    auto end = std::chrono::high_resolution_clock::now();
    float solve_time = std::chrono::duration<float>(end - start).count();

    d_x.download_to_host(x.data());

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));

    return create_stats(k, r1, k <= max_iter, solve_time);
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
        cpu_ilu_prec_ = std::make_unique<CPUILUPreconditioner>();
        cpu_ilu_prec_->setup(A.rows, A.row_ptr, A.col_ind, A.values);
    }

    r = b;
    float r1 = CPUOps::dot(r, r);

    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    float solve_time = std::chrono::duration<float>(end - start).count();

    return create_stats(k, r1, k <= max_iter, solve_time);
}
