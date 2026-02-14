#include "pcg_solver.h"
#include "preconditioner.h"
#include <algorithm>
#include <iostream>
#include <chrono>

// ============================================================================
// PCGSolver 模板实现
// ============================================================================

template<Precision P>
PCGSolver<P>::PCGSolver(const PCGConfig& config)
    : config_(config),
      gpu_preconditioner_(),  // GPU 预条件子（显式初始化为 nullptr）
      cpu_preconditioner_(nullptr),  // CPU 预条件子（显式初始化为 nullptr）
      d_buffer_spMV_(nullptr), buffer_spMV_size_(0) {

    backend_ = config.backend;

    if (backend_ == BACKEND_GPU) {
        blas_ = std::make_unique<CUBLASWrapper<P>>();
        sparse_ = std::make_shared<CUSparseWrapper<P>>();
    }
}

template<Precision P>
PCGSolver<P>::~PCGSolver() {
    free_workspace();
}

template<Precision P>
void PCGSolver<P>::allocate_workspace(int n) {
    if (backend_ == BACKEND_GPU) {
        d_r_ = std::make_unique<GPUVector<P>>(n);
        d_z_ = std::make_unique<GPUVector<P>>(n);
        d_rm2_ = std::make_unique<GPUVector<P>>(n);
        d_zm2_ = std::make_unique<GPUVector<P>>(n);
        d_p_ = std::make_unique<GPUVector<P>>(n);
        d_Ap_ = std::make_unique<GPUVector<P>>(n);
    }
}

template<Precision P>
void PCGSolver<P>::free_workspace() {
    if (backend_ == BACKEND_GPU) {
        d_r_.reset();
        d_z_.reset();
        d_rm2_.reset();
        d_zm2_.reset();
        d_p_.reset();
        d_Ap_.reset();
        d_buffer_spMV_.reset();
    }
}

template<Precision P>
SolveStats PCGSolver<P>::solve(const SparseMatrix<P>& A,
                                const std::vector<typename PCGSolver<P>::Scalar>& b,
                                std::vector<typename PCGSolver<P>::Scalar>& x) {
    if (backend_ == BACKEND_GPU) {
        return solve_gpu(A, b, x);
    } else {
        return solve_cpu(A, b, x);
    }
}

// ============================================================================
// 辅助函数：创建统计信息
// ============================================================================
inline SolveStats create_stats(int k, double r1, bool converged, double time) {
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

template<Precision P>
SolveStats PCGSolver<P>::solve_gpu(const SparseMatrix<P>& A,
                                   const std::vector<typename PCGSolver<P>::Scalar>& b,
                                   std::vector<typename PCGSolver<P>::Scalar>& x) {
    int n = A.rows;
    allocate_workspace(n);

    auto vecp = d_p_->create_dnvec_descr();
    auto vecAp = d_Ap_->create_dnvec_descr();
    auto vecR = d_r_->create_dnvec_descr();
    auto vecZ = d_z_->create_dnvec_descr();
    auto matA = A.create_sparse_descr();

    typename PCGSolver<P>::Scalar one = ScalarConstants<P>::one();
    typename PCGSolver<P>::Scalar zero = ScalarConstants<P>::zero();

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(sparse_->handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matA, vecp, &zero, vecAp,
                                           CudaDataType<P>::value, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buffer_spMV_size_));
    void* buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer, buffer_spMV_size_));
    d_buffer_spMV_.reset(buffer);

    if (config_.use_preconditioner && !gpu_preconditioner_) {
        switch (config_.preconditioner_type) {
            case PreconditionerType::ILU0:
                gpu_preconditioner_ = std::make_shared<GPUILUPreconditioner<P>>(sparse_);
                break;
            case PreconditionerType::AMG:
                // AMG 预条件子
                gpu_preconditioner_ = std::make_shared<GPUAMGPreconditioner<P>>(sparse_, config_.amg_config);
                break;
            case PreconditionerType::JACOBI:
            case PreconditionerType::NONE:
                // 对角预条件或无预条件，不需要设置
                break;
            default:
                // 对于未实现的预条件子类型，回退到 ILU0
                std::cerr << "Warning: Unsupported preconditioner type, falling back to ILU0" << std::endl;
                gpu_preconditioner_ = std::make_shared<GPUILUPreconditioner<P>>(sparse_);
                break;
        }
    }

    if (gpu_preconditioner_) {
        gpu_preconditioner_->setup(A);
    }

    GPUVector<P> d_x(n);
    d_x.upload_from_host(x.data());  // 初始解 x = 0
    d_r_->upload_from_host(b.data());  // 初始残差 r = b

    int k = 0;
    const double tol = config_.tolerance;
    const int max_iter = config_.max_iterations;

    auto start = std::chrono::high_resolution_clock::now();

    // 计算初始残差范数平方
    typename PCGSolver<P>::Scalar r1 = blas_->dot(n, d_r_->d_data, d_r_->d_data);
    // printf("[GPU PCG] Initial: ||r|| = %.6e, tolerance = %.1e\n", std::sqrt(r1), tol);

    while (r1 > tol * tol && k < max_iter) {
        if (gpu_preconditioner_) {
            gpu_preconditioner_->apply(*d_r_, *d_z_);
        } else {
            blas_->copy(n, d_r_->d_data, d_z_->d_data);
        }

        k++;

        if (k == 1) {
            blas_->copy(n, d_z_->d_data, d_p_->d_data);
        } else {
            typename PCGSolver<P>::Scalar numerator = blas_->dot(n, d_r_->d_data, d_z_->d_data);
            typename PCGSolver<P>::Scalar denominator = blas_->dot(n, d_rm2_->d_data, d_zm2_->d_data);
            typename PCGSolver<P>::Scalar beta = numerator / denominator;

            blas_->scal(n, beta, d_p_->d_data);
            blas_->axpy(n, one, d_z_->d_data, d_p_->d_data);
        }

        sparse_->spmv(matA, vecp, vecAp, one, zero, d_buffer_spMV_.get());

        typename PCGSolver<P>::Scalar numerator = blas_->dot(n, d_r_->d_data, d_z_->d_data);
        typename PCGSolver<P>::Scalar denominator = blas_->dot(n, d_p_->d_data, d_Ap_->d_data);
        typename PCGSolver<P>::Scalar alpha = numerator / denominator;

        blas_->axpy(n, alpha, d_p_->d_data, d_x.d_data);

        blas_->copy(n, d_r_->d_data, d_rm2_->d_data);
        blas_->copy(n, d_z_->d_data, d_zm2_->d_data);

        typename PCGSolver<P>::Scalar nalpha = -alpha;
        blas_->axpy(n, nalpha, d_Ap_->d_data, d_r_->d_data);

        r1 = blas_->dot(n, d_r_->d_data, d_r_->d_data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    d_x.download_to_host(x.data());

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));

    // 计算最终残差
    double final_residual = std::sqrt(r1);
    return create_stats(k, final_residual, r1 <= tol * tol, solve_time);
}

// ============================================================================
// CPU 求解实现
// ============================================================================

template<Precision P>
SolveStats PCGSolver<P>::solve_cpu(const SparseMatrix<P>& A,
                                   const std::vector<typename PCGSolver<P>::Scalar>& b,
                                   std::vector<typename PCGSolver<P>::Scalar>& x) {
    int n = A.rows;
    int k = 0;
    const double tol = config_.tolerance;
    const int max_iter = config_.max_iterations;

    std::vector<typename PCGSolver<P>::Scalar> r(n);
    std::vector<typename PCGSolver<P>::Scalar> z(n);
    std::vector<typename PCGSolver<P>::Scalar> p(n);
    std::vector<typename PCGSolver<P>::Scalar> Ap(n);
    std::vector<typename PCGSolver<P>::Scalar> rm2(n);
    std::vector<typename PCGSolver<P>::Scalar> zm2(n);

    // 设置预条件子（循环外一次性设置）
    if (config_.use_preconditioner) {
        switch (config_.preconditioner_type) {
            case PreconditionerType::AMG:
                cpu_preconditioner_ = std::make_unique<CPUAMGPreconditioner<P>>(config_.amg_config);
                cpu_preconditioner_->setup(A);
                break;
            case PreconditionerType::ILU0:
                cpu_preconditioner_ = std::make_unique<CPUILUPreconditioner<P>>();
                cpu_preconditioner_->setup(A);
                break;
            case PreconditionerType::JACOBI:
            case PreconditionerType::NONE:
                // Jacobi 预条件或无预条件，不需要设置
                break;
            default:
                break;
        }
    }

    r = b;  // 初始残差 r = b
    typename PCGSolver<P>::Scalar r1 = CPUOps::dot<P>(r, r);

    auto start = std::chrono::high_resolution_clock::now();

    while (r1 > tol * tol && k < max_iter) {
        if (cpu_preconditioner_) {
            cpu_preconditioner_->apply(r, z);
        } else {
            z = r;
        }

        k++;

        if (k == 1) {
            p = z;
        } else {
            typename PCGSolver<P>::Scalar numerator = CPUOps::dot<P>(r, z);
            typename PCGSolver<P>::Scalar denominator = CPUOps::dot<P>(rm2, zm2);
            typename PCGSolver<P>::Scalar beta = numerator / denominator;

            CPUOps::scal<P>(beta, p);
            CPUOps::axpy<P>(ScalarConstants<P>::one(), z, p);
        }

        CPUOps::spmv<P>(n, A.row_ptr, A.col_ind, A.values, p, Ap);

        typename PCGSolver<P>::Scalar numerator = CPUOps::dot<P>(r, z);
        typename PCGSolver<P>::Scalar denominator = CPUOps::dot<P>(p, Ap);
        typename PCGSolver<P>::Scalar alpha = numerator / denominator;

        CPUOps::axpy<P>(alpha, p, x);

        rm2 = r;
        zm2 = z;

        CPUOps::axpy<P>(-alpha, Ap, r);

        // 计算当前残差范数（用于循环条件检查）
        r1 = CPUOps::dot<P>(r, r);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    double final_residual = std::sqrt(r1);
    return create_stats(k, final_residual, r1 <= tol * tol, solve_time);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class PCGSolver<Precision::Float32>;
template class PCGSolver<Precision::Float64>;
