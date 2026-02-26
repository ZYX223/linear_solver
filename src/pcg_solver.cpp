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
SolveStats PCGSolver<P>::solve(const Matrix& A, const Vector& b, Vector& x) {
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
// 统一预条件子管理
// ============================================================================

template<Precision P>
void PCGSolver<P>::setup_gpu_preconditioner(const SparseMatrix<P>& A) {
    if (!config_.use_preconditioner || gpu_preconditioner_) {
        return;
    }

    switch (config_.preconditioner_type) {
        case PreconditionerType::ILU0:
            gpu_preconditioner_ = std::make_shared<GPUILUPreconditioner<P>>(sparse_);
            break;
        case PreconditionerType::IC0:
            gpu_preconditioner_ = std::make_shared<GPUIPCPreconditioner<P>>(sparse_);
            break;
        case PreconditionerType::AMG:
            gpu_preconditioner_ = std::make_shared<GPUAMGPreconditioner<P>>(sparse_, config_.amg_config);
            break;
        case PreconditionerType::NONE:
            break;
        default:
            std::cerr << "Warning: Unsupported preconditioner type, falling back to ILU0" << std::endl;
            gpu_preconditioner_ = std::make_shared<GPUILUPreconditioner<P>>(sparse_);
            break;
    }

    if (gpu_preconditioner_) {
        gpu_preconditioner_->setup(A);
    }
}

template<Precision P>
void PCGSolver<P>::setup_cpu_preconditioner(const SparseMatrix<P>& A) {
    if (!config_.use_preconditioner || cpu_preconditioner_) {
        return;
    }

    switch (config_.preconditioner_type) {
        case PreconditionerType::ILU0:
            cpu_preconditioner_ = std::make_unique<CPUILUPreconditioner<P>>();
            break;
        case PreconditionerType::IC0:
            cpu_preconditioner_ = std::make_unique<CPUIPCPreconditioner<P>>();
            break;
        case PreconditionerType::AMG:
            cpu_preconditioner_ = std::make_unique<CPUAMGPreconditioner<P>>(config_.amg_config);
            break;
        case PreconditionerType::NONE:
            break;
        default:
            std::cerr << "Warning: Unsupported preconditioner type, falling back to ILU0" << std::endl;
            cpu_preconditioner_ = std::make_unique<CPUILUPreconditioner<P>>();
            break;
    }

    if (cpu_preconditioner_) {
        cpu_preconditioner_->setup(A);
    }
}

// ============================================================================
// GPU 核心算法实现
// ============================================================================

template<Precision P>
SolveStats PCGSolver<P>::solve_core_gpu(
    int n,
    const Matrix& A,
    const Vector& b,
    Vector& x,
    const cusparseSpMatDescr_t matA,
    void* spmv_buffer) {

    Scalar one = ScalarConstants<P>::one();
    Scalar zero = ScalarConstants<P>::zero();

    // 创建向量描述符
    auto vecp = d_p_->create_dnvec_descr();
    auto vecAp = d_Ap_->create_dnvec_descr();

    // 准备初始解和初始残差
    GPUVector<P> d_x(n);
    d_x.upload_from_host(x.data());
    d_r_->upload_from_host(b.data());

    int k = 0;
    const double tol = config_.tolerance;
    const int max_iter = config_.max_iterations;

    auto start = std::chrono::high_resolution_clock::now();

    // 计算初始残差范数平方
    Scalar r1 = blas_->dot(n, d_r_->d_data, d_r_->d_data);

    while (r1 > tol * tol && k < max_iter) {
        // 应用预条件子: z = M^(-1) * r
        if (gpu_preconditioner_) {
            gpu_preconditioner_->apply(*d_r_, *d_z_);
        } else {
            blas_->copy(n, d_r_->d_data, d_z_->d_data);
        }

        k++;

        // 更新搜索方向 p
        if (k == 1) {
            blas_->copy(n, d_z_->d_data, d_p_->d_data);
        } else {
            Scalar numerator = blas_->dot(n, d_r_->d_data, d_z_->d_data);
            Scalar denominator = blas_->dot(n, d_rm2_->d_data, d_zm2_->d_data);
            Scalar beta = numerator / denominator;

            blas_->scal(n, beta, d_p_->d_data);
            blas_->axpy(n, one, d_z_->d_data, d_p_->d_data);
        }

        // 矩阵向量乘: Ap = A * p
        sparse_->spmv(matA, vecp, vecAp, one, zero, spmv_buffer);

        // 计算步长 alpha
        Scalar numerator = blas_->dot(n, d_r_->d_data, d_z_->d_data);
        Scalar denominator = blas_->dot(n, d_p_->d_data, d_Ap_->d_data);
        Scalar alpha = numerator / denominator;

        // 更新解: x = x + alpha * p
        blas_->axpy(n, alpha, d_p_->d_data, d_x.d_data);

        // 保存旧的 r 和 z 用于下次迭代
        blas_->copy(n, d_r_->d_data, d_rm2_->d_data);
        blas_->copy(n, d_z_->d_data, d_zm2_->d_data);

        // 更新残差: r = r - alpha * Ap
        Scalar nalpha = -alpha;
        blas_->axpy(n, nalpha, d_Ap_->d_data, d_r_->d_data);

        // 计算新的残差范数
        r1 = blas_->dot(n, d_r_->d_data, d_r_->d_data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    d_x.download_to_host(x.data());

    // 清理向量描述符
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));

    double final_residual = std::sqrt(r1);
    return create_stats(k, final_residual, r1 <= tol * tol, solve_time);
}

// ============================================================================
// CPU 核心算法实现
// ============================================================================

template<Precision P>
SolveStats PCGSolver<P>::solve_core_cpu(
    int n,
    const Matrix& A,
    const Vector& b,
    Vector& x) {

    // 分配工作向量
    std::vector<Scalar> r(n);
    std::vector<Scalar> z(n);
    std::vector<Scalar> p(n);
    std::vector<Scalar> Ap(n);
    std::vector<Scalar> rm2(n);
    std::vector<Scalar> zm2(n);

    int k = 0;
    const double tol = config_.tolerance;
    const int max_iter = config_.max_iterations;

    r = b;  // 初始残差 r = b
    Scalar r1 = CPUOps::dot<P>(r, r);

    auto start = std::chrono::high_resolution_clock::now();

    while (r1 > tol * tol && k < max_iter) {
        // 应用预条件子: z = M^(-1) * r
        if (cpu_preconditioner_) {
            cpu_preconditioner_->apply(r, z);
        } else {
            z = r;
        }

        k++;

        // 更新搜索方向 p
        if (k == 1) {
            p = z;
        } else {
            Scalar numerator = CPUOps::dot<P>(r, z);
            Scalar denominator = CPUOps::dot<P>(rm2, zm2);
            Scalar beta = numerator / denominator;

            CPUOps::scal<P>(beta, p);
            CPUOps::axpy<P>(ScalarConstants<P>::one(), z, p);
        }

        // 矩阵向量乘: Ap = A * p
        CPUOps::spmv<P>(n, A.row_ptr, A.col_ind, A.values, p, Ap);

        // 计算步长 alpha
        Scalar numerator = CPUOps::dot<P>(r, z);
        Scalar denominator = CPUOps::dot<P>(p, Ap);
        Scalar alpha = numerator / denominator;

        // 更新解: x = x + alpha * p
        CPUOps::axpy<P>(alpha, p, x);

        // 保存旧的 r 和 z 用于下次迭代
        rm2 = r;
        zm2 = z;

        // 更新残差: r = r - alpha * Ap
        CPUOps::axpy<P>(-alpha, Ap, r);

        // 计算新的残差范数
        r1 = CPUOps::dot<P>(r, r);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    double final_residual = std::sqrt(r1);
    return create_stats(k, final_residual, r1 <= tol * tol, solve_time);
}

// ============================================================================
// GPU 求解入口
// ============================================================================

template<Precision P>
SolveStats PCGSolver<P>::solve_gpu(const Matrix& A, const Vector& b, Vector& x) {
    int n = A.rows;
    allocate_workspace(n);

    // 创建矩阵描述符
    auto matA = A.create_sparse_descr();

    // 准备 SPMV buffer
    Scalar one = ScalarConstants<P>::one();
    Scalar zero = ScalarConstants<P>::zero();

    auto vecp = d_p_->create_dnvec_descr();
    auto vecAp = d_Ap_->create_dnvec_descr();

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(sparse_->handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matA, vecp, &zero, vecAp,
                                           CudaDataType<P>::value, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buffer_spMV_size_));

    void* buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer, buffer_spMV_size_));
    d_buffer_spMV_.reset(buffer);

    // 清理临时描述符
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecp));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));

    // 统一预条件子管理
    setup_gpu_preconditioner(A);

    // 调用核心算法
    SolveStats stats = solve_core_gpu(n, A, b, x, matA, d_buffer_spMV_.get());

    // 清理矩阵描述符
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    return stats;
}

// ============================================================================
// CPU 求解入口
// ============================================================================

template<Precision P>
SolveStats PCGSolver<P>::solve_cpu(const Matrix& A, const Vector& b, Vector& x) {
    int n = A.rows;

    // 统一预条件子管理
    setup_cpu_preconditioner(A);

    // 调用核心算法
    return solve_core_cpu(n, A, b, x);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class PCGSolver<Precision::Float32>;
template class PCGSolver<Precision::Float64>;
