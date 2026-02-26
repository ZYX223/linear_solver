#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <type_traits>

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/cg.hpp>
#include <cusparse_v2.h>

#include "../include/mm_reader.h"
#include "../../include/pcg_solver.h"
#include "../../include/amg_solver.h"
#include "../../include/sparse_utils.h"

using namespace std;

// ============================================================================
// 辅助函数
// ============================================================================

template <typename T>
double norm2(const std::vector<T>& v) {
    double sum = 0.0;
    for (const auto& val : v) {
        sum += static_cast<double>(val) * static_cast<double>(val);
    }
    return std::sqrt(sum);
}

template <typename T>
double CPUOps::dot(const std::vector<T>& a, const std::vector<T>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

template <typename Scalar>
void spmv_csr(const std::vector<int>& row_ptr,
               const std::vector<int>& col_ind,
               const std::vector<Scalar>& values,
               const std::vector<Scalar>& x,
               std::vector<Scalar>& y) {
    int n = row_ptr.size() - 1;
    // 先清零输出向量
    for (int i = 0; i < n; ++i) {
        y[i] = Scalar(0);
    }
    // 然后累加（与 amgcl 的实现方式保持一致）
    for (int i = 0; i < n; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            y[i] += values[j] * x[col_ind[j]];
        }
    }
}

// ============================================================================
// 结果结构体
// ============================================================================

template <Precision P>
struct PreconditionerResult {
    // PCG+ILU(0)
    int pcg_ilu_cpu_iters;
    double pcg_ilu_cpu_residual;
    double pcg_ilu_cpu_rel_residual;
    double pcg_ilu_cpu_time;
    bool pcg_ilu_cpu_converged;

    int pcg_ilu_gpu_iters;
    double pcg_ilu_gpu_residual;
    double pcg_ilu_gpu_rel_residual;
    double pcg_ilu_gpu_time;
    bool pcg_ilu_gpu_converged;

    // PCG+IC(0) (不完全 Cholesky)
    int pcg_ic0_cpu_iters;
    double pcg_ic0_cpu_residual;
    double pcg_ic0_cpu_rel_residual;
    double pcg_ic0_cpu_time;
    bool pcg_ic0_cpu_converged;

    int pcg_ic0_gpu_iters;
    double pcg_ic0_gpu_residual;
    double pcg_ic0_gpu_rel_residual;
    double pcg_ic0_gpu_time;
    bool pcg_ic0_gpu_converged;

    // PCG+AMG (CPU)
    int pcg_amg_cpu_iters;
    double pcg_amg_cpu_residual;
    double pcg_amg_cpu_rel_residual;
    double pcg_amg_cpu_time;
    bool pcg_amg_cpu_converged;

    // PCG+AMG (GPU)
    int pcg_amg_gpu_iters;
    double pcg_amg_gpu_residual;
    double pcg_amg_gpu_rel_residual;
    double pcg_amg_gpu_time;
    bool pcg_amg_gpu_converged;

    // 独立 AMG 求解器 (CPU)
    int amg_cpu_iters;
    double amg_cpu_residual;
    double amg_cpu_rel_residual;
    double amg_cpu_time;
    bool amg_cpu_converged;

    // PCG+AMG (amgcl CUDA backend)
    int pcg_amg_amgcl_gpu_iters;
    double pcg_amg_amgcl_gpu_residual;
    double pcg_amg_amgcl_gpu_rel_residual;
    double pcg_amg_amgcl_gpu_time;
    bool pcg_amg_amgcl_gpu_converged;

    // PCG+AMG (amgcl make_solver - 标准用法)
    int pcg_amg_make_solver_gpu_iters;
    double pcg_amg_make_solver_gpu_residual;
    double pcg_amg_make_solver_gpu_rel_residual;
    double pcg_amg_make_solver_gpu_time;
    bool pcg_amg_make_solver_gpu_converged;
};

// ============================================================================
// PCG 预条件子测试（支持 ILU(0)）
// ============================================================================

template <Precision P>
PreconditionerResult<P> test_pcg_preconditioner(const MMMatrix& mm,
                                                      const std::vector<double>& b_double,
                                                      bool use_gpu,
                                                      PreconditionerType prec_type) {
    using Scalar = ScalarT<P>;
    using SparseMatrixType = SparseMatrix<P>;

    // 转换矩阵
    std::vector<int> row_ptr, col_ind;
    std::vector<double> values_double;
    convertToCSR(mm, row_ptr, col_ind, values_double);

    SparseMatrixType A(mm.rows, mm.cols, mm.nnz);
    A.row_ptr = row_ptr;
    A.col_ind = col_ind;
    A.values.assign(values_double.begin(), values_double.end());

    if (use_gpu) {
        A.upload_to_gpu();
    }

    // 准备 RHS 和初始值
    std::vector<Scalar> b(mm.rows);
    b.assign(b_double.begin(), b_double.end());
    std::vector<Scalar> x(mm.rows, Scalar(0.0));

    double norm_b = CPUOps::dot(b, b);

    // PCG 配置
    PCGConfig config;
    config.max_iterations = 1000;
    config.tolerance = 1e-8;
    config.use_preconditioner = true;
    config.backend = use_gpu ? BACKEND_GPU : BACKEND_CPU;
    config.precision = P;
    config.preconditioner_type = prec_type;  // 设置预条件子类型

    // 配置 AMG 参数（如果使用 AMG 预条件子）
    if (prec_type == PreconditionerType::AMG ) {
        config.amg_config = std::make_shared<AMGConfig>();
        // GPU 使用较小的 damping factor
        if (use_gpu) {
            config.amg_config->damping_factor = 0.5;
        }
    }

    PCGSolver<P> solver(config);

    auto start = chrono::high_resolution_clock::now();
    SolveStats stats = solver.solve(A, b, x);
    auto end = chrono::high_resolution_clock::now();

    // 计算相对残差
    double rel_residual = [&] {
        std::vector<Scalar> Ax(mm.rows);
        spmv_csr<Scalar>(row_ptr, col_ind, A.values, x, Ax);
        for (int i = 0; i < mm.rows; ++i) {
            Ax[i] -= b[i];
        }
        return norm2(Ax) / norm_b;
    }();

    double time = chrono::duration<double>(end - start).count();

    // 返回结果
    PreconditionerResult<P> result{};

    if (prec_type == PreconditionerType::ILU0) {
        if (use_gpu) {
            result.pcg_ilu_gpu_iters = stats.iterations;
            result.pcg_ilu_gpu_time = time;
            result.pcg_ilu_gpu_rel_residual = rel_residual;
            result.pcg_ilu_gpu_residual = stats.final_residual;
            result.pcg_ilu_gpu_converged = stats.converged;
        } else {
            result.pcg_ilu_cpu_iters = stats.iterations;
            result.pcg_ilu_cpu_time = time;
            result.pcg_ilu_cpu_rel_residual = rel_residual;
            result.pcg_ilu_cpu_residual = stats.final_residual;
            result.pcg_ilu_cpu_converged = stats.converged;
        }
    } else if (prec_type == PreconditionerType::IC0) {
        if (use_gpu) {
            result.pcg_ic0_gpu_iters = stats.iterations;
            result.pcg_ic0_gpu_time = time;
            result.pcg_ic0_gpu_rel_residual = rel_residual;
            result.pcg_ic0_gpu_residual = stats.final_residual;
            result.pcg_ic0_gpu_converged = stats.converged;
        } else {
            result.pcg_ic0_cpu_iters = stats.iterations;
            result.pcg_ic0_cpu_time = time;
            result.pcg_ic0_cpu_rel_residual = rel_residual;
            result.pcg_ic0_cpu_residual = stats.final_residual;
            result.pcg_ic0_cpu_converged = stats.converged;
        }
    } else if (prec_type == PreconditionerType::AMG) {
        // PCG + AMG: 自动根据 use_gpu 选择 CPU 或 GPU 实现
        if (use_gpu) {
            result.pcg_amg_gpu_iters = stats.iterations;
            result.pcg_amg_gpu_time = time;
            result.pcg_amg_gpu_rel_residual = rel_residual;
            result.pcg_amg_gpu_residual = stats.final_residual;
            result.pcg_amg_gpu_converged = stats.converged;
        } else {
            // CPU AMG 结果存储在 pcg_amg_cpu_* 字段
            result.pcg_amg_cpu_iters = stats.iterations;
            result.pcg_amg_cpu_time = time;
            result.pcg_amg_cpu_rel_residual = rel_residual;
            result.pcg_amg_cpu_residual = stats.final_residual;
            result.pcg_amg_cpu_converged = stats.converged;
        }
    }

    return result;
}

// ============================================================================
// amgcl 库测试
// ============================================================================

template <typename Scalar>
struct AMGCLResult {
    int iters;
    double residual;
    double rel_residual;
    double time;
    bool converged;
};

template <typename Scalar>
AMGCLResult<Scalar> test_amgcl(const std::vector<int>& row_ptr_int,
                                const std::vector<int>& col_ind_int,
                                const std::vector<double>& values_double,
                                const std::vector<double>& b_double,
                                int n) {
    using Backend = amgcl::backend::builtin<Scalar>;

    // 转换为 amgcl 格式（使用 ptrdiff_t 类型）
    typedef typename Backend::matrix Matrix;
    typedef typename Backend::vector Vector;

    std::vector<Scalar> val(values_double.begin(), values_double.end());
    // Vector rhs(b_double.begin(), b_double.end());
    // Vector x(n, Scalar(0));
    std::vector<Scalar> rhs(b_double.begin(), b_double.end());
    std::vector<Scalar> x(n, Scalar(0));

    // 转换索引类型为 ptrdiff_t
    std::vector<ptrdiff_t> row_ptr(row_ptr_int.begin(), row_ptr_int.end());
    std::vector<ptrdiff_t> col_ind(col_ind_int.begin(), col_ind_int.end());

    auto A = std::make_shared<Matrix>(n, n, row_ptr, col_ind, val);

    // 配置求解器
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::gauss_seidel
        >,
        amgcl::solver::cg<Backend>
    > Solver;

    typename Solver::params prm;
    // 配置参数，与 test_amg 保持一致
    prm.solver.maxiter = 1000;  // 与 test_amg 的 max_iterations 一致
    // Float 精度使用更宽松的容差，避免精度问题
    if constexpr (std::is_same_v<Scalar, float>) {
        prm.solver.tol = 1e-6f;
    } else {
        prm.solver.tol = 1e-8;
    }
    // prm.precond.coarsening.aggr.eps_strong = 0.08f;
    // prm.precond.coarsening.aggr.block_size = 1;

    // 调试：打印 float 下的 tolerance 值
    if constexpr (std::is_same_v<Scalar, float>) {
        std::cout << "  [DEBUG] Float 精度，使用 tol = " << prm.solver.tol << " (更宽松)\n";
    }

    auto start = chrono::high_resolution_clock::now();
    Solver solve(*A, prm);
    auto setup_end = chrono::high_resolution_clock::now();

    size_t iters;
    Scalar error;
    std::tie(iters, error) = solve(*A, rhs, x);
    auto end = chrono::high_resolution_clock::now();

    AMGCLResult<Scalar> result{};
    result.iters = static_cast<int>(iters);
    result.residual = static_cast<double>(error);
    result.time = chrono::duration<double>(end - start).count();

    // 调试：打印 float 下的收敛判断
    if constexpr (std::is_same_v<Scalar, float>) {
        bool conv_check = (error < prm.solver.tol);
        std::cout << "  [DEBUG] error = " << error << ", tol = " << prm.solver.tol << ", converged = " << conv_check << "\n";
    }

    result.converged = (error < prm.solver.tol);  // 使用实际残差判断收敛

    // 计算相对残差
    std::vector<Scalar> ax(n, Scalar(0));  // 显式初始化为 0（与备份版本一致）
    amgcl::backend::spmv(1, *A, x, 0, ax);
    double rel_res = 0;
    double norm_rhs = 0;
    for (int i = 0; i < n; ++i) {
        Scalar r = rhs[i] - ax[i];
        rel_res += static_cast<double>(r * r);
        norm_rhs += static_cast<double>(rhs[i] * rhs[i]);
    }
    result.rel_residual = std::sqrt(rel_res) / std::sqrt(norm_rhs);

    return result;
}

// ============================================================================
// PCG+AMG 测试（使用 amgcl CUDA backend）
// ============================================================================

template <typename Scalar>
AMGCLResult<Scalar> test_amgcl_gpu(const std::vector<int>& row_ptr_int,
                                   const std::vector<int>& col_ind_int,
                                   const std::vector<double>& values_double,
                                   const std::vector<double>& b_double,
                                   int n) {
    using Backend = amgcl::backend::cuda<Scalar>;

    // 转换为 amgcl 格式（使用 ptrdiff_t 类型）
    typedef typename Backend::matrix Matrix;
    typedef typename Backend::vector Vector;

    std::vector<Scalar> val(values_double.begin(), values_double.end());
    std::vector<Scalar> rhs(b_double.begin(), b_double.end());
    std::vector<Scalar> x(n, Scalar(0));

    // 转换索引类型为 ptrdiff_t（与备份版本保持一致）
    std::vector<ptrdiff_t> row_ptr(row_ptr_int.begin(), row_ptr_int.end());
    std::vector<ptrdiff_t> col_ind(col_ind_int.begin(), col_ind_int.end());

    // 创建 CUDA 矩阵
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    auto A = std::make_shared<Matrix>(n, n, row_ptr, col_ind, val, handle);

    // 配置求解器
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::damped_jacobi
        >,
        amgcl::solver::cg<Backend>
    > Solver;

    typename Solver::params prm;
    prm.solver.maxiter = 1000;
    if constexpr (std::is_same_v<Scalar, float>) {
        prm.solver.tol = 1e-6f;
    } else {
        prm.solver.tol = 1e-8;
    }

    auto start = chrono::high_resolution_clock::now();
    Solver solve(*A, prm);
    auto setup_end = chrono::high_resolution_clock::now();

    size_t iters;
    Scalar error;
    std::tie(iters, error) = solve(*A, rhs, x);
    auto end = chrono::high_resolution_clock::now();

    cusparseDestroy(handle);

    AMGCLResult<Scalar> result{};
    result.iters = static_cast<int>(iters);
    result.residual = static_cast<double>(error);
    result.time = chrono::duration<double>(end - start).count();
    result.converged = (error < prm.solver.tol);

    // 计算相对残差（在 host 上）
    std::vector<Scalar> ax(n, Scalar(0));
    amgcl::backend::spmv(1, *A, x, Scalar(0), ax);
    double norm_r = 0.0;
    double norm_b = 0.0;
    for (int i = 0; i < n; ++i) {
        Scalar r = rhs[i] - ax[i];
        norm_r += static_cast<double>(r * r);
        norm_b += static_cast<double>(rhs[i] * rhs[i]);
    }
    result.rel_residual = std::sqrt(norm_r) / std::sqrt(norm_b);

    return result;
}

// ============================================================================
// AMG 测试
// ============================================================================

template <typename Scalar>
struct AMGResult {
    int iters;
    double residual;
    double rel_residual;
    double time;
    bool converged;
};

template <Precision P>
AMGResult<ScalarT<P>> test_amg(const std::vector<int>& row_ptr,
                                              const std::vector<int>& col_ind,
                                              const std::vector<double>& values_double,
                                              const std::vector<double>& b_double,
                                              int n) {
    using Scalar = ScalarT<P>;
    using SparseMatrixType = SparseMatrix<P>;

    // 构建稀疏矩阵
    SparseMatrixType A(n, n, values_double.size());
    A.row_ptr = row_ptr;
    A.col_ind = col_ind;
    A.values.assign(values_double.begin(), values_double.end());

    // 准备 RHS 和初始值
    std::vector<Scalar> b(n);
    b.assign(b_double.begin(), b_double.end());
    std::vector<Scalar> x(n, Scalar(0.0));

    double norm_b = 0.0;
    for (const auto& v : b) norm_b += v * v;
    norm_b = std::sqrt(norm_b);

    // 配置 AMG 求解器
    AMGConfig config;
    config.max_iterations = 1000;
    // Float 精度使用更宽松的容差，避免精度问题
    if constexpr (std::is_same_v<Scalar, float>) {
        config.tolerance = 1e-6;
    } else {
        config.tolerance = 1e-8;
    }
    config.precision = P;

    AMGSolver<P> solver(config);

    auto start = chrono::high_resolution_clock::now();
    SolveStats stats = solver.solve(A, b, x);
    auto end = chrono::high_resolution_clock::now();

    // 计算相对残差
    double rel_residual = [&] {
        std::vector<Scalar> Ax(n);
        spmv_csr<Scalar>(row_ptr, col_ind, A.values, x, Ax);
        for (int i = 0; i < n; ++i) {
            Ax[i] -= b[i];
        }
        return norm2(Ax) / norm_b;
    }();

    AMGResult<Scalar> result{};
    result.iters = stats.iterations;
    result.residual = stats.final_residual;
    result.rel_residual = rel_residual;
    result.time = chrono::duration<double>(end - start).count();
    result.converged = stats.converged;

    return result;
}

// ============================================================================
// 主测试函数
// ============================================================================

template <Precision P>
void run_combined_test(const std::string& matrix_file, bool use_gpu) {
    using Scalar = ScalarT<P>;
    std::string precision_str = (P == Precision::Float32) ? "float" : "double";

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " 测试: " << precision_str << " 精度\n";
    std::cout << "========================================\n";

    // 读取矩阵
    std::cout << "[1/3] 读取矩阵...\n";
    MMMatrix mm = readMatrixMarket(matrix_file);
    std::cout << "读取矩阵: " << mm.rows << " x " << mm.cols << ", nnz = " << mm.nnz;
    if (mm.symmetric) std::cout << " [对称]" << std::endl;

    // 转换为 CSR
    std::vector<int> row_ptr, col_ind;
    std::vector<double> values_double;
    convertToCSR(mm, row_ptr, col_ind, values_double);

    // 准备 RHS
    std::cout << "[2/3] 准备 RHS...\n";
    std::vector<double> b_double;

    if (!tryReadRHS(matrix_file, b_double)) {
        std::cout << "  未找到 RHS 文件，使用 x_exact = [1,...,1] 生成\n";
        std::vector<Scalar> x_exact(mm.rows, Scalar(1.0));
        b_double.resize(mm.rows);

        for (int i = 0; i < mm.rows; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                b_double[i] += values_double[j] * Scalar(1.0);
            }
        }
    } else {
        std::cout << "  找到 RHS 文件: " << matrix_file << "_b.mtx\n";
        std::cout << "  读取稀疏向量: " << b_double.size() << " 个元素\n";
    }

    double norm_b = 0.0;
    for (const auto& v : b_double) norm_b += v * v;
    norm_b = std::sqrt(norm_b);
    std::cout << "  ||b|| = " << norm_b << std::endl;

    // ========================================================================
    // 测试 PCG 预条件子
    // ========================================================================
    std::cout << "[3/5] 测试 PCG 预条件子...\n";

    // 测试 ILU(0) (CPU)
    std::cout << "  [3.1/5] PCG+ILU(0) (CPU)...\n";
    auto ilu0_cpu_result = test_pcg_preconditioner<P>(mm, b_double, false, PreconditionerType::ILU0);
    std::cout << "  完成\n";

    // 测试 IC(0) (CPU)
    std::cout << "  [3.2/5] PCG+IC(0) (CPU)...\n";
    auto ic0_cpu_result = test_pcg_preconditioner<P>(mm, b_double, false, PreconditionerType::IC0);
    std::cout << "  完成\n";

    // 测试 AMG (CPU)
    std::cout << "  [3.3/5] PCG+AMG (CPU)...\n";
    auto amg_cpu_result = test_pcg_preconditioner<P>(mm, b_double, false, PreconditionerType::AMG);
    std::cout << "  完成\n";

    // 测试 ILU(0) (GPU)
    PreconditionerResult<P> ilu0_gpu_result{};
    if (use_gpu) {
        std::cout << "  [3.4/5] PCG+ILU(0) (GPU)...\n";
        ilu0_gpu_result = test_pcg_preconditioner<P>(mm, b_double, true, PreconditionerType::ILU0);
        std::cout << "  完成\n";
    }

    // 测试 IC(0) (GPU)
    PreconditionerResult<P> ic0_gpu_result{};
    if (use_gpu) {
        std::cout << "  [3.5/5] PCG+IC(0) (GPU)...\n";
        ic0_gpu_result = test_pcg_preconditioner<P>(mm, b_double, true, PreconditionerType::IC0);
        std::cout << "  完成\n";
    }

    // 测试 AMG (GPU)
    PreconditionerResult<P> amg_gpu_result{};
    if (use_gpu) {
        std::cout << "  [3.6/5] PCG+AMG (GPU)...\n";
        amg_gpu_result = test_pcg_preconditioner<P>(mm, b_double, true, PreconditionerType::AMG);
        std::cout << "  完成\n";
    }

    // ========================================================================
    // 测试独立 AMG 求解器
    // ========================================================================
    std::cout << "\n[4.1/5] 测试 AMG (CPU)...\n";
    auto amg_result = test_amg<P>(row_ptr, col_ind, values_double, b_double, mm.rows);
    std::cout << "  完成\n";

    // 测试 amgcl 库
    std::cout << "\n[4.2/5] 测试 amgcl (CPU)...\n";
    auto amgcl_result = test_amgcl<Scalar>(row_ptr, col_ind, values_double, b_double, mm.rows);
    std::cout << "  完成\n";

    // 测试 amgcl 库 (GPU backend - 标准 PCG+AMG 用法)
    // 暂时禁用，amgcl CUDA backend 存在兼容性问题
    AMGCLResult<Scalar> amgcl_gpu_result{};
    amgcl_gpu_result.converged = false;
    amgcl_gpu_result.iters = 0;
    amgcl_gpu_result.rel_residual = 0.0;
    amgcl_gpu_result.time = 0.0;
    // TODO: 启用 amgcl GPU 测试需要修复 CUDA backend 兼容性问题
    // if (use_gpu) {
    //     std::cout << "\n[4.3/5] 测试 amgcl (GPU - 标准 PCG+AMG)...\n";
    //     amgcl_gpu_result = test_amgcl_gpu<Scalar>(row_ptr, col_ind, values_double, b_double, mm.rows);
    //     std::cout << "  完成\n";
    // }

    // ========================================================================
    // 打印结果
    // ========================================================================
    {
        std::cout << "\n";
        std::cout << "  " << std::string(88, '=') << "\n";
        std::cout << "  " << std::left << std::setw(22) << "Method"
                  << std::right << std::setw(8) << "Iters"
                  << std::setw(14) << "RelResidual"
                  << std::setw(12) << "Time (s)"
                  << std::setw(10) << "Converged"
                  << "\n";
        std::cout << "  " << std::string(88, '-') << "\n";

        // PCG + ILU(0) (CPU)
        std::cout << "  " << std::left << std::setw(22) << "PCG+ILU(0) (CPU)"
                  << std::right << std::setw(8) << ilu0_cpu_result.pcg_ilu_cpu_iters
                  << std::scientific << std::setprecision(2) << std::setw(14) << ilu0_cpu_result.pcg_ilu_cpu_rel_residual
                  << std::fixed << std::setprecision(3) << std::setw(12) << ilu0_cpu_result.pcg_ilu_cpu_time
                  << std::setw(10) << (ilu0_cpu_result.pcg_ilu_cpu_converged ? "Yes" : "No")
                  << "\n";

        // PCG + ILU(0) (GPU)
        if (use_gpu) {
            std::cout << "  " << std::left << std::setw(22) << "PCG+ILU(0) (GPU)"
                      << std::right << std::setw(8) << ilu0_gpu_result.pcg_ilu_gpu_iters
                      << std::scientific << std::setprecision(2) << std::setw(14) << ilu0_gpu_result.pcg_ilu_gpu_rel_residual
                      << std::fixed << std::setprecision(3) << std::setw(12) << ilu0_gpu_result.pcg_ilu_gpu_time
                      << std::setw(10) << (ilu0_gpu_result.pcg_ilu_gpu_converged ? "Yes" : "No")
                      << "\n";
        }

        // PCG + IC(0) (CPU)
        std::cout << "  " << std::left << std::setw(22) << "PCG+IC(0) (CPU)"
                  << std::right << std::setw(8) << ic0_cpu_result.pcg_ic0_cpu_iters
                  << std::scientific << std::setprecision(2) << std::setw(14) << ic0_cpu_result.pcg_ic0_cpu_rel_residual
                  << std::fixed << std::setprecision(3) << std::setw(12) << ic0_cpu_result.pcg_ic0_cpu_time
                  << std::setw(10) << (ic0_cpu_result.pcg_ic0_cpu_converged ? "Yes" : "No")
                  << "\n";

        // PCG + IC(0) (GPU)
        if (use_gpu) {
            std::cout << "  " << std::left << std::setw(22) << "PCG+IC(0) (GPU)"
                      << std::right << std::setw(8) << ic0_gpu_result.pcg_ic0_gpu_iters
                      << std::scientific << std::setprecision(2) << std::setw(14) << ic0_gpu_result.pcg_ic0_gpu_rel_residual
                      << std::fixed << std::setprecision(3) << std::setw(12) << ic0_gpu_result.pcg_ic0_gpu_time
                      << std::setw(10) << (ic0_gpu_result.pcg_ic0_gpu_converged ? "Yes" : "No")
                      << "\n";
        }

        // PCG + AMG (CPU)
        std::cout << "  " << std::left << std::setw(22) << "PCG+AMG (CPU)"
                  << std::right << std::setw(8) << amg_cpu_result.pcg_amg_cpu_iters
                  << std::scientific << std::setprecision(2) << std::setw(14) << amg_cpu_result.pcg_amg_cpu_rel_residual
                  << std::fixed << std::setprecision(3) << std::setw(12) << amg_cpu_result.pcg_amg_cpu_time
                  << std::setw(10) << (amg_cpu_result.pcg_amg_cpu_converged ? "Yes" : "No")
                  << "\n";

        // PCG + AMG (GPU)
        if (use_gpu) {
            std::cout << "  " << std::left << std::setw(22) << "PCG+AMG (GPU)"
                      << std::right << std::setw(8) << amg_gpu_result.pcg_amg_gpu_iters
                      << std::scientific << std::setprecision(2) << std::setw(14) << amg_gpu_result.pcg_amg_gpu_rel_residual
                      << std::fixed << std::setprecision(3) << std::setw(12) << amg_gpu_result.pcg_amg_gpu_time
                      << std::setw(10) << (amg_gpu_result.pcg_amg_gpu_converged ? "Yes" : "No")
                      << "\n";
        }

        // amgcl (GPU) - 暂时禁用，amgcl CUDA backend 存在兼容性问题
        // if (use_gpu && amgcl_gpu_result.iters > 0) {
        //     std::cout << "  " << std::left << std::setw(22) << "amgcl (GPU)"
        //               << std::right << std::setw(8) << amgcl_gpu_result.iters
        //               << std::scientific << std::setprecision(2) << std::setw(14) << amgcl_gpu_result.rel_residual
        //               << std::fixed << std::setprecision(3) << std::setw(12) << amgcl_gpu_result.time
        //               << std::setw(10) << (amgcl_gpu_result.converged ? "Yes" : "No")
        //               << "\n";
        // }

        // 测试 amgcl make_solver (GPU) - 标准 PCG+AMG 用法
        // 暂时禁用，因为 amgcl CUDA backend 接口版本不兼容问题
        /*
        AMGCLResult<Scalar> amgcl_make_solver_result{};
        if (use_gpu) {
            std::cout << "\n[4.4/5] 测试 amgcl make_solver (GPU)...\n";
            amgcl_make_solver_result = test_amgcl_make_solver<Scalar>(row_ptr, col_ind, values_double, b_double, mm.rows);
            std::cout << "  完成\n";
        }

        result.pcg_amg_make_solver_gpu_iters = amgcl_make_solver_result.iters;
        result.pcg_amg_make_solver_gpu_residual = amgcl_make_solver_result.residual;
        result.pcg_amg_make_solver_gpu_rel_residual = amgcl_make_solver_result.rel_residual;
        result.pcg_amg_make_solver_gpu_time = amgcl_make_solver_result.time;
        result.pcg_amg_make_solver_gpu_converged = amgcl_make_solver_result.converged;
        */

        // 独立 AMG 求解器 (CPU)
        std::cout << "  " << std::left << std::setw(22) << "AMG (CPU)"
                  << std::right << std::setw(8) << amg_result.iters
                  << std::scientific << std::setprecision(2) << std::setw(14) << amg_result.rel_residual
                  << std::fixed << std::setprecision(3) << std::setw(12) << amg_result.time
                  << std::setw(10) << (amg_result.converged ? "Yes" : "No")
                  << "\n";

        // amgcl 库求解器 (CPU)
        std::cout << "  " << std::left << std::setw(22) << "amgcl (CPU)"
                  << std::right << std::setw(8) << amgcl_result.iters
                  << std::scientific << std::setprecision(2) << std::setw(14) << amgcl_result.rel_residual
                  << std::fixed << std::setprecision(3) << std::setw(12) << amgcl_result.time
                  << std::setw(10) << (amgcl_result.converged ? "Yes" : "No")
                  << "\n";

        std::cout << "  " << std::string(88, '=') << "\n";
    }

    // ========================================================================
    // 对比分析
    // ========================================================================
    {
        std::cout << "\n  求解器说明:\n";
        std::cout << "  - PCG+ILU(0): 预条件共轭梯度法，使用 ILU(0) 预条件子\n";
        std::cout << "  - PCG+AMG (CPU): 预条件共轭梯度法，使用 AMG 预条件子 (CPU)\n";
        std::cout << "  - PCG+AMG (GPU): 预条件共轭梯度法，使用 AMG 预条件子 (GPU) \n";
        std::cout << "  - AMG (CPU): 独立代数多重网格求解器 \n";
        std::cout << "  - amgcl (CPU): amgcl 库的独立求解器 \n";
        std::cout << "  - 对称矩阵: " << (mm.symmetric ? "是" : "否") << "\n";

        // 性能对比
    //     if (use_gpu) {
    //         std::cout << "\n  性能对比:\n";
    //         if (ilu0_gpu_result.pcg_ilu_gpu_converged && amg_gpu_result.pcg_amg_gpu_converged) {
    //             double speedup = ilu0_gpu_result.pcg_ilu_gpu_time / amg_gpu_result.pcg_amg_gpu_time;
    //             std::cout << "  - PCG+AMG (GPU) 相比 PCG+ILU(0) (GPU) 加速比: "
    //                       << std::fixed << std::setprecision(2) << speedup << "x\n";
    //         }
    //         if (amg_cpu_result.pcg_amg_cpu_converged && amg_gpu_result.pcg_amg_gpu_converged) {
    //             double speedup = amg_cpu_result.pcg_amg_cpu_time / amg_gpu_result.pcg_amg_gpu_time;
    //             std::cout << "  - PCG+AMG (GPU) 相比 PCG+AMG (CPU) 加速比: "
    //                       << std::fixed << std::setprecision(2) << speedup << "x\n";
    //         }

    //         // CPU vs GPU AMG 收敛性对比
    //         std::cout << "\n  AMG 预条件子对比:\n";
    //         if (amg_cpu_result.pcg_amg_cpu_converged) {
    //             std::cout << "  - PCG+AMG (CPU): " << amg_cpu_result.pcg_amg_cpu_iters << " 次迭代, 收敛\n";
    //         } else {
    //             std::cout << "  - PCG+AMG (CPU): 未收敛\n";
    //         }

    //         if (amg_gpu_result.pcg_amg_gpu_converged) {
    //             std::cout << "  - PCG+AMG (GPU): " << amg_gpu_result.pcg_amg_gpu_iters << " 次迭代, 收敛\n";
    //         } else {
    //             std::cout << "  - PCG+AMG (GPU): 未收敛（达到最大迭代次数 " << amg_gpu_result.pcg_amg_gpu_iters << "）\n";
    //         }

    //         // 对比 CPU 和 GPU 的差异
    //         if (amg_cpu_result.pcg_amg_cpu_converged && !amg_gpu_result.pcg_amg_gpu_converged) {
    //             std::cout << "\n  ⚠ 警告: CPU 版本收敛但 GPU 版本未收敛！\n";
    //             std::cout << "  可能的原因:\n";
    //             std::cout << "    1. GPU AMG 预条件子实现存在数值精度问题\n";
    //             std::cout << "    2. GPU 上的 smoother 实现（如 damped_jacobi）参数不合适\n";
    //             std::cout << "    3. GPU 向量运算的累积误差较大\n";
    //             std::cout << "  建议: 检查 GPU AMG 预条件子的实现，特别是 smoother 部分和向量运算\n";
    //         } else if (amg_cpu_result.pcg_amg_cpu_converged && amg_gpu_result.pcg_amg_gpu_converged) {
    //             if (amg_cpu_result.pcg_amg_cpu_iters != amg_gpu_result.pcg_amg_gpu_iters) {
    //                 std::cout << "\n  ⚠ 注意: CPU 和 GPU 迭代次数不同\n";
    //                 std::cout << "    差异: " << (amg_gpu_result.pcg_amg_gpu_iters > amg_cpu_result.pcg_amg_cpu_iters ? "+" : "")
    //                           << (amg_gpu_result.pcg_amg_gpu_iters - amg_cpu_result.pcg_amg_cpu_iters) << " 次迭代\n";
    //             }
    //         }
    //     }

    //     // 独立 AMG 求解器对比
    //     std::cout << "\n  求解器对比:\n";
    //     if (amg_cpu_result.pcg_amg_cpu_converged && amg_result.converged) {
    //         double speedup = amg_cpu_result.pcg_amg_cpu_time / amg_result.time;
    //         std::cout << "  - 独立 AMG (CPU) 相比 PCG+AMG (CPU) 加速比: "
    //                   << std::fixed << std::setprecision(2) << speedup << "x\n";
    //     }
    //     if (amg_result.converged && amgcl_result.converged) {
    //         double speedup = amg_result.time / amgcl_result.time;
    //         std::cout << "  - amgcl (CPU) 相比 独立 AMG (CPU) 加速比: "
    //                   << std::fixed << std::setprecision(2) << speedup << "x\n";
    //         std::cout << "    注: amgcl 是经过高度优化的库，性能通常更好\n";
    //     }
    // }

    // // AMG 验证：对比我们自己的 AMG 和 amgcl
    // {
    //     std::cout << "\n  AMG 验证: ";
    //     if (!amg_result.converged || !amgcl_result.converged) {
    //         std::cout << "✗ 至少一个求解器未收敛\n";
    //     } else if (amg_result.iters == amgcl_result.iters) {
    //         std::cout << "✓ 完全一致\n";
    //         std::cout << "    迭代次数: " << amg_result.iters << " (linear_solver) vs " << amgcl_result.iters << " (amgcl)\n";
    //         std::cout << "    相对残差: " << std::scientific << std::setprecision(3) << amg_result.rel_residual
    //                   << " (linear_solver) vs " << amgcl_result.rel_residual << " (amgcl)\n";
    //     } else {
    //         std::cout << "⚠ 迭代次数不同\n";
    //         std::cout << "    迭代次数: " << amg_result.iters << " (linear_solver) vs " << amgcl_result.iters << " (amgcl)\n";
    //         std::cout << "    相对残差: " << std::scientific << std::setprecision(3) << amg_result.rel_residual
    //                   << " (linear_solver) vs " << amgcl_result.rel_residual << " (amgcl)\n";
    //     }
    // }
    }  // 关闭对比分析作用域
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <matrix_file.mtx> [选项]\n";
        std::cout << "\n选项:\n";
        std::cout << "  --float       使用单精度 (float32)\n";
        std::cout << "  --double      使用双精度 (float64, 默认)\n";
        std::cout << "  --both        测试两种精度\n";
        std::cout << "  --cpu-only    仅使用 CPU 后端\n";
        std::cout << "\n说明:\n";
        std::cout << "  测试不同的求解器配置:\n";
        std::cout << "  - PCG+ILU(0) (CPU): 预条件共轭梯度法，使用 ILU(0) 预条件子\n";
        std::cout << "  - PCG+ILU(0) (GPU): 预条件共轭梯度法，使用 ILU(0) 预条件子\n";
        std::cout << "  - PCG+AMG (CPU): 预条件共轭梯度法，使用 AMG 预条件子 (CPU)\n";
        std::cout << "  - PCG+AMG (GPU): 预条件共轭梯度法，使用 AMG 预条件子 (GPU)\n";
        std::cout << "  - AMG (CPU): 独立代数多重网格求解器 \n";
        std::cout << "  - amgcl (CPU): amgcl 库的独立求解器 \n";
        std::cout << "\n  注意: 默认情况下同时测试 CPU 和 GPU 后端\n";
        std::cout << "\n示例:\n";
        std::cout << "  " << argv[0] << " matrices/nos1.mtx              # 默认: GPU+double\n";
        std::cout << "  " << argv[0] << " matrices/bcsstk14.mtx --both   # GPU + float+double\n";
        std::cout << "  " << argv[0] << " matrices/bcsstk14.mtx --cpu-only --both  # CPU + float+double\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    bool use_gpu = true;
    std::string precision_str = "double";  // 默认双精度

    // 检测 GPU
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "\n警告: 未检测到 CUDA 设备，将仅运行 CPU 测试\n";
        use_gpu = false;
    }

    // 解析参数
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu-only") {
            use_gpu = false;
        } else if (arg == "--float") {
            precision_str = "float";
        } else if (arg == "--double") {
            precision_str = "double";
        } else if (arg == "--both") {
            precision_str = "both";
        }
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " 线性求解器综合测试\n";
    std::cout << "========================================\n";
    std::cout << "矩阵文件: " << matrix_file << "\n";
    std::cout << "后端: " << (use_gpu ? "CPU + GPU" : "CPU only") << "\n";
    std::cout << "精度: " << precision_str << "\n";
    std::cout << "========================================\n";

    try {
        if (precision_str == "float") {
            run_combined_test<Precision::Float32>(matrix_file, use_gpu);
        } else if (precision_str == "double") {
            run_combined_test<Precision::Float64>(matrix_file, use_gpu);
        } else {
            // --both: 测试两种精度
            run_combined_test<Precision::Float64>(matrix_file, use_gpu);
            run_combined_test<Precision::Float32>(matrix_file, use_gpu);
        }

        std::cout << "\n========================================\n";
        std::cout << " 测试完成！\n";
        std::cout << "========================================\n";

    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// 模板实例化声明
// TODO: 修复 amgcl CUDA backend 兼容性问题后启用
// template AMGCLResult<float> test_amgcl_gpu<float>(const std::vector<int>&, const std::vector<int>&, const std::vector<double>&, const std::vector<double>&, int);
// template AMGCLResult<double> test_amgcl_gpu<double>(const std::vector<int>&, const std::vector<int>&, const std::vector<double>&, const std::vector<double>&, int);
