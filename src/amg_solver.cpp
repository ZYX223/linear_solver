#include "amg_solver.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/solver/cg.hpp>

// ============================================================================
// AMGCL 求解器的实现
// ============================================================================

namespace {

// 模板化的实现类
template<Precision P>
class AMGSolverImplTyped : public AMGSolverImplBase {
public:
    using Scalar = ScalarT<P>;

    explicit AMGSolverImplTyped(const AMGConfig& config)
        : config_(config), setup_done_(false) {}

    void setup(const void* A, Precision precision) override {
        if (precision != P) {
            throw std::runtime_error("AMGSolver: precision mismatch");
        }

        const auto& mat = *static_cast<const SparseMatrix<P>*>(A);

        // 转换矩阵到 amgcl 格式
        size_t n = mat.rows;
        ptr_.resize(n + 1);
        col_.resize(mat.nnz);
        val_.resize(mat.nnz);

        for (size_t i = 0; i <= n; ++i) {
            ptr_[i] = static_cast<ptrdiff_t>(mat.row_ptr[i]);
        }
        for (size_t i = 0; i < mat.nnz; ++i) {
            col_[i] = static_cast<ptrdiff_t>(mat.col_ind[i]);
            val_[i] = static_cast<Scalar>(mat.values[i]);
        }

        // 创建 amgcl 后端矩阵
        A_amgcl_ = std::make_shared<BackendMatrix>(n, n, ptr_, col_, val_);

        // 配置求解器
        typename Solver::params prm;

        // AMG 预条件子参数
        prm.precond.coarsening.aggr.eps_strong = static_cast<float>(config_.aggregation_eps);
        prm.precond.npre = config_.pre_smooth_steps;
        prm.precond.coarsening.aggr.block_size = config_.block_size;
        prm.precond.npost = config_.post_smooth_steps;
        prm.precond.ncycle = config_.ncycle;
        prm.precond.pre_cycles = config_.pre_cycles;
        prm.precond.direct_coarse = config_.direct_coarse;

        // AMG 层级参数
        prm.precond.max_levels = config_.max_levels;
        prm.precond.coarse_enough = config_.coarse_grid_size;

        // CG 求解器参数
        prm.solver.tol = static_cast<Scalar>(config_.tolerance);
        prm.solver.maxiter = config_.max_iterations;

        // 创建求解器
        solver_ = std::make_unique<Solver>(*A_amgcl_, prm);
        setup_done_ = true;
    }

    std::pair<int, double> solve(const void* b, void* x, Precision precision) override {
        if (precision != P) {
            throw std::runtime_error("AMGSolver: precision mismatch");
        }

        if (!setup_done_) {
            throw std::runtime_error("AMGSolver: setup() must be called before solve()");
        }

        const auto& rhs = *static_cast<const std::vector<Scalar>*>(b);
        auto& sol = *static_cast<std::vector<Scalar>*>(x);

        // amgcl 返回 std::tuple<size_t, Scalar>
        auto result = (*solver_)(*A_amgcl_, rhs, sol);
        size_t iters = std::get<0>(result);
        double error = static_cast<double>(std::get<1>(result));
        return {static_cast<int>(iters), error};
    }

private:
    AMGConfig config_;
    bool setup_done_;

    // amgcl 类型定义
    typedef amgcl::backend::builtin<Scalar> Backend;
    typedef typename Backend::matrix BackendMatrix;

    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::gauss_seidel
        >,
        amgcl::solver::cg<Backend>
    > Solver;

    std::unique_ptr<Solver> solver_;
    std::shared_ptr<BackendMatrix> A_amgcl_;

    // 适配后的矩阵数据（amgcl 使用 ptrdiff_t）
    std::vector<ptrdiff_t> ptr_;
    std::vector<ptrdiff_t> col_;
    std::vector<Scalar> val_;
};

} // anonymous namespace

// ============================================================================
// AMGSolver 模板实现
// ============================================================================

template<Precision P>
AMGSolver<P>::AMGSolver(const AMGConfig& config)
    : config_(config),
      impl_(std::make_unique<AMGSolverImplTyped<P>>(config)) {
}

template<Precision P>
AMGSolver<P>::~AMGSolver() = default;

template<Precision P>
SolveStats AMGSolver<P>::solve(const Matrix& A, const Vector& b, Vector& x) {
    // 设置求解器
    impl_->setup(&A, P);

    // 初始化解向量
    int n = A.rows;
    if (x.size() != static_cast<size_t>(n)) {
        x.resize(n, Scalar(0));
    } else {
        // 大小正确时也要清零，确保初始值为 0
        std::fill(x.begin(), x.end(), Scalar(0));
    }

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 求解
    int iters;
    double error;
    std::tie(iters, error) = impl_->solve(&b, &x, P);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    // 准备统计信息
    SolveStats stats;
    stats.iterations = iters;
    stats.converged = (error < config_.tolerance);  // 使用实际残差判断收敛

    // 计算相对残差 ||b - Ax|| / ||b||
    Vector ax(n);
    CPUOps::spmv<P>(n, A.row_ptr, A.col_ind, A.values, x, ax);

    double r_norm = 0;
    double b_norm = 0;
    for (int i = 0; i < n; ++i) {
        Scalar r = b[i] - ax[i];
        r_norm += static_cast<double>(r * r);
        b_norm += static_cast<double>(b[i] * b[i]);
    }

    stats.final_residual = std::sqrt(r_norm) / std::sqrt(b_norm);
    stats.solve_time = solve_time;

    return stats;
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class AMGSolver<Precision::Float32>;
template class AMGSolver<Precision::Float64>;
