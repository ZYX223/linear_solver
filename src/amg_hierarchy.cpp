#include "amg_hierarchy.h"
#include "amg_solver.h"
#include "amg_coarsening.h"
#include "amg_relaxation.h"
#include "amg_interpolation.h"
#include "sparse_utils.h"
#include <algorithm>

// ============================================================================
// AMGHierarchy 模板实现
// ============================================================================

template<Precision P>
void AMGHierarchy<P>::setup(const SparseMatrix<P>& A, const AMGConfig& config) {
    levels.clear();

    // 创建最细网格层（第0层）
    auto level0 = std::make_unique<AMGLevel<P>>(A.rows);
    level0->row_ptr = A.row_ptr;
    level0->col_ind = A.col_ind;
    level0->values = A.values;

    // 计算对角线逆
    AMGInterpolation<P>::compute_diag_inv(A.rows, A.row_ptr, A.col_ind,
                                          A.values, level0->diag_inv);

    levels.push_back(std::move(level0));

    // 构建多层结构
    int current_level = 0;
    while (current_level < config.max_levels - 1) {
        const auto& fine_level = levels[current_level];
        int n_fine = fine_level->n;

        // 检查是否已经足够粗糙
        if (n_fine <= config.coarse_grid_size) {
            break;
        }

        // 创建临时SparseMatrix用于聚合
        SparseMatrix<P> A_temp(n_fine, n_fine, fine_level->row_ptr[n_fine]);
        A_temp.row_ptr = fine_level->row_ptr;
        A_temp.col_ind = fine_level->col_ind;
        A_temp.values = fine_level->values;

        // 1. 构建聚合
        std::vector<int> aggregates = SmoothedAggregation<P>::build_aggregates(
            A_temp, config.aggregation_eps
        );

        // 计算粗网格大小
        int n_coarse = 0;
        for (int agg : aggregates) {
            n_coarse = std::max(n_coarse, agg + 1);
        }

        // 检查粗网格是否太小
        if (n_coarse >= n_fine * 3 / 4) {
            // 粗化效果不好，停止
            break;
        }

        // 2. 构建常数插值算子
        std::vector<int> P_row_ptr, P_col_ind;
        std::vector<Scalar> P_values;

        SmoothedAggregation<P>::build_constant_interpolation(
            A_temp, aggregates, P_row_ptr, P_col_ind, P_values
        );

        // 3. 平滑插值算子
        SmoothedAggregation<P>::smooth_interpolation(
            A_temp, P_row_ptr, P_col_ind, P_values, Scalar(2.0/3.0)
        );

        // 4. 计算限制算子 R = P^T
        std::vector<int> R_row_ptr, R_col_ind;
        std::vector<Scalar> R_values;

        AMGInterpolation<P>::transpose_csr(
            n_fine, n_coarse,
            P_row_ptr, P_col_ind, P_values,
            R_row_ptr, R_col_ind, R_values
        );

        // 5. Galerkin乘积: A_coarse = R * A * P
        std::vector<int> Ac_row_ptr, Ac_col_ind;
        std::vector<Scalar> Ac_values;

        AMGInterpolation<P>::galerkin_product(
            n_fine, n_coarse,
            R_row_ptr, R_col_ind, R_values,
            fine_level->row_ptr, fine_level->col_ind, fine_level->values,
            P_row_ptr, P_col_ind, P_values,
            Ac_row_ptr, Ac_col_ind, Ac_values
        );

        // 6. 创建粗网格层
        auto coarse_level = std::make_unique<AMGLevel<P>>(n_coarse);
        coarse_level->row_ptr = Ac_row_ptr;
        coarse_level->col_ind = Ac_col_ind;
        coarse_level->values = Ac_values;

        // 计算粗网格层的对角线逆
        AMGInterpolation<P>::compute_diag_inv(n_coarse, Ac_row_ptr, Ac_col_ind,
                                              Ac_values, coarse_level->diag_inv);

        // 7. 将插值和限制算子存储到细网格层
        levels[current_level]->P_row_ptr = P_row_ptr;
        levels[current_level]->P_col_ind = P_col_ind;
        levels[current_level]->P_values = P_values;
        levels[current_level]->R_row_ptr = R_row_ptr;
        levels[current_level]->R_col_ind = R_col_ind;
        levels[current_level]->R_values = R_values;

        levels.push_back(std::move(coarse_level));
        current_level++;
    }
}

template<Precision P>
void AMGHierarchy<P>::cycle(const Vector& b, Vector& x, int level, const AMGConfig& config) {
    auto& lvl = *levels[level];

    // 1. 前光滑
    AMGRelaxation<P>::gauss_seidel(lvl.n, lvl.row_ptr, lvl.col_ind,
                                   lvl.values, b, x, config.pre_smooth_steps);

    // 2. 计算残差 r = b - A*x
    Vector r(lvl.n);
    CPUOps::spmv<P>(lvl.n, lvl.row_ptr, lvl.col_ind, lvl.values, x, r);
    for (int i = 0; i < lvl.n; ++i) {
        r[i] = b[i] - r[i];
    }

    // 3. 最粗网格直接求解
    if (level == static_cast<int>(levels.size()) - 1) {
        solve_coarse_grid(b, x);
        return;
    }

    // 4. 限制: r_c = R * r
    int nc = levels[level + 1]->n;
    Vector r_c(nc, Scalar(0));
    CPUOps::spmv<P>(nc, lvl.R_row_ptr, lvl.R_col_ind, lvl.R_values, r, r_c);

    // 5. 递归V-cycle
    Vector e_c(nc, Scalar(0));
    Vector b_c = r_c;  // 在粗网格上求解 A_c * e_c = r_c
    cycle(b_c, e_c, level + 1, config);

    // 6. 插值校正: x += P * e_c
    Vector corr(lvl.n, Scalar(0));
    CPUOps::spmv<P>(lvl.n, lvl.P_row_ptr, lvl.P_col_ind, lvl.P_values, e_c, corr);
    for (int i = 0; i < lvl.n; ++i) {
        x[i] += corr[i];
    }

    // 7. 后光滑
    AMGRelaxation<P>::gauss_seidel(lvl.n, lvl.row_ptr, lvl.col_ind,
                                   lvl.values, b, x, config.post_smooth_steps);
}

template<Precision P>
void AMGHierarchy<P>::solve_coarse_grid(const Vector& b, Vector& x) {
    // 最粗网格求解：使用简单的迭代方法
    int n = levels.back()->n;
    const auto& lvl = levels.back();

    // 对于小规模问题，使用多次Gauss-Seidel迭代
    int max_iters = 50;
    Vector r(n);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Gauss-Seidel迭代
        AMGRelaxation<P>::gauss_seidel(n, lvl->row_ptr, lvl->col_ind,
                                       lvl->values, b, x, 1);

        // 计算残差
        CPUOps::spmv<P>(n, lvl->row_ptr, lvl->col_ind, lvl->values, x, r);
        double residual = 0.0;
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - r[i];
            residual += static_cast<double>(r[i] * r[i]);
        }
        residual = std::sqrt(residual);

        // 收敛检查
        if (residual < 1e-10) {
            break;
        }
    }
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class AMGHierarchy<Precision::Float32>;
template class AMGHierarchy<Precision::Float64>;
