#include "preconditioner.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// GPU AMG 预条件子实现
// 使用 AMGCL 原生 CUDA 后端 + Damped Jacobi 松弛
// ============================================================================

template<Precision P>
GPUAMGPreconditioner<P>::GPUAMGPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse,
                                                std::shared_ptr<AMGConfig> config)
    : sparse_(sparse), config_(config), is_setup_(false) {
    if (!config_) config_ = std::make_shared<AMGConfig>();

    // 初始化 CUDA backend 参数，使用 CUSPARSE handle
    backend_params_ = typename Backend::params(sparse_->handle());
}

template<Precision P>
GPUAMGPreconditioner<P>::~GPUAMGPreconditioner() {
    // thrust::device_vector 自动释放，无需手动清理
}

template<Precision P>
void GPUAMGPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    n_ = A.rows;
    nnz_ = A.nnz;

    // 转换矩阵为 amgcl 格式（使用 ptrdiff_t 类型）
    ptr_.resize(n_ + 1);
    col_.resize(nnz_);
    val_.resize(nnz_);
    for (size_t i = 0; i <= n_; ++i) ptr_[i] = A.row_ptr[i];
    for (size_t i = 0; i < nnz_; ++i) col_[i] = A.col_ind[i];
    for (size_t i = 0; i < nnz_; ++i) val_[i] = A.values[i];

    // 创建 CPU 端矩阵用于构建层级
    A_build_ = std::make_shared<typename BuildBackend::matrix>(n_, n_, ptr_, col_, val_);

    // 配置 amgcl AMG 预条件子参数
    typename AMGPreconditioner::params prm;
    prm.coarsening.aggr.eps_strong = config_->aggregation_eps;
    prm.max_levels = config_->max_levels;
    prm.coarse_enough = config_->coarse_grid_size;
    prm.npre = config_->pre_smooth_steps;
    prm.npost = config_->post_smooth_steps;
    prm.relax.damping = config_->damping_factor;
    prm.pre_cycles = config_->pre_cycles;

    // 使用 CUDA 后端创建 AMG 预条件子
    // AMGCL 会自动将层级结构传输到 GPU
    amg_prec_ = std::make_shared<AMGPreconditioner>(A_build_, prm, backend_params_);

    // 分配 GPU 端向量
    r_dev_.resize(n_);
    z_dev_.resize(n_);
    is_setup_ = true;
}

template<Precision P>
void GPUAMGPreconditioner<P>::apply(const GPUVector<P>& r, GPUVector<P>& z) const {
    if (!is_setup_) return;

    // 使用 cudaMemcpy 进行设备内存拷贝
    // r -> r_dev_
    cudaMemcpy(thrust::raw_pointer_cast(r_dev_.data()),
               r.d_data,
               n_ * sizeof(Scalar),
               cudaMemcpyDeviceToDevice);

    // 应用 amgcl CUDA 预条件子（完全在 GPU 上执行）
    amg_prec_->apply(r_dev_, z_dev_);

    // z_dev_ -> z
    cudaMemcpy(z.d_data,
               thrust::raw_pointer_cast(z_dev_.data()),
               n_ * sizeof(Scalar),
               cudaMemcpyDeviceToDevice);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

#ifndef __CPU_ONLY__
template class GPUAMGPreconditioner<Precision::Float32>;
template class GPUAMGPreconditioner<Precision::Float64>;
#endif
