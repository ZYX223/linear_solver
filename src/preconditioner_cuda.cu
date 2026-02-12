#include "preconditioner.h"
#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// ============================================================================
// GPU AMG 预条件子实现（使用 NVCC 编译）
// ============================================================================

template<Precision P>
GPUAMGPreconditioner<P>::GPUAMGPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse,
                                                std::shared_ptr<AMGConfig> config)
    : sparse_(sparse), config_(config), is_setup_(false) {
    if (!config_) {
        config_ = std::make_shared<AMGConfig>();
    }
}

template<Precision P>
void GPUAMGPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    // 1. 转换矩阵到主机端 amgcl 格式
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

    // 2. 创建主机端 amgcl 矩阵
    A_host_ = std::make_shared<typename BackendHost::matrix>(n, n, ptr_, col_, val_);

    // 3. 创建 CUDA 后端参数
    typename BackendDevice::params cuda_params(sparse_->handle());

    // 4. 配置 AMG 参数
    typename AMGPreconditioner::params prm;
    prm.coarsening.aggr.eps_strong = static_cast<float>(config_->aggregation_eps);
    prm.max_levels = config_->max_levels;
    prm.coarse_enough = config_->coarse_grid_size;
    prm.npre = config_->pre_smooth_steps;
    prm.npost = config_->post_smooth_steps;

    // 5. 创建 GPU AMG 预条件子
    // 注意：AMG 构造函数会自动将输入矩阵转换为 build_matrix (builtin::matrix)
    // 然后在主机端构建层次结构，最后在后端 (CUDA) 上执行
    amg_prec_ = std::make_shared<AMGPreconditioner>(*A_host_, prm, cuda_params);

    is_setup_ = true;
}

template<Precision P>
void GPUAMGPreconditioner<P>::apply(const GPUVector<P>& r, GPUVector<P>& z) const {
    if (!is_setup_) {
        printf("Error: GPUAMGPreconditioner not setup!\n");
        exit(1);
    }

    // 创建临时 thrust 向量（amgcl CUDA 后端要求）
    thrust::device_vector<Scalar> r_device(r.n);
    thrust::device_vector<Scalar> z_device(z.n);

    // 拷贝输入数据：GPUVector → thrust vector
    thrust::copy(
        thrust::device_ptr<Scalar>(r.d_data),
        thrust::device_ptr<Scalar>(r.d_data + r.n),
        r_device.begin()
    );

    // 应用 AMG V-cycle（全部 GPU 操作）
    amg_prec_->apply(r_device, z_device);

    // 拷贝输出数据：thrust vector → GPUVector
    thrust::copy(
        z_device.begin(),
        z_device.end(),
        thrust::device_ptr<Scalar>(z.d_data)
    );
}

// ============================================================================
// 显式模板实例化（NVCC 编译）
// ============================================================================

#ifndef __CPU_ONLY__
template class GPUAMGPreconditioner<Precision::Float32>;
template class GPUAMGPreconditioner<Precision::Float64>;
#endif
