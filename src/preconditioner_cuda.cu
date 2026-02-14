#include "preconditioner.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// 抑制 nvcc 的废弃 API 警告
// ILU0/IC0 相关 API 在 CUDA 12.x 中已废弃
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

// ============================================================================
// GPUILUPreconditioner 模板实现（不完全 LU 分解）
// 使用 cuSPARSE 的 csrilu02 进行 ILU(0) 分解
// ============================================================================

template<Precision P>
GPUILUPreconditioner<P>::GPUILUPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse)
    : sparse_(sparse),
      rows_(0), nnz_(0),
      d_valsILU0_(nullptr),
      matLU_descr_(nullptr),
      ilu0_info_(nullptr),
      d_bufferILU0_(nullptr),
      bufferILU0_size_(0),
      matL_(nullptr), matU_(nullptr),
      spsvDescrL_(nullptr), spsvDescrU_(nullptr),
      d_bufferL_(nullptr), d_bufferU_(nullptr),
      bufferSizeL_(0), bufferSizeU_(0),
      d_y_(nullptr),
      is_setup_(false) {
}

template<Precision P>
GPUILUPreconditioner<P>::~GPUILUPreconditioner() {
    if (d_valsILU0_) cudaFree(d_valsILU0_);
    if (d_bufferILU0_) cudaFree(d_bufferILU0_);
    if (d_bufferL_) cudaFree(d_bufferL_);
    if (d_bufferU_) cudaFree(d_bufferU_);
    if (d_y_) cudaFree(d_y_);

    if (matLU_descr_) cusparseDestroyMatDescr(matLU_descr_);
    if (ilu0_info_) cusparseDestroyCsrilu02Info(ilu0_info_);
    if (matL_) cusparseDestroySpMat(matL_);
    if (matU_) cusparseDestroySpMat(matU_);
    if (spsvDescrL_) cusparseSpSV_destroyDescr(spsvDescrL_);
    if (spsvDescrU_) cusparseSpSV_destroyDescr(spsvDescrU_);
}

template<Precision P>
void GPUILUPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    rows_ = A.rows;
    nnz_ = A.nnz;

    using Scalar = typename GPUILUPreconditioner<P>::Scalar;

    // Step 1: 分配并复制矩阵值到 GPU
    CHECK_CUDA(cudaMalloc(&d_valsILU0_, nnz_ * sizeof(Scalar)));
    CHECK_CUDA(cudaMemcpy(d_valsILU0_, A.d_values, nnz_ * sizeof(Scalar),
                          cudaMemcpyDeviceToDevice));

    // Step 2: 创建矩阵描述符
    CHECK_CUSPARSE(cusparseCreateMatDescr(&matLU_descr_));
    CHECK_CUSPARSE(cusparseSetMatType(matLU_descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(matLU_descr_, CUSPARSE_INDEX_BASE_ZERO));

    // Step 3: 创建 ILU(0) info 对象
    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&ilu0_info_));

    // Step 4: ILU(0) 分析和计算
    sparse_->ilu0_setup(rows_, nnz_, matLU_descr_,
                        d_valsILU0_, A.d_row_ptr, A.d_col_ind,
                        ilu0_info_, &d_bufferILU0_, &bufferILU0_size_);

    sparse_->ilu0_compute(rows_, nnz_, matLU_descr_,
                          d_valsILU0_, A.d_row_ptr, A.d_col_ind,
                          ilu0_info_, d_bufferILU0_);

    // Step 5: 创建 L 和 U 矩阵描述符（共享相同的 CSR 数据）
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    // L 矩阵：下三角 + 单位对角线
    CHECK_CUSPARSE(cusparseCreateCsr(&matL_, rows_, rows_, nnz_,
                                     A.d_row_ptr, A.d_col_ind, d_valsILU0_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_lower, sizeof(fill_lower)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_unit, sizeof(diag_unit)));

    // U 矩阵：上三角 + 非单位对角线
    CHECK_CUSPARSE(cusparseCreateCsr(&matU_, rows_, rows_, nnz_,
                                     A.d_row_ptr, A.d_col_ind, d_valsILU0_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_upper, sizeof(fill_upper)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_non_unit, sizeof(diag_non_unit)));

    // Step 6: 创建三角求解描述符
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL_));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrU_));

    // 分配辅助向量
    CHECK_CUDA(cudaMalloc(&d_y_, rows_ * sizeof(Scalar)));

    // Step 7: 分析三角求解
    GPUVector<P> dummy_r(rows_);
    GPUVector<P> dummy_x(rows_);
    auto vecR = dummy_r.create_dnvec_descr();
    auto vecX = dummy_x.create_dnvec_descr();

    sparse_->triangular_solve_setup(matL_, spsvDescrL_, vecR, vecX,
                                    &d_bufferL_, &bufferSizeL_);
    sparse_->triangular_solve_setup(matU_, spsvDescrU_, vecR, vecX,
                                    &d_bufferU_, &bufferSizeU_);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));

    is_setup_ = true;
}

template<Precision P>
void GPUILUPreconditioner<P>::apply(const GPUVector<P>& r, GPUVector<P>& z) const {
    if (!is_setup_) {
        printf("Error: ILUPreconditioner not setup!\n");
        exit(1);
    }

    auto vecR = r.create_dnvec_descr();
    auto vecY = cusparseDnVecDescr_t();
    auto vecZ = z.create_dnvec_descr();

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows_, d_y_, CudaDataType<P>::value));

    // 两步三角求解: L * y = r, U * z = y
    sparse_->triangular_solve(matL_, spsvDescrL_, vecR, vecY);
    sparse_->triangular_solve(matU_, spsvDescrU_, vecY, vecZ);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));
}

// ============================================================================
// GPUIPCPreconditioner 模板实现（不完全 Cholesky 分解）
// 实现：CPU 分解得到上三角 R + GPU 三角求解
// 使用 NVIDIA 推荐方式：A ≈ R^T * R，用 TRANSPOSE 操作避免显式构建 R^T
// 参考：NVIDIA cuSPARSE White Paper "Incomplete-LU and Cholesky Preconditioned Iterative Methods"
// ============================================================================

template<Precision P>
GPUIPCPreconditioner<P>::GPUIPCPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse)
    : sparse_(sparse),
      rows_(0), nnz_(0),
      d_row_ptr_(nullptr),
      d_col_ind_(nullptr),
      d_values_(nullptr),
      matR_(nullptr),
      spsvDescrRt_(nullptr), spsvDescrR_(nullptr),
      d_bufferRt_(nullptr), d_bufferR_(nullptr),
      bufferSizeRt_(0), bufferSizeR_(0),
      d_t_(nullptr),
      is_setup_(false) {
}

template<Precision P>
GPUIPCPreconditioner<P>::~GPUIPCPreconditioner() {
    if (d_row_ptr_) cudaFree(d_row_ptr_);
    if (d_col_ind_) cudaFree(d_col_ind_);
    if (d_values_) cudaFree(d_values_);
    if (d_bufferRt_) cudaFree(d_bufferRt_);
    if (d_bufferR_) cudaFree(d_bufferR_);
    if (d_t_) cudaFree(d_t_);

    if (matR_) cusparseDestroySpMat(matR_);
    if (spsvDescrRt_) cusparseSpSV_destroyDescr(spsvDescrRt_);
    if (spsvDescrR_) cusparseSpSV_destroyDescr(spsvDescrR_);
}

template<Precision P>
void GPUIPCPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    rows_ = A.rows;

    using Scalar = typename GPUIPCPreconditioner<P>::Scalar;

    // ========================================================================
    // Step 1: 在 CPU 上提取上三角部分并执行 IC(0) 分解
    // IC(0): A ≈ R^T * R，其中 R 是上三角矩阵
    // ========================================================================
    row_ptr_.resize(rows_ + 1, 0);

    // 统计每行上三角元素数量（包括对角线）
    for (int i = 0; i < rows_; i++) {
        int count = 0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] >= i) count++;  // 上三角：col >= row
        }
        row_ptr_[i + 1] = row_ptr_[i] + count;
    }

    nnz_ = row_ptr_[rows_];
    col_ind_.resize(nnz_);
    values_.resize(nnz_);

    // 填充上三角部分
    for (int i = 0; i < rows_; i++) {
        int idx = row_ptr_[i];
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] >= i) {
                col_ind_[idx] = A.col_ind[j];
                values_[idx] = A.values[j];
                idx++;
            }
        }
    }

    // IC(0) 分解: A ≈ R^T * R
    // R 是上三角，按列计算
    // 对于 col >= row 的元素 R[row,col]:
    //   R[row,col] = (A[row,col] - sum_{k<row} R[k,row] * R[k,col]) / R[row,row]  (row < col)
    //   R[row,row] = sqrt(A[row,row] - sum_{k<row} R[k,row]^2)                    (row == col)
    for (int row = 0; row < rows_; row++) {
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

    // ========================================================================
    // Step 2: 将 R 上传到 GPU
    // ========================================================================
    CHECK_CUDA(cudaMalloc(&d_row_ptr_, (rows_ + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values_, nnz_ * sizeof(Scalar)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr_, row_ptr_.data(), (rows_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind_, col_ind_.data(), nnz_ * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values_, values_.data(), nnz_ * sizeof(Scalar),
                          cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 3: 创建 R 矩阵描述符（上三角）
    // ========================================================================
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    CHECK_CUSPARSE(cusparseCreateCsr(&matR_, rows_, rows_, nnz_,
                                     d_row_ptr_, d_col_ind_, d_values_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matR_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_upper, sizeof(fill_upper)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matR_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_non_unit, sizeof(diag_non_unit)));

    // ========================================================================
    // Step 4: 创建三角求解描述符
    // R^T 求解（TRANSPOSE，相当于下三角）和 R 求解（NON_TRANSPOSE，上三角）
    // ========================================================================
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrRt_));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrR_));

    // 分配辅助向量
    CHECK_CUDA(cudaMalloc(&d_t_, rows_ * sizeof(Scalar)));

    // 分析三角求解
    GPUVector<P> dummy_r(rows_);
    GPUVector<P> dummy_x(rows_);
    auto vecR = dummy_r.create_dnvec_descr();
    auto vecX = dummy_x.create_dnvec_descr();

    // R^T 求解分析（使用 TRANSPOSE 操作）
    Scalar one = ScalarConstants<P>::one();
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(sparse_->handle(), CUSPARSE_OPERATION_TRANSPOSE,
                                           &one, matR_, vecR, vecX,
                                           CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrRt_, &bufferSizeRt_));
    CHECK_CUDA(cudaMalloc(&d_bufferRt_, bufferSizeRt_));
    CHECK_CUSPARSE(cusparseSpSV_analysis(sparse_->handle(), CUSPARSE_OPERATION_TRANSPOSE,
                                         &one, matR_, vecR, vecX,
                                         CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                         spsvDescrRt_, d_bufferRt_));

    // R 求解分析（使用 NON_TRANSPOSE 操作）
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(sparse_->handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matR_, vecR, vecX,
                                           CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrR_, &bufferSizeR_));
    CHECK_CUDA(cudaMalloc(&d_bufferR_, bufferSizeR_));
    CHECK_CUSPARSE(cusparseSpSV_analysis(sparse_->handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one, matR_, vecR, vecX,
                                         CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                         spsvDescrR_, d_bufferR_));

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));

    is_setup_ = true;
}

template<Precision P>
void GPUIPCPreconditioner<P>::apply(const GPUVector<P>& r, GPUVector<P>& z) const {
    if (!is_setup_) {
        printf("Error: IPCPreconditioner not setup!\n");
        exit(1);
    }

    using Scalar = typename GPUIPCPreconditioner<P>::Scalar;
    Scalar one = ScalarConstants<P>::one();

    auto vecR = r.create_dnvec_descr();
    auto vecT = cusparseDnVecDescr_t();
    auto vecZ = z.create_dnvec_descr();

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecT, rows_, d_t_, CudaDataType<P>::value));

    // Step 1: R^T * t = r （使用 TRANSPOSE 操作，相当于下三角求解）
    CHECK_CUSPARSE(cusparseSpSV_solve(sparse_->handle(), CUSPARSE_OPERATION_TRANSPOSE,
                                      &one, matR_, vecR, vecT,
                                      CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                      spsvDescrRt_));

    // Step 2: R * z = t （使用 NON_TRANSPOSE 操作，上三角求解）
    CHECK_CUSPARSE(cusparseSpSV_solve(sparse_->handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &one, matR_, vecT, vecZ,
                                      CudaDataType<P>::value, CUSPARSE_SPSV_ALG_DEFAULT,
                                      spsvDescrR_));

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecT));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));
}

// ============================================================================
// GPU AMG 预条件子实现
// ============================================================================

template<Precision P>
GPUAMGPreconditioner<P>::GPUAMGPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse,
                                                std::shared_ptr<AMGConfig> config)
    : sparse_(sparse), config_(config), is_setup_(false) {
    if (!config_) config_ = std::make_shared<AMGConfig>();
    backend_params_ = typename Backend::params(sparse_->handle());
}

template<Precision P>
GPUAMGPreconditioner<P>::~GPUAMGPreconditioner() {}

template<Precision P>
void GPUAMGPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    n_ = A.rows;
    nnz_ = A.nnz;

    ptr_.resize(n_ + 1);
    col_.resize(nnz_);
    val_.resize(nnz_);
    for (size_t i = 0; i <= n_; ++i) ptr_[i] = A.row_ptr[i];
    for (size_t i = 0; i < nnz_; ++i) col_[i] = A.col_ind[i];
    for (size_t i = 0; i < nnz_; ++i) val_[i] = A.values[i];

    A_build_ = std::make_shared<typename BuildBackend::matrix>(n_, n_, ptr_, col_, val_);

    typename AMGPreconditioner::params prm;
    prm.coarsening.aggr.eps_strong = config_->aggregation_eps;
    prm.max_levels = config_->max_levels;
    prm.coarse_enough = config_->coarse_grid_size;
    prm.npre = config_->pre_smooth_steps;
    prm.npost = config_->post_smooth_steps;
    prm.relax.damping = config_->damping_factor;
    prm.pre_cycles = config_->pre_cycles;

    amg_prec_ = std::make_shared<AMGPreconditioner>(A_build_, prm, backend_params_);

    r_dev_.resize(n_);
    z_dev_.resize(n_);
    is_setup_ = true;
}

template<Precision P>
void GPUAMGPreconditioner<P>::apply(const GPUVector<P>& r, GPUVector<P>& z) const {
    if (!is_setup_) return;

    cudaMemcpy(thrust::raw_pointer_cast(r_dev_.data()),
               r.d_data, n_ * sizeof(Scalar), cudaMemcpyDeviceToDevice);

    amg_prec_->apply(r_dev_, z_dev_);

    cudaMemcpy(z.d_data,
               thrust::raw_pointer_cast(z_dev_.data()),
               n_ * sizeof(Scalar), cudaMemcpyDeviceToDevice);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

#ifndef __CPU_ONLY__
template class GPUILUPreconditioner<Precision::Float32>;
template class GPUILUPreconditioner<Precision::Float64>;

template class GPUIPCPreconditioner<Precision::Float32>;
template class GPUIPCPreconditioner<Precision::Float64>;

template class GPUAMGPreconditioner<Precision::Float32>;
template class GPUAMGPreconditioner<Precision::Float64>;
#endif

// 恢复废弃 API 警告
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
#endif
