#include "preconditioner.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <unordered_map>
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
// 实现：CPU 分解 + GPU 三角求解，使用隐式 L^T（TRANSPOSE）
// 优化：页锁定内存加速数据传输
// ============================================================================

template<Precision P>
GPUIPCPreconditioner<P>::GPUIPCPreconditioner(std::shared_ptr<CUSparseWrapper<P>> sparse)
    : sparse_(sparse),
      rows_(0), nnz_(0),
      h_row_ptr_(nullptr), h_col_ind_(nullptr), h_values_(nullptr),
      d_row_ptr_(nullptr), d_col_ind_(nullptr), d_values_(nullptr),
      matL_(nullptr),
      spsvDescrL_(nullptr), spsvDescrLt_(nullptr),
      d_bufferL_(nullptr), d_bufferLt_(nullptr),
      bufferSizeL_(0), bufferSizeLt_(0),
      d_y_(nullptr),
      is_setup_(false) {
}

template<Precision P>
GPUIPCPreconditioner<P>::~GPUIPCPreconditioner() {
    // 释放页锁定内存
    if (h_row_ptr_) cudaFreeHost(h_row_ptr_);
    if (h_col_ind_) cudaFreeHost(h_col_ind_);
    if (h_values_) cudaFreeHost(h_values_);

    // 释放 GPU 内存
    if (d_row_ptr_) cudaFree(d_row_ptr_);
    if (d_col_ind_) cudaFree(d_col_ind_);
    if (d_values_) cudaFree(d_values_);
    if (d_bufferL_) cudaFree(d_bufferL_);
    if (d_bufferLt_) cudaFree(d_bufferLt_);
    if (d_y_) cudaFree(d_y_);

    // 释放 cuSPARSE 资源
    if (matL_) cusparseDestroySpMat(matL_);
    if (spsvDescrL_) cusparseSpSV_destroyDescr(spsvDescrL_);
    if (spsvDescrLt_) cusparseSpSV_destroyDescr(spsvDescrLt_);
}

template<Precision P>
void GPUIPCPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    rows_ = A.rows;

    // ========================================================================
    // Step 1: 分配页锁定内存（Pinned Memory，加速 H2D 传输）
    // ========================================================================
    // 先统计下三角元素数量
    std::vector<int> tmp_row_ptr(rows_ + 1, 0);
    for (int i = 0; i < rows_; i++) {
        int count = 0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] <= i) count++;
        }
        tmp_row_ptr[i + 1] = tmp_row_ptr[i] + count;
    }
    nnz_ = tmp_row_ptr[rows_];

    // 分配页锁定内存
    CHECK_CUDA(cudaMallocHost(&h_row_ptr_, (rows_ + 1) * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_col_ind_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_values_, nnz_ * sizeof(Scalar)));

    // 复制 row_ptr
    std::copy(tmp_row_ptr.begin(), tmp_row_ptr.end(), h_row_ptr_);

    // 填充下三角部分
    for (int i = 0; i < rows_; i++) {
        int idx = h_row_ptr_[i];
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (A.col_ind[j] <= i) {
                h_col_ind_[idx] = A.col_ind[j];
                h_values_[idx] = A.values[j];
                idx++;
            }
        }
    }

    // ========================================================================
    // Step 2: 在 CPU 上执行 IC(0) 分解
    // IC(0): A ≈ L * L^T
    // ========================================================================
    // 构建列索引映射（加速 O(1) 查找）
    std::vector<std::unordered_map<int, int>> col_map(rows_);
    for (int row = 0; row < rows_; row++) {
        for (int j = h_row_ptr_[row]; j < h_row_ptr_[row + 1]; j++) {
            col_map[row][h_col_ind_[j]] = j;
        }
    }

    // 构建列到行的反向映射
    std::vector<std::vector<int>> col_rows(rows_);
    for (int k = 0; k < rows_; k++) {
        for (int j = h_row_ptr_[k]; j < h_row_ptr_[k + 1]; j++) {
            int col = h_col_ind_[j];
            col_rows[col].push_back(k);
        }
    }

    // IC(0) 分解
    for (int row = 0; row < rows_; row++) {
        auto it_diag = col_map[row].find(row);
        if (it_diag == col_map[row].end()) continue;
        int diag_idx = it_diag->second;

        // 计算非对角线元素 L[row,col]
        for (int j_idx = h_row_ptr_[row]; j_idx < diag_idx; j_idx++) {
            int col = h_col_ind_[j_idx];
            Scalar sum_prod = ScalarConstants<P>::zero();
            const auto& rows_row = col_rows[row];
            const auto& rows_col = col_rows[col];

            size_t i1 = 0, i2 = 0;
            while (i1 < rows_row.size() && i2 < rows_col.size()) {
                int k1 = rows_row[i1];
                int k2 = rows_col[i2];
                if (k1 >= col || k2 >= col) break;
                if (k1 == k2) {
                    auto it_row = col_map[row].find(k1);
                    auto it_col = col_map[col].find(k1);
                    if (it_row != col_map[row].end() && it_col != col_map[col].end()) {
                        sum_prod += h_values_[it_row->second] * h_values_[it_col->second];
                    }
                    i1++; i2++;
                } else if (k1 < k2) {
                    i1++;
                } else {
                    i2++;
                }
            }

            auto it_col_diag = col_map[col].find(col);
            if (it_col_diag != col_map[col].end()) {
                Scalar l_col_col = h_values_[it_col_diag->second];
                if (l_col_col != ScalarConstants<P>::zero()) {
                    h_values_[j_idx] = (h_values_[j_idx] - sum_prod) / l_col_col;
                }
            }
        }

        // 计算对角线元素 L[row,row]（带数值稳定性处理）
        Scalar sum_sq = ScalarConstants<P>::zero();
        for (int k : col_rows[row]) {
            if (k < row) {
                auto it = col_map[row].find(k);
                if (it != col_map[row].end()) {
                    Scalar l_row_k = h_values_[it->second];
                    sum_sq += l_row_k * l_row_k;
                }
            }
        }

        Scalar diag_val = h_values_[diag_idx] - sum_sq;
        if (diag_val > ScalarConstants<P>::zero()) {
            h_values_[diag_idx] = std::sqrt(diag_val);
        } else {
            // 数值稳定性处理：确保对角线为正
            h_values_[diag_idx] = std::sqrt(std::fabs(diag_val) + 1e-12);
        }
    }

    // ========================================================================
    // Step 3: 分配 GPU 内存并使用异步拷贝传输数据
    // ========================================================================
    CHECK_CUDA(cudaMalloc(&d_row_ptr_, (rows_ + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind_, nnz_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values_, nnz_ * sizeof(Scalar)));

    // 使用异步拷贝（页锁定内存支持异步传输）
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(d_row_ptr_, h_row_ptr_, (rows_ + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_col_ind_, h_col_ind_, nnz_ * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_values_, h_values_, nnz_ * sizeof(Scalar),
                               cudaMemcpyHostToDevice, stream));

    // 等待传输完成
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));

    // ========================================================================
    // Step 4: 创建 L 矩阵描述符（用于三角求解）
    // ========================================================================
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    CHECK_CUSPARSE(cusparseCreateCsr(&matL_, rows_, rows_, nnz_,
                                     d_row_ptr_, d_col_ind_, d_values_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_lower, sizeof(fill_lower)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_non_unit, sizeof(diag_non_unit)));

    // ========================================================================
    // Step 5: 创建三角求解描述符
    // L 求解用 NON_TRANSPOSE，L^T 求解用 TRANSPOSE（隐式转置，节省内存）
    // ========================================================================
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL_));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrLt_));

    // 分配辅助向量
    CHECK_CUDA(cudaMalloc(&d_y_, rows_ * sizeof(Scalar)));

    // 分析三角求解
    GPUVector<P> dummy_r(rows_);
    GPUVector<P> dummy_x(rows_);
    auto vecR = dummy_r.create_dnvec_descr();
    auto vecX = dummy_x.create_dnvec_descr();

    // L 求解分析（NON_TRANSPOSE）
    sparse_->triangular_solve_setup(matL_, spsvDescrL_, vecR, vecX,
                                    &d_bufferL_, &bufferSizeL_,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE);

    // L^T 求解分析（TRANSPOSE）
    sparse_->triangular_solve_setup(matL_, spsvDescrLt_, vecR, vecX,
                                    &d_bufferLt_, &bufferSizeLt_,
                                    CUSPARSE_OPERATION_TRANSPOSE);

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

    auto vecR = r.create_dnvec_descr();
    auto vecY = cusparseDnVecDescr_t();
    auto vecZ = z.create_dnvec_descr();

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows_, d_y_, CudaDataType<P>::value));

    // Step 1: L * y = r （下三角求解）
    sparse_->triangular_solve(matL_, spsvDescrL_, vecR, vecY,
                              CUSPARSE_OPERATION_NON_TRANSPOSE);

    // Step 2: L^T * z = y （隐式转置求解）
    sparse_->triangular_solve(matL_, spsvDescrLt_, vecY, vecZ,
                              CUSPARSE_OPERATION_TRANSPOSE);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
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
    if (!is_setup_) {
        printf("Error: GPUAMGPreconditioner not setup!\n");
        exit(1);
    }

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
