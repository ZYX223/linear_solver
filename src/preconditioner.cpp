#include "preconditioner.h"
#include <stdio.h>

// ============================================================================
// NonePreconditioner 实现
// ============================================================================

void NonePreconditioner::setup(const SparseMatrix& A) {
    // 无需设置
}

void NonePreconditioner::apply(const GPUVector& r, GPUVector& z) {
    // z = r (单位预处理)
    CHECK_CUDA(cudaMemcpy(z.d_data, r.d_data, r.n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// ============================================================================
// GPUILUPreconditioner 实现（GPU ILU0 预处理器）
// ============================================================================

GPUILUPreconditioner::GPUILUPreconditioner(std::shared_ptr<CUSparseWrapper> sparse)
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

GPUILUPreconditioner::~GPUILUPreconditioner() {
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

void GPUILUPreconditioner::setup(const SparseMatrix& A) {
    rows_ = A.rows;
    nnz_ = A.nnz;

    // 1. 分配并复制A的值到ILU(0)存储
    CHECK_CUDA(cudaMalloc(&d_valsILU0_, nnz_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_valsILU0_, A.d_values, nnz_ * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // 2. 创建ILU(0)描述符
    CHECK_CUSPARSE(cusparseCreateMatDescr(&matLU_descr_));
    CHECK_CUSPARSE(cusparseSetMatType(matLU_descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(matLU_descr_, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&ilu0_info_));

    // 3. 执行ILU(0)分析（分配buffer）
    sparse_->ilu0_setup(rows_, nnz_, matLU_descr_,
                        d_valsILU0_, A.d_row_ptr, A.d_col_ind,
                        ilu0_info_, &d_bufferILU0_, &bufferILU0_size_);

    // 4. 执行ILU(0)分解
    sparse_->ilu0_compute(rows_, nnz_, matLU_descr_,
                          d_valsILU0_, A.d_row_ptr, A.d_col_ind,
                          ilu0_info_, d_bufferILU0_);

    // 5. 创建L和U矩阵描述符
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    // L矩阵 (下三角，单位对角)
    CHECK_CUSPARSE(cusparseCreateCsr(&matL_, rows_, rows_, nnz_,
                                     A.d_row_ptr, A.d_col_ind, d_valsILU0_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_lower, sizeof(fill_lower)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_unit, sizeof(diag_unit)));

    // U矩阵 (上三角，非单位对角)
    CHECK_CUSPARSE(cusparseCreateCsr(&matU_, rows_, rows_, nnz_,
                                     A.d_row_ptr, A.d_col_ind, d_valsILU0_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_upper, sizeof(fill_upper)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_non_unit, sizeof(diag_non_unit)));

    // 6. 创建SpSV描述符
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL_));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrU_));

    // 7. 分配辅助向量（用于L和U之间的中间结果）
    CHECK_CUDA(cudaMalloc(&d_y_, rows_ * sizeof(float)));

    // 8. 创建临时向量描述符用于analysis（后续apply时会重新创建）
    GPUVector dummy_r(rows_);
    GPUVector dummy_x(rows_);
    auto vecR = dummy_r.create_dnvec_descr();
    auto vecX = dummy_x.create_dnvec_descr();

    // 9. 执行三角求解分析
    sparse_->triangular_solve_setup(matL_, spsvDescrL_, vecR, vecX,
                                    &d_bufferL_, &bufferSizeL_);
    sparse_->triangular_solve_setup(matU_, spsvDescrU_, vecR, vecX,
                                    &d_bufferU_, &bufferSizeU_);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));

    is_setup_ = true;
}

void GPUILUPreconditioner::apply(const GPUVector& r, GPUVector& z) {
    if (!is_setup_) {
        printf("Error: ILUPreconditioner not setup!\n");
        exit(1);
    }

    // 创建向量描述符
    auto vecR = r.create_dnvec_descr();
    auto vecY = cusparseDnVecDescr_t();
    auto vecZ = z.create_dnvec_descr();

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows_, d_y_, CUDA_R_32F));

    // z = U^(-1) * L^(-1) * r
    // 先解下三角: y = L^(-1) * r
    sparse_->triangular_solve(matL_, spsvDescrL_, vecR, vecY);

    // 再解上三角: z = U^(-1) * y
    sparse_->triangular_solve(matU_, spsvDescrU_, vecY, vecZ);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));
}

// ============================================================================
// CPUIILUPreconditioner 实现（CPU ILU0 预处理器）
// ============================================================================

CPUIILUPreconditioner::CPUIILUPreconditioner() : n_(0) {}

CPUIILUPreconditioner::~CPUIILUPreconditioner() {}

void CPUIILUPreconditioner::setup(int n, const std::vector<int>& row_ptr,
                                  const std::vector<int>& col_ind,
                                  const std::vector<float>& values) {
    n_ = n;
    row_ptr_ = row_ptr;
    col_ind_ = col_ind;
    values_ = values;

    // ILU(0) 分解（基于 IKJ 版本的 Gaussian 消去法）
    for (int i = 0; i < n_; i++) {
        for (int k_idx = row_ptr_[i]; k_idx < row_ptr_[i + 1]; k_idx++) {
            int k = col_ind_[k_idx];

            if (k >= i) break;

            float a_ik = values_[k_idx];

            float a_kk = 0.0f;
            for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                if (col_ind_[p] == k) {
                    a_kk = values_[p];
                    break;
                }
            }

            if (a_kk != 0.0f) {
                a_ik /= a_kk;
                values_[k_idx] = a_ik;

                for (int j_idx = k_idx + 1; j_idx < row_ptr_[i + 1]; j_idx++) {
                    int j = col_ind_[j_idx];
                    if (j <= k) continue;

                    float a_kj = 0.0f;
                    for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                        if (col_ind_[p] == j) {
                            a_kj = values_[p];
                            break;
                        }
                    }

                    if (a_kj != 0.0f) {
                        values_[j_idx] -= a_ik * a_kj;
                    }
                }
            }
        }
    }
}

void CPUIILUPreconditioner::apply(const std::vector<float>& r, std::vector<float>& z) const {
    std::vector<float> y(n_);
    forward_substitute(r, y);
    backward_substitute(y, z);
}

void CPUIILUPreconditioner::forward_substitute(const std::vector<float>& b, std::vector<float>& y) const {
    for (int i = 0; i < n_; i++) {
        float sum = b[i];
        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            int col = col_ind_[j];
            if (col < i) {
                sum -= values_[j] * y[col];
            }
        }
        y[i] = sum;
    }
}

void CPUIILUPreconditioner::backward_substitute(const std::vector<float>& y, std::vector<float>& x) const {
    for (int i = n_ - 1; i >= 0; i--) {
        float sum = y[i];
        float diag = 1.0f;

        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            int col = col_ind_[j];
            if (col > i) {
                sum -= values_[j] * x[col];
            } else if (col == i) {
                diag = values_[j];
            }
        }

        x[i] = sum / diag;
    }
}
