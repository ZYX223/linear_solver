#include "preconditioner.h"
#include <stdio.h>

// ============================================================================
// GPUILUPreconditioner 模板实现
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

    CHECK_CUDA(cudaMalloc(&d_valsILU0_, nnz_ * sizeof(typename GPUILUPreconditioner<P>::Scalar)));
    CHECK_CUDA(cudaMemcpy(d_valsILU0_, A.d_values, nnz_ * sizeof(typename GPUILUPreconditioner<P>::Scalar),
                          cudaMemcpyDeviceToDevice));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&matLU_descr_));
    CHECK_CUSPARSE(cusparseSetMatType(matLU_descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(matLU_descr_, CUSPARSE_INDEX_BASE_ZERO));

    CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&ilu0_info_));

    sparse_->ilu0_setup(rows_, nnz_, matLU_descr_,
                        d_valsILU0_, A.d_row_ptr, A.d_col_ind,
                        ilu0_info_, &d_bufferILU0_, &bufferILU0_size_);

    sparse_->ilu0_compute(rows_, nnz_, matLU_descr_,
                          d_valsILU0_, A.d_row_ptr, A.d_col_ind,
                          ilu0_info_, d_bufferILU0_);

    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    CHECK_CUSPARSE(cusparseCreateCsr(&matL_, rows_, rows_, nnz_,
                                     A.d_row_ptr, A.d_col_ind, d_valsILU0_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_lower, sizeof(fill_lower)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_unit, sizeof(diag_unit)));

    CHECK_CUSPARSE(cusparseCreateCsr(&matU_, rows_, rows_, nnz_,
                                     A.d_row_ptr, A.d_col_ind, d_valsILU0_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CudaDataType<P>::value));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU_, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_upper, sizeof(fill_upper)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matU_, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_non_unit, sizeof(diag_non_unit)));

    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL_));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrU_));

    CHECK_CUDA(cudaMalloc(&d_y_, rows_ * sizeof(typename GPUILUPreconditioner<P>::Scalar)));

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

    sparse_->triangular_solve(matL_, spsvDescrL_, vecR, vecY);
    sparse_->triangular_solve(matU_, spsvDescrU_, vecY, vecZ);

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecZ));
}

// ============================================================================
// CPUILUPreconditioner 模板实现
// ============================================================================

template<Precision P>
CPUILUPreconditioner<P>::CPUILUPreconditioner() : n_(0) {}

template<Precision P>
CPUILUPreconditioner<P>::~CPUILUPreconditioner() {}

template<Precision P>
void CPUILUPreconditioner<P>::setup(const SparseMatrix<P>& A) {
    setup(A.rows, A.row_ptr, A.col_ind, A.values);
}

template<Precision P>
void CPUILUPreconditioner<P>::setup(int n, const std::vector<int>& row_ptr,
                                     const std::vector<int>& col_ind,
                                     const std::vector<typename CPUILUPreconditioner<P>::Scalar>& values) {
    n_ = n;
    row_ptr_ = row_ptr;
    col_ind_ = col_ind;
    values_ = values;

    for (int i = 0; i < n_; i++) {
        for (int k_idx = row_ptr_[i]; k_idx < row_ptr_[i + 1]; k_idx++) {
            int k = col_ind_[k_idx];
            if (k >= i) break;

            typename CPUILUPreconditioner<P>::Scalar a_ik = values_[k_idx];

            typename CPUILUPreconditioner<P>::Scalar a_kk = ScalarConstants<P>::zero();
            for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                if (col_ind_[p] == k) {
                    a_kk = values_[p];
                    break;
                }
            }

            if (a_kk != ScalarConstants<P>::zero()) {
                a_ik /= a_kk;
                values_[k_idx] = a_ik;

                for (int j_idx = k_idx + 1; j_idx < row_ptr_[i + 1]; j_idx++) {
                    int j = col_ind_[j_idx];
                    if (j <= k) continue;

                    typename CPUILUPreconditioner<P>::Scalar a_kj = ScalarConstants<P>::zero();
                    for (int p = row_ptr_[k]; p < row_ptr_[k + 1]; p++) {
                        if (col_ind_[p] == j) {
                            a_kj = values_[p];
                            break;
                        }
                    }

                    if (a_kj != ScalarConstants<P>::zero()) {
                        values_[j_idx] -= a_ik * a_kj;
                    }
                }
            }
        }
    }
}

template<Precision P>
void CPUILUPreconditioner<P>::apply(const std::vector<typename CPUILUPreconditioner<P>::Scalar>& r,
                                     std::vector<typename CPUILUPreconditioner<P>::Scalar>& z) const {
    std::vector<typename CPUILUPreconditioner<P>::Scalar> y(n_);
    forward_substitute(r, y);
    backward_substitute(y, z);
}

template<Precision P>
void CPUILUPreconditioner<P>::forward_substitute(const std::vector<typename CPUILUPreconditioner<P>::Scalar>& b,
                                                  std::vector<typename CPUILUPreconditioner<P>::Scalar>& y) const {
    for (int i = 0; i < n_; i++) {
        typename CPUILUPreconditioner<P>::Scalar sum = b[i];
        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; j++) {
            int col = col_ind_[j];
            if (col < i) {
                sum -= values_[j] * y[col];
            }
        }
        y[i] = sum;
    }
}

template<Precision P>
void CPUILUPreconditioner<P>::backward_substitute(const std::vector<typename CPUILUPreconditioner<P>::Scalar>& y,
                                                   std::vector<typename CPUILUPreconditioner<P>::Scalar>& x) const {
    for (int i = n_ - 1; i >= 0; i--) {
        typename CPUILUPreconditioner<P>::Scalar sum = y[i];
        typename CPUILUPreconditioner<P>::Scalar diag = ScalarConstants<P>::one();

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

// ============================================================================
// 显式模板实例化
// ============================================================================

template class GPUILUPreconditioner<Precision::Float32>;
template class GPUILUPreconditioner<Precision::Float64>;

template class CPUILUPreconditioner<Precision::Float32>;
template class CPUILUPreconditioner<Precision::Float64>;
