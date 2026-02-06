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
// ILUPreconditioner 实现
// ============================================================================

ILUPreconditioner::ILUPreconditioner(std::shared_ptr<CUSparseWrapper> sparse)
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

ILUPreconditioner::~ILUPreconditioner() {
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

void ILUPreconditioner::setup(const SparseMatrix& A) {
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

void ILUPreconditioner::apply(const GPUVector& r, GPUVector& z) {
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
