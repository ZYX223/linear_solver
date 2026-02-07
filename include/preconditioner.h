#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "sparse_utils.h"
#include <cusparse.h>
#include <memory>
#include <vector>

// ============================================================================
// 预处理器模板基类（统一 CPU/GPU 接口）
// ============================================================================
template<typename VectorType>
class PreconditionerBase {
public:
    virtual ~PreconditionerBase() = default;

    // 初始化预处理器（如ILU分解）
    virtual void setup(const SparseMatrix& A) = 0;

    // 应用预处理器: z = M^(-1) * r
    virtual void apply(const VectorType& r, VectorType& z) const = 0;
};

// ============================================================================
// GPU 预处理器
// ============================================================================

// ILU(0)预处理器（GPU）
class GPUILUPreconditioner : public PreconditionerBase<GPUVector> {
public:
    GPUILUPreconditioner(std::shared_ptr<CUSparseWrapper> sparse);
    ~GPUILUPreconditioner();

    void setup(const SparseMatrix& A) override;
    void apply(const GPUVector& r, GPUVector& z) const override;

private:
    std::shared_ptr<CUSparseWrapper> sparse_;

    int rows_, nnz_;

    // ILU(0)分解的L和U因子（存储在同一个数组中）
    float* d_valsILU0_;

    // 矩阵描述符
    cusparseMatDescr_t matLU_descr_;
    csrilu02Info_t ilu0_info_;

    // ILU(0) buffer
    void* d_bufferILU0_;
    int bufferILU0_size_;

    // L和U矩阵描述符（分别用于三角求解）
    cusparseSpMatDescr_t matL_;
    cusparseSpMatDescr_t matU_;

    // 三角求解描述符和buffer
    cusparseSpSVDescr_t spsvDescrL_;
    cusparseSpSVDescr_t spsvDescrU_;
    void* d_bufferL_;
    void* d_bufferU_;
    size_t bufferSizeL_;
    size_t bufferSizeU_;

    // 辅助向量（用于L和U之间的中间结果）
    float* d_y_;

    bool is_setup_;
};

// ============================================================================
// CPU ILU(0) 预处理器
// ============================================================================

class CPUILUPreconditioner : public PreconditionerBase<std::vector<float>> {
public:
    CPUILUPreconditioner();
    ~CPUILUPreconditioner() override;

    // 实现基类接口
    void setup(const SparseMatrix& A) override;
    void apply(const std::vector<float>& r, std::vector<float>& z) const override;

    // 重载版本（保持向后兼容）
    void setup(int n, const std::vector<int>& row_ptr,
              const std::vector<int>& col_ind,
              const std::vector<float>& values);

private:
    int n_;
    std::vector<int> row_ptr_;
    std::vector<int> col_ind_;
    std::vector<float> values_;  // ILU(0) 分解后的值

    void forward_substitute(const std::vector<float>& b, std::vector<float>& y) const;
    void backward_substitute(const std::vector<float>& y, std::vector<float>& x) const;
};

#endif // PRECONDITIONER_H
