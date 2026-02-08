#include "amg_relaxation.h"
#include <algorithm>

// ============================================================================
// AMGRelaxation 模板实现
// ============================================================================

template<Precision P>
void AMGRelaxation<P>::gauss_seidel(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_ind,
    const std::vector<Scalar>& values,
    const std::vector<Scalar>& b,
    std::vector<Scalar>& x,
    int steps
) {
    for (int step = 0; step < steps; ++step) {
        // 前向Gauss-Seidel
        for (int i = 0; i < n; ++i) {
            Scalar diagonal = Scalar(0);
            Scalar sum = Scalar(0);

            // 计算非对角元素的贡献
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                int col = col_ind[j];
                if (col == i) {
                    diagonal = values[j];
                } else {
                    sum += values[j] * x[col];
                }
            }

            // 更新 x[i]
            if (diagonal != Scalar(0)) {
                x[i] = (b[i] - sum) / diagonal;
            }
        }
    }
}

// ============================================================================
// 显式模板实例化
// ============================================================================

template class AMGRelaxation<Precision::Float32>;
template class AMGRelaxation<Precision::Float64>;
