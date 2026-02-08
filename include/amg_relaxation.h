#ifndef AMG_RELAXATION_H
#define AMG_RELAXATION_H

#include "precision_traits.h"
#include <vector>

// ============================================================================
// AMG 松弛算子
// ============================================================================

template<Precision P>
class AMGRelaxation {
public:
    using Scalar = PRECISION_SCALAR(P);

    // Gauss-Seidel松弛（串行前向）
    static void gauss_seidel(
        int n,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        const std::vector<Scalar>& values,
        const std::vector<Scalar>& b,
        std::vector<Scalar>& x,
        int steps = 1
    );
};

#endif // AMG_RELAXATION_H
