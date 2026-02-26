#ifndef AMG_CONFIG_H
#define AMG_CONFIG_H

#include <limits>
#include "precision_traits.h"

// ============================================================================
// AMG 求解器配置
// ============================================================================

struct AMGConfig {
    // 求解器类型选择
    bool use_pcg = true;                // true: PCG (像amgcl), false: 纯AMG V-cycle
    bool use_scale = false;             // 是否使用对角缩放 (scale_diagonal)

    int max_iterations = 100;           // 最大迭代次数
    double tolerance = 1e-8;           // 与 amgcl 保持一致

    // AMG特定参数
    int max_levels = std::numeric_limits<int>::max(); 
    int coarse_grid_size = 3000;        // 与 skyline_lu 默认一致
    int pre_smooth_steps = 3;           // 前光滑步数（GPU推荐: 3）
    int post_smooth_steps = 3;          // 后光滑步数（GPU推荐: 3）
    int ncycle = 1;                     // V-cycle 数量（amgcl 默认=1）
    int pre_cycles = 2;                 // 预处理循环数（GPU推荐: 2）
    bool direct_coarse = true;            // 使用直接求解器（amgcl 默认=true）
    int block_size = 1;                 // 聚合块大小（1=标量，与 amgcl 默认值一致）
    double aggregation_eps = 0.08;      // amgcl 默认 0.08，smoothed_aggregation 会将其*0.5=0.04
    double damping_factor = 0.72;       // Damped Jacobi damping 因子（amgcl 默认=0.72）

    // Spectral radius 估计（amgcl 支持，默认禁用）
    bool estimate_spectral_radius = false;  // 默认 false，与 amgcl 保持一致
    int power_iters = 0;                  // 默认 0，与 amgcl 保持一致（使用 Gershgorin 定理）

    Precision precision = Precision::Float64;
};

#endif // AMG_CONFIG_H
