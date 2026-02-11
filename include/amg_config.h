#ifndef AMG_CONFIG_H
#define AMG_CONFIG_H

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

    // AMG特定参数（与amgcl完全一致）
    int max_levels = 10;                // 最大层数
    int coarse_grid_size = 500;         // 最粗网格大小阈值（与 amgcl 默认值一致）
    int pre_smooth_steps = 1;           // 前光滑步数（amgcl: npre=1）
    int post_smooth_steps = 1;          // 后光滑步数（amgcl: npost=1）
    double aggregation_eps = 0.08;      // 聚合阈值（amgcl: eps_strong=0.08）

    // Spectral radius 估计（新增）
    bool estimate_spectral_radius = false;  // 默认关闭（测试发现可能不总是有益）
    int power_iters = 10;                  // Power iteration 次数

    Precision precision = Precision::Float32;
};

#endif // AMG_CONFIG_H
