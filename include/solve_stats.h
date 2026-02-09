#ifndef SOLVE_STATS_H
#define SOLVE_STATS_H

// ============================================================================
// 后端类型
// ============================================================================

enum Backend {
    BACKEND_GPU,
    BACKEND_CPU
};

// ============================================================================
// 求解器统计信息(通用)
// ============================================================================

struct SolveStats {
    int iterations;
    double final_residual;  // 改为double以适应两种精度
    bool converged;
    double solve_time;      // 求解时间(秒)
};

#endif // SOLVE_STATS_H
