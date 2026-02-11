#ifndef SOLVERS_H
#define SOLVERS_H

// ============================================================================
// 统一求解器入口 - 包含所有求解器相关的公共定义和头文件
// ============================================================================

// 公共定义(Backend, SolveStats)
#include "solve_stats.h"

// 基础类型
#include "precision_traits.h"
#include "sparse_utils.h"

// 所有求解器
#include "pcg_solver.h"
#include "amg_solver.h"  // AMG 求解器（基于 amgcl 库）

#endif // SOLVERS_H
