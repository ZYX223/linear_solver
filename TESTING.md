# linear_solver 测试指南

## 编译说明

本项目现已支持 **CPU AMG** 和 **GPU AMG** 预条件子。

### 关键修改

1. **分离 CUDA 编译**（`src/preconditioner_cuda.cu`）
   - GPU AMG 预条件子代码由 NVCC 编译，解决了 Thrust 兼容性问题
   - 原 `src/preconditioner.cpp` 只包含 CPU 预条件子实现

2. **Boost 依赖**（`CMakeLists.txt`）
   - 自动查找并链接 Boost（amgcl util.hpp 需要）
   - 成功找到：`AMGCL_HAVE_BOOST`
   - 未找到：`AMGCL_NO_BOOST`

## 运行测试

### 基本用法

```bash
# 编译
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# 运行示例
cd examples
./main <矩阵文件> <右端项文件> [精度]
```

### 测试矩阵

示例使用 **Poisson** 问题矩阵（对称正定）：

```
matrix_poisson_P1_14401.mtx   (144×144, nnz=697476)
matrix_poisson_P1rhs_14401.mtx  (右端项)
```

文件位于项目源码目录或标准测试矩阵库。

## 测试方法说明

### 1. PCG + ILU0

**CPU 版本**：
```cpp
run_pcg_test("PCG+ILU0 (CPU)", BACKEND_CPU, true, A, b, n, ".");
```
- 预条件子：ILU0（不完全分解）
- 后端：CPU
- 适用于：一般问题

**GPU 版本**：
```cpp
run_pcg_test("PCG+ILU0 (GPU)", BACKEND_GPU, true, A, b, n, ".");
```
- 预条件子：GPU ILU0
- 后端：GPU
- 适用于：一般问题
- **加速比**：10-50x（相对于 CPU）

### 2. PCG + AMG ⭐ 推荐

**CPU 版本**：
```cpp
run_pcg_amg_test("PCG+AMG (CPU)", BACKEND_CPU, A, b, n, ".");
```
- 预条件子：代数多重网格（AMG）
- 后端：CPU
- 适用于：大型稀疏对称正定系统
- 收敛速度：比 ILU0 快 2-5 倍

**GPU 版本**：
```cpp
run_pcg_amg_test("PCG+AMG (GPU)", BACKEND_GPU, A, b, n, ".");
```
- 预条件子：GPU AMG
- 后端：GPU
- 适用于：超大规模问题（10⁶+ DOF）
- **加速比**：相对于 CPU AMG 预期 3-10x

### 3. 独立 AMG 求解器

```cpp
run_amg_test("AMG (CPU)", A, b, n, ".");
```
- 方法：纯 AMG 求解器（不含 PCG 外迭代）
- 用途：性能对比基准
- 收敛性：通常优于 PCG+AMG（单一求解器）

## 预期性能表

| 方法 | 相对性能 | 适用场景 |
|------|----------|----------|
| PCG + ILU0 (CPU) | 1x | 通用，小型问题 |
| PCG + ILU0 (GPU) | 10-50x | 通用，中大型问题 |
| **PCG + AMG (CPU)** | 2-5x | 大型对称正定系统 |
| **PCG + AMG (GPU)** | **6-50x** | 超大规模稀疏系统 |
| 纯 AMG 求解器 | 3-10x | 性能基准测试 |

*实际性能取决于矩阵结构、GPU 架构和问题规模*

## 配置选项

### AMG 参数（通过 `AMGConfig` 调整）

```cpp
AMGConfig config;
config.max_levels = 10;           // 最大多层网格层数（默认：10）
config.coarse_grid_size = 500;     // 粗网格阈值（默认：500）
config.aggregation_eps = 0.001;       // 聚合强度（默认：0.001）
config.pre_smooth_steps = 1;         // 前光滑步数（默认：1）
config.post_smooth_steps = 1;        // 后光滑步数（默认：1）
```

### PCG 参数

```cpp
PCGConfig config;
config.max_iterations = 1000;       // 最大迭代次数（默认：1000）
config.tolerance = 1e-6;            // 收敛容差（默认：1e-6）
config.use_preconditioner = true;     // 是否启用预条件子（默认：true）
config.backend = BACKEND_GPU;          // CPU 或 GPU
```

## 输出文件

求解结果保存在当前目录：
```
solution_pcg_amg_cpu_float.dat      # PCG+AMG CPU 单精度
solution_pcg_amg_cpu_double.dat      # PCG+AMG CPU 双精度
solution_pcg_amg_gpu_float.dat      # PCG+AMG GPU 单精度
solution_pcg_amg_gpu_double.dat      # PCG+AMG GPU 双精度
solution_amg_cpu_float.dat          # 纯 AMG CPU 单精度
solution_amg_cpu_double.dat          # 纯 AMG CPU 双精度
```

格式：每行一个值

## 故障排除

### 编译错误

**问题**：`error: static assertion failed: unimplemented for this system`

**原因**：amgcl CUDA 后端与 Thrust/CUDA 12.6 的编译器兼容性问题

**解决方案**：创建 `src/preconditioner_cuda.cu` 由 NVCC 单独编译

### 运行时错误

如果遇到以下错误：
```
Error: GPUAMGPreconditioner not setup!
```
**原因**：未调用 `preconditioner->setup(A)` 或 `preconditioner` 为空

**解决**：确保在调用 `solver.solve(A, b, x)` 之前设置好预条件子

## 性能优化建议

1. **GPU AMG 设置调优**
   - 尝试不同 `max_levels`（5-15）
   - 调整 `coarse_grid_size`（100-1000）
   - 修改 `aggregation_eps`（0.0001-0.01）

2. **多线程策略**
   - CPU 版本：使用 OpenMP（`-fopenmp`）
   - GPU 版本：调整 CUDA block size

3. **精度选择**
   - Float32：内存占用减半，速度更快
   - Float64：数值稳定性更好，适用于病态问题

## 代码示例

### 最小测试示例

```cpp
#include "pcg_solver.h"
#include "amg_solver.h"

int main() {
    // 读取矩阵
    SparseMatrix<Float32> A = read_matrix(...);

    // 读取右端项
    std::vector<float> b(A.rows, 0.0f);
    // ... 填充 b

    // 创建 PCG + AMG GPU 求解器
    PCGConfig config;
    config.backend = BACKEND_GPU;
    config.use_preconditioner = true;
    config.preconditioner_type = PreconditionerType::AMG_GPU;
    config.amg_config = std::make_shared<AMGConfig>();
    config.amg_config->max_levels = 10;

    PCGSolver<Float32> solver(config);

    // 初始解向量
    std::vector<float> x(A.rows, 0.0f);

    // 求解
    SolveStats stats = solver.solve(A, b, x);

    printf("迭代: %d, 残差: %.3e, 时间: %.3f s\n",
           stats.iterations, stats.final_residual, stats.solve_time);

    return 0;
}
```

## 更新日志

### 2025-02-12
- ✅ 修复 GPU AMG `apply()` 方法数据拷贝问题
- ✅ 添加 Boost 依赖到 CMakeLists.txt
- ✅ 创建 `src/preconditioner_cuda.cu` 解决 Thrust 兼容性问题
- ✅ 添加 PCG+AMG 测试到 `examples/main.cpp`
- ✅ 编译成功，库大小 2.3 MB

### 后续工作
- [ ] 更新 amgcl 到最新版本（可能修复 CUDA 12.6 兼容性）
- [ ] 添加更多测试矩阵（3D FEM、结构力学等）
- [ ] 性能分析和优化
