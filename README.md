# PCG (CPU/GPU) + AMG (CPU) 线性求解器

基于 C++ 和 NVIDIA CUDA 实现的高性能线性求解器,支持 PCG(CPU/GPU)和 AMG(CPU)两种算法,提供单/双精度计算。

## 特性

- ✅ **多种求解算法**：PCG + ILU(0)、AMG (Smoothed Aggregation)
- ✅ **CPU/GPU 统一接口**：通过 `Backend` 枚举选择计算后端
- ✅ **单/双精度支持**：完整支持 float32 和 float64，类型安全，零运行时开销
- ✅ **GPU 加速**：使用 CUDA、cuSPARSE、cuBLAS
- ✅ **模块化架构**：清晰的分层设计，易于扩展
- ✅ **现代 C++**：C++14 标准，RAII 资源管理
- ✅ **完全 RAII**：智能指针自动管理内存，异常安全
- ✅ **模板化设计**：编译期多态，类型安全

## 性能

使用 Poisson 矩阵 (13761×13761, nnz=95065) 测试：

### 单精度

| 方法 | 迭代次数 | 最终残差 | 求解时间 | 加速比 |
|------|---------|-------------|----------|--------|
| PCG (CPU) | 130 | 9.27e-07 | 0.446s | 1.0x |
| PCG (GPU) | 129 | 9.33e-07 | 0.031s | 14.4x |
| AMG (CPU) | 19 | 9.27e-07 | 0.175s | 2.5x |

### 双精度

| 方法 | 迭代次数 | 最终残差 | 求解时间 | 加速比 |
|------|---------|-------------|----------|--------|
| PCG (CPU) | 129 | 9.08e-07 | 0.427s | 1.0x |
| PCG (GPU) | 129 | 9.08e-07 | 0.033s | 12.9x |
| AMG (CPU) | 17 | 8.77e-07 | 0.151s | 2.8x |



## 项目结构

```
pcg_solver/
├── README.md               # 本文件
│
├── include/                # 公共头文件
│   ├── precision_traits.h  # 精度特征和类型映射
│   ├── pcg_solver.h        # PCG 求解器（CPU/GPU 统一接口）
│   ├── amg_solver.h        # AMG 求解器
│   ├── amg_hierarchy.h     # AMG 层次结构
│   ├── amg_coarsening.h    # AMG 粗化策略
│   ├── amg_relaxation.h    # AMG 松弛算子
│   ├── amg_interpolation.h # AMG 插值算子
│   ├── preconditioner.h    # 预处理器（模板基类 + 实现）
│   └── sparse_utils.h      # 稀疏矩阵和 CUDA 封装
│
├── src/                    # 核心库源文件
│   ├── pcg_solver.cpp      # PCG 求解器实现
│   ├── amg_solver.cpp      # AMG 求解器实现
│   ├── amg_hierarchy.cpp   # AMG 层次结构实现
│   ├── amg_coarsening.cpp  # AMG 粗化实现
│   ├── amg_relaxation.cpp  # AMG 松弛实现
│   ├── amg_interpolation.cpp # AMG 插值实现
│   ├── preconditioner.cpp  # 预处理器实现
│   └── sparse_utils.cpp    # 稀疏矩阵和 CUDA 封装实现
│
└── examples/               # 示例程序
    ├── main.cpp            # 综合测试程序
    ├── CMakeLists.txt      # CMake 配置
    ├── run_test.sh         # 测试脚本（支持精度选择）
    ├── matrix_poisson_P1_14401      # 测试矩阵
    └── matrix_poisson_P1rhs_14401   # 测试右端项
```

## 依赖

- **CUDA** 12.6（其他版本可能需要调整 `CMakeLists.txt` 中的路径）
- **cuSPARSE** - 稀疏矩阵运算
- **cuBLAS** - 向量运算
- **CMake** >= 3.20
- **C++14** 编译器（GCC 5.0+, Clang 3.3+）

## 编译

### 基本编译

```bash
cd examples
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 编译选项

```bash
# Release 模式（优化）
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## 使用

### 1. 快速测试

```bash
cd examples

# 测试两种精度（默认）
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401

# 仅测试单精度
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 float

# 仅测试双精度
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 double
```

### 2. 直接运行

```bash
cd examples/build

# 单精度测试
./main ../matrix_poisson_P1_14401 ../matrix_poisson_P1rhs_14401 float

# 双精度测试
./main ../matrix_poisson_P1_14401 ../matrix_poisson_P1rhs_14401 double
```

输出示例：

```
========================================
 求解器综合测试
========================================

[1/3] 读取矩阵文件: matrix_poisson_P1_14401
[2/3] 读取右端项文件: matrix_poisson_P1rhs_14401
[3/3] 运行测试...

问题规模: 13761 × 13761, nnz = 95065
右端项大小: 13761
  ------------------------------------------------------------------------------------
  Method                   |   Iters |    Residual |   Time (s) |  Converged |
  ------------------------------------------------------------------------------------
  PCG (CPU)                |     129 |    9.08e-07 |     0.427 |        Yes |
  PCG (GPU)                |     129 |    9.08e-07 |     0.033 |        Yes |
  AMG (CPU)                |      17 |    8.77e-07 |     0.151 |        Yes |
  ------------------------------------------------------------------------------------
```

### 3. 矩阵文件格式

**矩阵文件**：
```
行数 列数 非零元素数
行索引 列索引 值
...
```

**右端项文件**：
```
向量大小
值1
值2
...
```

## C++ API

### PCG 求解器

```cpp
#include "pcg_solver.h"
#include "preconditioner.h"
#include "precision_traits.h"

// 配置求解器
PCGConfig config;
config.max_iterations = 1000;
config.tolerance = 1e-6;
config.use_preconditioner = true;
config.backend = BACKEND_GPU;
config.precision = Precision::Float32;

// 创建求解器
PCGSolver<Precision::Float32> solver(config);

// 准备矩阵和右端项
SparseMatrix<Precision::Float32> A(rows, cols, nnz);
// ... 填充矩阵数据 ...
A.upload_to_gpu();

std::vector<float> b(n, 1.0f);  // 右端项
std::vector<float> x(n, 0.0f);  // 初始解

// 求解 Ax = b
SolveStats stats = solver.solve(A, b, x);
```

### AMG 求解器

```cpp
#include "amg_solver.h"
#include "precision_traits.h"

// 配置求解器
AMGConfig config;
config.max_iterations = 100;
config.tolerance = 1e-6;
config.precision = Precision::Float32;
config.max_levels = 10;           // 最大层数
config.coarse_grid_size = 50;     // 最粗网格大小
config.pre_smooth_steps = 1;      // 前光滑步数
config.post_smooth_steps = 1;     // 后光滑步数

// 创建求解器
AMGSolver<Precision::Float32> solver(config);

// 准备矩阵和右端项
SparseMatrix<Precision::Float32> A(rows, cols, nnz);
// ... 填充矩阵数据 ...

std::vector<float> b(n, 1.0f);  // 右端项
std::vector<float> x(n, 0.0f);  // 初始解

// 求解 Ax = b
SolveStats stats = solver.solve(A, b, x);
```

### 使用类型别名

```cpp
// PCG 类型别名
using PCGSolverFloat = PCGSolver<Precision::Float32>;
using SparseMatrixFloat = SparseMatrix<Precision::Float32>;

PCGSolverFloat solver(config);
SparseMatrixFloat A(rows, cols, nnz);

// AMG 类型别名
using AMGSolverFloat = AMGSolver<Precision::Float32>;

AMGSolverFloat solver(config);
```

## 算法说明

### 1. 预处理共轭梯度法（PCG）

```
给定：矩阵 A，右端项 b，初始解 x=0
1. r = b - A*x
2. while ||r|| > tol:
    z = M^(-1) * r        # 预处理步骤
    if k == 1:
        p = z
    else:
        β = (r•z) / (r_old•z_old)
        p = z + β*p
    α = (r•z) / (p•A*p)
    x = x + α*p
    r = r - α*A*p
```

### 2. ILU(0) 预处理

- **不完全 LU 分解**：保持 L 和 U 的稀疏结构与原矩阵相同
- **零填充**：不添加新的非零元素
- **三角求解**：前向替换（L）+ 后向替换（U）

### 3. 代数多重网格（AMG）

**核心思想**：使用多层网格加速收敛

```
V-Cycle 算法：
1. 前光滑 (Pre-smoothing): Gauss-Seidel
2. 残差计算: r = b - A*x
3. 限制到粗网格: r_c = R * r
4. 粗网格求解: A_c * e_c = r_c (递归 V-Cycle)
5. 插值回细网格: corr = P * e_c
6. 校正解: x = x + corr
7. 后光滑 (Post-smoothing): Gauss-Seidel
```

**关键组件**：
- **Smoothed Aggregation 粗化**：基于强连接构建聚合
- **插值平滑**：P = (I - ωD^(-1)A) * P_const
- **Galerkin 乘积**：A_coarse = R * A * P
- **最粗网格求解**：Gauss-Seidel 迭代

## 架构设计

### 分层架构

```
┌─────────────────────────────────────┐
│         应用层 (Application)         │
│  - examples/main.cpp                │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       求解器层 (Solver Layer)        │
│  - PCGSolver (CPU/GPU 统一接口)      │
│  - AMGSolver (CPU, 多层次结构)       │
│  - PreconditionerBase<VectorType>    │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      运算层 (Operation Layer)        │
│  - SparseMatrix (CSR 格式)          │
│  - GPUVector (RAII, 移动语义)       │
│  - CUBLASWrapper (RAII 封装)         │
│  - CUSparseWrapper (RAII 封装)        │
│  - CPUOps (namespace 简单实现)       │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      库接口层 (Library Layer)        │
│  - cuBLAS / cuSPARSE / CUDA         │
│  - 标准库 (CPU 后端)                 │
└─────────────────────────────────────┘
```

## 性能优化建议

### 选择合适的求解器

| 场景 | 推荐求解器 | 原因 |
|------|-----------|------|
| **大规模问题** | AMG (CPU) | 迭代次数少，收敛快 |
| **有 GPU 可用** | PCG (GPU) | 利用 GPU 并行加速 |
| **内存受限** | PCG (CPU+ILU) | AMG 需要存储层次结构 |
| **高质量解** | AMG (CPU) | 收敛更稳定 |

### AMG 参数调优

```cpp
AMGConfig config;

// 加速收敛（增加迭代步数）
config.pre_smooth_steps = 2;
config.post_smooth_steps = 2;

// 提高精度（减小粗网格阈值）
config.coarse_grid_size = 30;

// 处理更复杂的问题（增加层数）
config.max_levels = 15;
```

## 扩展指南

### 添加新的预处理器

1. 继承 `PreconditionerBase<VectorType>` 模板基类
2. 实现 `setup()` 和 `apply()` 方法
3. 使用 `std::shared_ptr` 设置自定义预处理器

```cpp
// GPU 预处理器示例
class MyGPUPreconditioner : public PreconditionerBase<GPUVector> {
public:
    void setup(const SparseMatrix& A) override;
    void apply(const GPUVector& r, GPUVector& z) const override;
};

// 设置预处理器
PCGSolver solver(config);
solver.set_preconditioner(std::make_shared<MyGPUPreconditioner>());
```

### 添加新的计算后端

1. 实现后端特定的向量运算
2. 使用 `Backend` 枚举添加新选项
3. 在 `solve()` 方法中添加分支


## 参考文献

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*
2. Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial*
3. Stuben, K. (2001). Algebraic multigrid (AMG): An introduction with applications
