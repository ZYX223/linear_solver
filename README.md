# PCG Solver - CPU/GPU 预处理共轭梯度求解器

基于 NVIDIA CUDA 实现的预处理共轭梯度（PCG）求解器，支持 CPU 和 GPU 后端。

## 特性

- ✅ **CPU/GPU 统一接口**：通过 `Backend` 枚举选择计算后端
- ✅ **ILU(0) 预处理**：显著减少迭代次数（~69%）
- ✅ **GPU 加速**：使用 CUDA、cuSPARSE、cuBLAS
- ✅ **模块化架构**：清晰的分层设计，易于扩展
- ✅ **现代 C++**：C++14 标准，RAII 资源管理
- ✅ **完全 RAII**：智能指针自动管理内存，异常安全

## 性能

使用 Poisson 矩阵 (13761×13761, nnz=95065) 测试：

```
  ------------------------------------------------------------------------------------
  Method                   |   Iters |    Residual |   Time (s) |  Converged |
  ------------------------------------------------------------------------------------
  CPU (无预处理)           |     426 |    9.79e-07 |     0.9088 |        Yes |
  CPU + ILU(0)             |     130 |    9.27e-07 |     0.4282 |        Yes |
  GPU (无预处理)           |     397 |    9.90e-07 |     0.0588 |        Yes |
  GPU + ILU(0)             |     129 |    9.33e-07 |     0.0258 |        Yes |
  ------------------------------------------------------------------------------------
```

**性能提升**：
- ILU(0) 减少迭代次数：~69%
- GPU 加速比：15-18x（vs 无预处理 CPU）
- GPU 加速比：10x（vs ILU0 CPU）

## 项目结构

```
pcg_solver/
├── README.md               # 本文件
│
├── include/                # 公共头文件
│   ├── pcg_solver.h        # PCG 求解器（CPU/GPU 统一接口）
│   ├── preconditioner.h    # 预处理器（模板基类 + 实现）
│   └── sparse_utils.h      # 稀疏矩阵和 CUDA 封装
│
├── src/                    # 核心库源文件
│   ├── pcg_solver.cpp      # PCG 求解器实现
│   ├── preconditioner.cpp  # 预处理器实现
│   └── sparse_utils.cpp    # 稀疏矩阵和 CUDA 封装实现
│
└── examples/               # 示例程序
    ├── main.cpp            # 综合测试程序
    ├── CMakeLists.txt      # CMake 配置
    ├── run_test.sh         # 测试脚本
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
# 不编译示例程序
cmake -DBUILD_EXAMPLES=OFF ..

# Release 模式（优化）
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## 使用

### 1. 示例程序（读取矩阵文件）

```bash
cd examples
./build/main matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401
```

输出示例：

```
========================================
  PCG 求解器综合测试
========================================

[1/3] 读取矩阵文件: matrix_poisson_P1_14401
  矩阵信息: 13761 x 13761, nnz = 95065

[2/3] 读取右端项文件: matrix_poisson_P1rhs_14401
  右端项大小: 13761

[3/3] 运行测试...
  ------------------------------------------------------------------------------------
  Method                   |   Iters |    Residual |   Time (s) |  Converged |
  ------------------------------------------------------------------------------------
  CPU (无预处理)           |     426 |    9.79e-07 |     0.9088 |        Yes |
  CPU + ILU(0)             |     130 |    9.27e-07 |     0.4282 |        Yes |
  GPU (无预处理)           |     397 |    9.90e-07 |     0.0588 |        Yes |
  GPU + ILU(0)             |     129 |    9.33e-07 |     0.0258 |        Yes |
  ------------------------------------------------------------------------------------
```

### 2. 矩阵文件格式

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

### 基本使用

```cpp
#include "pcg_solver.h"
#include "preconditioner.h"

// 配置求解器
PCGConfig config;
config.max_iterations = 1000;
config.tolerance = 1e-6f;
config.use_preconditioner = true;
config.backend = BACKEND_GPU;  // 或 BACKEND_CPU

// 创建求解器
PCGSolver solver(config);

// 准备矩阵和右端项
SparseMatrix A(rows, cols, nnz);
// ... 填充矩阵数据 ...
A.upload_to_gpu();

std::vector<float> b(n, 1.0f);  // 右端项
std::vector<float> x(n, 0.0f);  // 初始解

// 求解 Ax = b
SolveStats stats = solver.solve(A, b, x);

// 结果
std::cout << "Iterations: " << stats.iterations << std::endl;
std::cout << "Final residual: " << stats.final_residual << std::endl;
std::cout << "Converged: " << (stats.converged ? "Yes" : "No") << std::endl;
std::cout << "Time: " << stats.solve_time << " s" << std::endl;
```

### 高级用法：自定义预处理器

```cpp
#include "preconditioner.h"

// 1. 继承预处理器基类
template<typename VectorType>
class MyPreconditioner : public PreconditionerBase<VectorType> {
public:
    void setup(const SparseMatrix& A) override {
        // 初始化预处理器
    }

    void apply(const VectorType& r, VectorType& z) const override {
        // 实现 z = M^(-1) * r
    }
};

// 2. GPU 预处理器
class MyGPUPreconditioner : public PreconditionerBase<GPUVector> {
public:
    void setup(const SparseMatrix& A) override;
    void apply(const GPUVector& r, GPUVector& z) const override;
};

// 3. 设置自定义预处理器
PCGSolver solver(config);
auto my_prec = std::make_shared<MyGPUPreconditioner>();
solver.set_preconditioner(my_prec);
```

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
│  - PreconditionerBase<VectorType>    │
│    ├── GPUILUPreconditioner          │
│    └── CPUILUPreconditioner          │
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

## 算法说明

### 预处理共轭梯度法（PCG）

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

### ILU(0) 预处理

- **不完全 LU 分解**：保持 L 和 U 的稀疏结构与原矩阵相同
- **零填充**：不添加新的非零元素
- **三角求解**：前向替换（L）+ 后向替换（U）

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
