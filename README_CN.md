# PCG (CPU/GPU) + AMG 线性求解器

基于 C++ 和 NVIDIA CUDA 实现的高性能线性求解器，支持 PCG (CPU/GPU) 和 AMG (基于 amgcl) 两种算法，提供单/双精度计算。

## 特性

- ✅ **多种求解算法**：PCG + ILU(0)/IC(0)/AMG 预处理、AMG 独立求解器 (基于 amgcl)
- ✅ **多种预处理器**：ILU(0)、IC(0)、AMG、Jacobi
- ✅ **CPU/GPU 统一接口**：通过 `Backend` 枚举选择计算后端
- ✅ **单/双精度支持**：完整支持 float32 和 float64，类型安全，零运行时开销
- ✅ **GPU 加速**：使用 CUDA、cuSPARSE、cuBLAS
- ✅ **模块化架构**：清晰的分层设计，易于扩展
- ✅ **现代 C++**：C++17 标准，RAII 资源管理
- ✅ **完全 RAII**：智能指针自动管理内存，异常安全
- ✅ **模板化设计**：编译期多态，类型安全

## 性能

**测试环境**:
- CPU: Intel Xeon Gold 6348 @ 2.60GHz
- GPU: NVIDIA A100 80GB PCIe
- 编译器: GCC with C++17, CUDA 12.6

使用 Poisson 矩阵 (13761×13761, nnz=95065) 测试：

### 单精度 (float)

| 方法              | 迭代次数 | 最终残差  | 求解时间 |
|-------------------|----------|-----------|----------|
| PCG+ILU(0) (CPU)  | 130      | 9.63e-04  | 0.511s   |
| PCG+ILU(0) (GPU)  | 129      | 9.66e-04  | 0.032s   |
| PCG+IC(0) (CPU)   | 144      | 9.84e-04  | 0.407s   |
| PCG+IC(0) (GPU)   | 129      | 9.66e-04  | 0.026s   |
| PCG+AMG (CPU)     | 12       | 7.36e-04  | 0.148s   |
| PCG+AMG (GPU)     | 10       | 6.26e-04  | 0.011s   |
| AMG (CPU)         | 11       | 1.67e-06  | 0.144s   |

### 双精度 (double)

| 方法              | 迭代次数 | 最终残差  | 求解时间 |
|-------------------|----------|-----------|----------|
| PCG+ILU(0) (CPU)  | 129      | 9.53e-04  | 0.430s   |
| PCG+ILU(0) (GPU)  | 129      | 9.53e-04  | 0.034s   |
| PCG+IC(0) (CPU)   | 142      | 9.82e-04  | 0.405s   |
| PCG+IC(0) (GPU)   | 129      | 9.53e-04  | 0.028s   |
| PCG+AMG (CPU)     | 12       | 7.36e-04  | 0.151s   |
| PCG+AMG (GPU)     | 10       | 6.26e-04  | 0.011s   |
| AMG (CPU)         | 11       | 2.53e-09  | 0.143s   |




## 项目结构

```
linear_solver/
├── CMakeLists.txt          # 顶层构建配置
├── README.md               # 本文件
│
├── include/                # 公共头文件
│   ├── precision_traits.h  # 精度特征和类型映射
│   ├── solve_stats.h       # 求解统计和 Backend 枚举
│   ├── solvers.h           # 统一求解器入口
│   ├── pcg_solver.h        # PCG 求解器（CPU/GPU 统一接口）
│   ├── amg_solver.h        # AMG 求解器（基于 amgcl）
│   ├── amg_config.h        # AMG 配置参数
│   ├── preconditioner.h    # 预处理器（ILU/IC/AMG/Jacobi）
│   └── sparse_utils.h      # 稀疏矩阵和 CUDA 封装
│
├── src/                    # 核心库源文件
│   ├── pcg_solver.cpp      # PCG 求解器实现
│   ├── amg_solver.cpp      # AMG 求解器实现（amgcl 封装）
│   ├── preconditioner.cpp  # CPU 预处理器实现
│   ├── preconditioner_cuda.cu # GPU 预处理器实现
│   └── sparse_utils.cpp    # 稀疏矩阵和 CUDA 封装实现
│
├── examples/               # 示例程序
│   ├── main.cpp            # 综合测试程序
│   ├── CMakeLists.txt      # Examples CMake 配置
│   ├── run_test.sh         # 测试脚本（支持精度选择）
│   ├── matrix_poisson_P1_14401      # 测试矩阵
│   └── matrix_poisson_P1rhs_14401   # 测试右端项
│
└── test/                   # 综合测试
    ├── src/combined_test.cpp   # 测试程序
    ├── include/mm_reader.h     # Matrix Market 读取器
    ├── run_all_tests.sh        # 批量测试脚本
    └── matrices_florida/       # Florida 稀疏矩阵集

build/                      # 构建目录（编译后生成）
├── liblinear_solver.a      # 静态库
└── examples/
    └── main                # 可执行文件
```

## 依赖

- **CUDA** 11.8+（自动检测，推荐 12.6）
- **cuSPARSE** - 稀疏矩阵运算
- **cuBLAS** - 向量运算
- **amgcl** - AMG 求解器库
- **CMake** >= 3.20
- **C++17** 编译器（GCC 7.0+, Clang 5.0+）

## 编译

### 顶层构建（推荐）

```bash
# 在项目根目录
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 编译选项

```bash
# Release 模式（优化）
cmake -DCMAKE_BUILD_TYPE=Release ..

# 指定 CUDA 架构
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..

# 指定 CUDA 路径（如果自动检测失败）
cmake -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 ..
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
  PCG+ILU0 (CPU)           |     130 |    9.63e-04 |     0.5114 |        Yes |
  PCG+ILU0 (GPU)           |     129 |    9.66e-04 |     0.0321 |        Yes |
  PCG+IC0 (CPU)            |     144 |    9.84e-04 |     0.4068 |        Yes |
  PCG+IC0 (GPU)            |     129 |    9.66e-04 |     0.0264 |        Yes |
  PCG+AMG (CPU)            |      12 |    7.36e-04 |     0.1479 |        Yes |
  PCG+AMG (GPU)            |      10 |    6.26e-04 |     0.0106 |        Yes |
  AMG (CPU)                |      11 |    1.67e-06 |     0.1444 |        Yes |
  ------------------------------------------------------------------------------------
```

### 2. 批量测试 Florida 矩阵集

`test/` 目录包含来自 [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) 的测试矩阵：

```bash
cd test

# 运行所有矩阵测试（单精度+双精度）
./run_all_tests.sh

# 测试结果保存在 test_results_<timestamp>/ 目录
# - results.csv      CSV 格式结果
# - test_report.txt  详细测试报告
```

测试矩阵列表：

| 矩阵      | 尺寸            | 非零元   | 类型       |
|-----------|-----------------|----------|------------|
| nos4      | 100×100         | 347      | 结构问题   |
| nos1      | 237×237         | 627      | 特征值问题 |
| poisson2D | 367×367         | 2,417    | Poisson方程|
| 494_bus   | 494×494         | 1,080    | 电力网络   |
| nos3      | 960×960         | 8,402    | 特征值问题 |
| bcsstk27  | 1,224×1,224     | 28,675   | 结构力学   |
| bcsstk15  | 3,948×3,948     | 60,882   | 结构力学   |

### 3. 在其他项目中使用库

有两种方式可以在你的项目中使用 `linear_solver` 库：

---

#### 方式一：add_subdirectory

适合开发阶段，自动处理所有依赖：

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX CUDA)

# 设置 linear_solver 路径（可从命令行覆盖）
set(LINEAR_SOLVER_ROOT "/path/to/linear_solver" CACHE PATH "Path to linear_solver")

# 添加 linear_solver 为子目录（自动处理头文件和依赖）
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
add_subdirectory(${LINEAR_SOLVER_ROOT} ${CMAKE_BINARY_DIR}/linear_solver)

# 创建可执行文件
add_executable(my_app main.cpp)

# 链接 linear_solver（自动包含 CUDA 库）
target_link_libraries(my_app PRIVATE linear_solver)
```

#### 方式二：预编译库

适合部署和生产环境，编译速度快：

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX)

# 设置 linear_solver 路径（可从命令行覆盖）
set(LINEAR_SOLVER_ROOT "/path/to/linear_solver" CACHE PATH "Path to linear_solver")

# 查找预编译的静态库
find_library(LINEAR_SOLVER_LIB
    NAMES liblinear_solver.a
    PATHS
        ${LINEAR_SOLVER_ROOT}/build
        ${LINEAR_SOLVER_ROOT}/build/Release
        ${LINEAR_SOLVER_ROOT}/build/Debug
    NO_DEFAULT_PATH
)

# 创建可执行文件
add_executable(my_app main.cpp)

# 手动指定头文件
target_include_directories(my_app PRIVATE ${LINEAR_SOLVER_ROOT}/include)

# 链接预编译的静态库
target_link_libraries(my_app PRIVATE ${LINEAR_SOLVER_LIB})

# 查找并链接 CUDA 库（必须）
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_app PRIVATE CUDA::cusparse CUDA::cublas CUDA::cudart)
```


### 4. 矩阵文件格式

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
config.tolerance = 1e-12;
config.use_preconditioner = true;
config.preconditioner_type = PreconditionerType::ILU0;  // ILU0, IC0, AMG, NONE
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

AMG 求解器基于 [amgcl](https://github.com/ddemidov/amgcl) 库实现，支持 CPU 后端。

```cpp
#include "amg_solver.h"
#include "amg_config.h"
#include "precision_traits.h"

// 配置求解器
AMGConfig config;
config.max_iterations = 100;
config.tolerance = 1e-8;
config.precision = Precision::Float32;
config.use_pcg = true;               // true: PCG预处理的AMG, false: 纯V-cycle
config.coarse_grid_size = 3000;      // 最粗网格大小
config.pre_smooth_steps = 3;         // 前光滑步数
config.post_smooth_steps = 3;        // 后光滑步数
config.direct_coarse = true;         // 粗网格使用直接求解器

// 创建求解器（运行时精度选择）
AMGSolverFloat solver(config);  // 或 AMGSolverDouble

// 准备矩阵和右端项
std::vector<int> row_ptr, col_idx;
std::vector<float> values;
// ... 填充 CSR 格式矩阵数据 ...

std::vector<float> b(n, 1.0f);  // 右端项
std::vector<float> x(n, 0.0f);  // 初始解

// 求解 Ax = b
SolveStats stats = solver.solve(n, row_ptr, col_idx, values, b, x);
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

## 架构设计

### 分层架构

```
┌─────────────────────────────────────┐
│         应用层 (Application)         │
│  - examples/main.cpp                │
│  - test/combined_test.cpp           │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       求解器层 (Solver Layer)        │
│  - PCGSolver (CPU/GPU 统一接口)      │
│  - AMGSolver (基于 amgcl)           │
│  - PreconditionerBase<VectorType>   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      运算层 (Operation Layer)        │
│  - SparseMatrix (CSR 格式)          │
│  - GPUVector (RAII, 移动语义)        │
│  - CUBLASWrapper (RAII 封装)         │
│  - CUSparseWrapper (RAII 封装)       │
│  - CPUOps (namespace 简单实现)       │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      库接口层 (Library Layer)        │
│  - cuBLAS / cuSPARSE / CUDA         │
│  - amgcl (AMG 实现)                 │
│  - 标准库 (CPU 后端)                 │
└─────────────────────────────────────┘
```

## 性能优化建议

### AMG 参数调优

```cpp
AMGConfig config;

// 加速收敛（增加光滑步数）
config.pre_smooth_steps = 3;
config.post_smooth_steps = 3;

// 提高精度（减小粗网格阈值，使用直接求解器）
config.coarse_grid_size = 1000;
config.direct_coarse = true;

// 使用 PCG 预处理的 AMG（推荐）
config.use_pcg = true;

// 调整松弛因子
config.damping_factor = 0.72;  // Damped Jacobi 参数
```

## 扩展指南

### 添加新的预处理器

1. 继承 `PreconditionerBase<Precision, VectorType>` 模板基类
2. 实现 `setup()` 和 `apply()` 方法
3. 使用 `std::shared_ptr` 设置自定义预处理器

```cpp
// GPU 预处理器示例
class MyGPUPreconditioner : public PreconditionerBase<Precision::Float32, GPUVector<Precision::Float32>> {
public:
    void setup(const SparseMatrix<Precision::Float32>& A) override;
    void apply(const GPUVector<Precision::Float32>& r, GPUVector<Precision::Float32>& z) const override;
};

// 设置预处理器
PCGSolver<Precision::Float32> solver(config);
solver.set_preconditioner(std::make_shared<MyGPUPreconditioner>());
```

### 添加新的计算后端

1. 实现后端特定的向量运算
2. 使用 `Backend` 枚举添加新选项
3. 在 `solve()` 方法中添加分支