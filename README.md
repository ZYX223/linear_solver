# PCG Solver v5 - CPU/GPU 统一预处理共轭梯度求解器

基于 NVIDIA CUDA 实现的高性能预处理共轭梯度（PCG）求解器，支持 CPU 和 GPU 双后端。

## 特性

- ✅ **CPU/GPU 统一接口**：通过 `Backend` 枚举选择计算后端
- ✅ **ILU(0) 预处理**：显著减少迭代次数（60-70%）
- ✅ **GPU 加速**：使用 CUDA、cuSPARSE、cuBLAS
- ✅ **模块化架构**：清晰的分层设计，易于扩展
- ✅ **标准 CMake**：完整的编译和安装系统
- ✅ **C++11 标准**：广泛的兼容性

## 性能

使用 Poisson 矩阵 (13761×13761, nnz=95065) 测试：

| 方法 | 迭代次数 | 残差 | 收敛 |
|------|---------|------|------|
| CPU (无预处理) | 426 | 9.79e-07 | Yes |
| CPU + ILU(0) | 130 | 9.27e-07 | Yes |
| GPU (无预处理) | 397 | 9.90e-07 | Yes |
| GPU + ILU(0) | 129 | 9.33e-07 | Yes |

**关键发现**：
- ILU(0) 减少迭代次数：~69%
- GPU 比 CPU 快：14-17x

## 项目结构

```
pcg_v5/
├── CMakeLists.txt          # 顶层 CMake 配置
├── README.md               # 本文件
│
├── include/                # 公共头文件
│   ├── pcg_solver.h        # PCG 求解器（CPU/GPU 统一接口）
│   ├── preconditioner.h    # 预处理器（ILU0、None）
│   └── sparse_utils.h      # 稀疏矩阵和 CUDA 封装
│
├── src/                    # 核心库源文件
│   ├── pcg_solver.cpp      # PCG 求解器实现
│   ├── preconditioner.cpp  # 预处理器实现
│   └── sparse_utils.cpp    # 稀疏矩阵和 CUDA 封装实现
│
├── examples/               # 示例程序
│   ├── main.cpp            # 综合测试程序
│   ├── CMakeLists.txt      # examples CMake 配置
│   ├── run_test.sh         # 测试脚本
│   ├── README.md           # 示例说明
│   ├── matrix_poisson_P1_14401      # 测试矩阵
│   └── matrix_poisson_P1rhs_14401   # 测试右端项
│
├── tests/                  # 单元测试（待实现）
│   └── CMakeLists.txt
│
└── cmake/                  # CMake 模块
    └── pcg_v5Config.cmake.in
```

## 依赖

- **CUDA** 12.6（其他版本可能需要调整）
- **cuSPARSE** - 稀疏矩阵运算
- **cuBLAS** - 向量运算
- **CMake** >= 3.20
- **C++11** 编译器（GCC 5.0+, Clang 3.3+）

## 编译

### 基本编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 编译选项

```bash
# 不编译示例程序
cmake -DBUILD_EXAMPLES=OFF ..

# 编译测试（如果实现了）
cmake -DBUILD_TESTS=ON ..

# Release 模式（优化）
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 安装

```bash
sudo make install
# 安装到 /usr/local/lib, /usr/local/bin, /usr/local/include
```

## 使用

### 1. 示例程序（读取矩阵文件）

```bash
cd examples
./build/main matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401
```

或使用测试脚本：

```bash
cd examples
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401
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
  -----------------------------------------------------------------------------------
  Method                    | Iters  | Residual   | Converged |
  -----------------------------------------------------------------------------------
  CPU (无预处理)        | 426    | 9.786829e-07 | Yes       |
  CPU + ILU(0)              | 130    | 9.266e-07  | Yes       |
  GPU (无预处理)        | 397    | 9.897e-07  | Yes       |
  GPU + ILU(0)              | 129    | 9.333e-07  | Yes       |
  -----------------------------------------------------------------------------------
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

// 设置 ILU(0) 预处理器（仅 GPU）
if (config.backend == BACKEND_GPU) {
    auto sparse = std::make_shared<CUSparseWrapper>();
    auto ilu_prec = std::make_shared<ILUPreconditioner>(sparse);
    solver.set_preconditioner(ilu_prec);
}

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
```

### 作为库使用

在其他项目的 CMakeLists.txt 中：

```cmake
find_package(pcg_v5 REQUIRED)

target_link_libraries(your_target PRIVATE pcg_v5_lib)
```

## 算法说明

### 预处理共轭梯度法（PCG）

```
给定：矩阵 A，右端项 b，初始解 x=0
1. r = b - A*x
2. while ||r|| > tol:
3.     z = M^(-1) * r        # 预处理步骤
4.     if k == 1:
5.         p = z
6.     else:
7.         β = (r•z) / (r_old•z_old)
8.         p = z + β*p
9.     α = (r•z) / (p•A*p)
10.    x = x + α*p
11.    r = r - α*A*p
```

### ILU(0) 预处理

- **不完全 LU 分解**：保持 L 和 U 的稀疏结构与原矩阵相同
- **零填充**：不添加新的非零元素
- **三角求解**：前向替换（L）+ 后向替换（U）

## 架构设计

```
┌─────────────────────────────────────┐
│         应用层 (Application)         │
│  - examples/main.cpp                 │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       求解器层 (Solver Layer)        │
│  - PCGSolver (CPU/GPU 统一接口)     │
│  - Preconditioner (接口)             │
│    - ILUPreconditioner (GPU ILU0)   │
│    - NonePreconditioner             │
│    - CPUIILUPreconditioner (CPU)    │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      运算层 (Operation Layer)        │
│  - SparseMatrix (CSR 格式)          │
│  - GPUVector                        │
│  - CUBLASWrapper (BLAS 运算)        │
│  - CUSparseWrapper (稀疏运算)       │
│  - CPUOps (CPU 向量运算)            │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│      库接口层 (Library Layer)        │
│  - cuBLAS / cuSPARSE / CUDA         │
│  - 标准库 (CPU 后端)                 │
└─────────────────────────────────────┘
```

## 扩展指南

### 添加新的预处理器

1. 在 `include/preconditioner.h` 中添加类声明
2. 在 `src/preconditioner.cpp` 中实现
3. 在用户代码中设置

```cpp
class MyPreconditioner : public Preconditioner {
public:
    void setup(const SparseMatrix& A) override;
    void apply(const GPUVector& r, GPUVector& z) const override;
};
```

### 添加新的示例

1. 在 `examples/` 中创建新的 `.cpp` 文件
2. 在 `examples/CMakeLists.txt` 中添加：
   ```cmake
   add_executable(my_example my_example.cpp)
   target_link_libraries(my_example PRIVATE pcg_v5_lib)
   ```

### 添加单元测试

1. 在 `tests/` 中创建测试文件
2. 在 `tests/CMakeLists.txt` 中添加：
   ```cmake
   add_executable(test_xxx test_xxx.cpp)
   target_link_libraries(test_xxx PRIVATE pcg_v5_lib)
   add_test(NAME test_xxx COMMAND test_xxx)
   ```


