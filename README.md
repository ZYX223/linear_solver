# PCG (CPU/GPU) + AMG Linear Solver

A high-performance linear solver implemented in C++ and NVIDIA CUDA, supporting PCG (CPU/GPU) and AMG (based on amgcl) algorithms with single/double precision.

## Features

- Multiple solver algorithms: PCG + ILU(0)/IC(0)/AMG preconditioning, standalone AMG solver (based on amgcl)
- Multiple preconditioners: ILU(0), IC(0), AMG, Jacobi
- Unified CPU/GPU interface via `Backend` enum
- Full float32 and float64 support with type safety and zero runtime overhead
- GPU acceleration using CUDA, cuSPARSE, cuBLAS
- Modular architecture with clear layered design
- Modern C++17 with RAII resource management
- Template-based design with compile-time polymorphism

## Performance

**Test Environment**:
- CPU: Intel Xeon Gold 6348 @ 2.60GHz
- GPU: NVIDIA A100 80GB PCIe
- Compiler: GCC with C++17, CUDA 12.6

Tested with Poisson matrix (13761x13761, nnz=95065):

### Single Precision (tolerance=1e-6)

| Method             | Iterations | Final Residual | Time   |
|--------------------|------------|----------------|--------|
| PCG+ILU0 (CPU)     | 112        | 9.30e-07       | 0.478s |
| PCG+ILU0 (GPU)     | 112        | 9.17e-07       | 0.028s |
| PCG+IC0 (CPU)      | 123        | 9.24e-07       | 0.348s |
| PCG+IC0 (GPU)      | 123        | 9.22e-07       | 0.025s |
| PCG+AMG (CPU)      | 7          | 1.19e-07       | 0.099s |
| PCG+AMG (GPU)      | 5          | 6.09e-07       | 0.028s |
| AMG (CPU)          | 4          | 4.86e-07       | 0.108s |

### Double Precision (tolerance=1e-12)

| Method             | Iterations | Final Residual | Time   |
|--------------------|------------|----------------|--------|
| PCG+ILU0 (CPU)     | 194        | 9.82e-13       | 0.809s |
| PCG+ILU0 (GPU)     | 194        | 9.76e-13       | 0.046s |
| PCG+IC0 (CPU)      | 217        | 8.44e-13       | 0.618s |
| PCG+IC0 (GPU)      | 217        | 8.44e-13       | 0.045s |
| PCG+AMG (CPU)      | 13         | 3.90e-13       | 0.186s |
| PCG+AMG (GPU)      | 11         | 3.12e-13       | 0.062s |
| AMG (CPU)          | 9          | 4.70e-14       | 0.242s |

## Project Structure

```
linear_solver/
├── CMakeLists.txt          # Top-level build configuration
├── README.md               # This file
│
├── include/                # Public headers
│   ├── precision_traits.h  # Precision traits and type mapping
│   ├── solve_stats.h       # Solve statistics and Backend enum
│   ├── solvers.h           # Unified solver entry point
│   ├── pcg_solver.h        # PCG solver (CPU/GPU unified interface)
│   ├── amg_solver.h        # AMG solver (based on amgcl)
│   ├── amg_config.h        # AMG configuration parameters
│   ├── preconditioner.h    # Preconditioners (ILU/IC/AMG/Jacobi)
│   └── sparse_utils.h      # Sparse matrix and CUDA wrappers
│
├── src/                    # Core library source files
│   ├── pcg_solver.cpp      # PCG solver implementation
│   ├── amg_solver.cpp      # AMG solver implementation (amgcl wrapper)
│   ├── preconditioner.cpp  # CPU preconditioner implementations
│   ├── preconditioner_cuda.cu # GPU preconditioner implementations
│   └── sparse_utils.cpp    # Sparse matrix and CUDA wrapper implementations
│
├── examples/               # Example programs
│   ├── main.cpp            # Comprehensive test program
│   ├── CMakeLists.txt      # Examples CMake configuration
│   ├── run_test.sh         # Test script (supports precision selection)
│   ├── matrix_poisson_P1_14401      # Test matrix
│   └── matrix_poisson_P1rhs_14401   # Test RHS vector
│
└── test/                   # Comprehensive tests
    ├── src/combined_test.cpp   # Test program
    ├── include/mm_reader.h     # Matrix Market reader
    ├── run_all_tests.sh        # Batch test script
    └── matrices_florida/       # Florida sparse matrix collection

build/                      # Build directory (generated after compilation)
├── liblinear_solver.a      # Static library
└── examples/
    └── main                # Executable
```

## Dependencies

- **CUDA** 11.8+ (auto-detected, 12.6 recommended)
- **cuSPARSE** - Sparse matrix operations
- **cuBLAS** - Vector operations
- **amgcl** - AMG solver library (embedded in project)
- **CMake** >= 3.20
- **C++17** compiler (GCC 7.0+, Clang 5.0+)

## Build

### Top-level Build (Recommended)

```bash
# In project root directory
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

```bash
# Release mode (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Specify CUDA architecture
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..

# Specify CUDA path (if auto-detection fails)
cmake -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 ..
```

## Usage

### 1. Quick Test

```bash
cd examples

# Test both precisions (default)
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401

# Test single precision only
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 float

# Test double precision only
./run_test.sh matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 double
```

Sample output:

```
========================================
 Solver Comprehensive Test
========================================

[1/3] Reading matrix file: matrix_poisson_P1_14401
[2/3] Reading RHS file: matrix_poisson_P1rhs_14401
[3/3] Running tests...

Problem size: 13761 × 13761, nnz = 95065
RHS size: 13761
  ------------------------------------------------------------------------------------
  Method                   |   Iters |    Residual |   Time (s) |  Converged |
  ------------------------------------------------------------------------------------
  PCG+ILU0 (CPU)           |     112 |    9.30e-07 |     0.4781 |        Yes |
  PCG+ILU0 (GPU)           |     112 |    9.17e-07 |     0.0283 |        Yes |
  PCG+IC0 (CPU)            |     123 |    9.24e-07 |     0.3480 |        Yes |
  PCG+IC0 (GPU)            |     123 |    9.22e-07 |     0.0248 |        Yes |
  PCG+AMG (CPU)            |       7 |    1.19e-07 |     0.0990 |        Yes |
  PCG+AMG (GPU)            |       5 |    6.09e-07 |     0.0282 |        Yes |
  AMG (CPU)                |       4 |    4.86e-07 |     0.1077 |        Yes |
  ------------------------------------------------------------------------------------
```

### 2. Batch Testing with Florida Matrix Collection

The `test/` directory contains test matrices from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/):

```bash
cd test

# Run all matrix tests (single + double precision)
./run_all_tests.sh

# Results saved in test_results_<timestamp>/ directory
# - results.csv      CSV format results
# - test_report.txt  Detailed test report
```

Test matrices:

| Matrix    | Size            | NNZ     | Type           |
|-----------|-----------------|---------|----------------|
| nos4      | 100x100         | 347     | Structural     |
| nos1      | 237x237         | 627     | Eigenvalue     |
| poisson2D | 367x367         | 2,417   | Poisson PDE    |
| 494_bus   | 494x494         | 1,080   | Power network  |
| nos3      | 960x960         | 8,402   | Eigenvalue     |
| bcsstk27  | 1,224x1,224     | 28,675  | Structural     |
| bcsstk15  | 3,948x3,948     | 60,882  | Structural     |

### 3. Using the Library in Other Projects

#### Method 1: add_subdirectory

Suitable for development phase, handles all dependencies automatically:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX CUDA)

# Set linear_solver path (can be overridden from command line)
set(LINEAR_SOLVER_ROOT "/path/to/linear_solver" CACHE PATH "Path to linear_solver")

# Add linear_solver as subdirectory (handles headers and dependencies)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
add_subdirectory(${LINEAR_SOLVER_ROOT} ${CMAKE_BINARY_DIR}/linear_solver)

# Create executable
add_executable(my_app main.cpp)

# Link linear_solver (CUDA libraries included automatically)
target_link_libraries(my_app PRIVATE linear_solver)
```

#### Method 2: Precompiled Library

Suitable for deployment and production:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_project LANGUAGES CXX)

# Set linear_solver path
set(LINEAR_SOLVER_ROOT "/path/to/linear_solver" CACHE PATH "Path to linear_solver")

# Find precompiled static library
find_library(LINEAR_SOLVER_LIB
    NAMES liblinear_solver.a
    PATHS
        ${LINEAR_SOLVER_ROOT}/build
        ${LINEAR_SOLVER_ROOT}/build/Release
        ${LINEAR_SOLVER_ROOT}/build/Debug
    NO_DEFAULT_PATH
)

# Create executable
add_executable(my_app main.cpp)

# Specify include directories
target_include_directories(my_app PRIVATE ${LINEAR_SOLVER_ROOT}/include)

# Link precompiled library
target_link_libraries(my_app PRIVATE ${LINEAR_SOLVER_LIB})

# Find and link CUDA libraries (required)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_app PRIVATE CUDA::cusparse CUDA::cublas CUDA::cudart)
```

### 4. Matrix File Format

**Matrix file**:
```
rows cols nnz
row_index col_index value
...
```

**RHS file**:
```
vector_size
value1
value2
...
```

## C++ API

### PCG Solver

```cpp
#include "pcg_solver.h"
#include "preconditioner.h"
#include "precision_traits.h"

// Configure solver
PCGConfig config;
config.max_iterations = 1000;
config.tolerance = 1e-8;
config.use_preconditioner = true;
config.preconditioner_type = PreconditionerType::ILU0;  // ILU0, IC0, AMG, NONE
config.backend = BACKEND_GPU;
config.precision = Precision::Float64;

// Create solver
PCGSolver<Precision::Float64> solver(config);

// Prepare matrix and RHS
SparseMatrix<Precision::Float64> A(rows, cols, nnz);
// ... fill matrix data ...
A.upload_to_gpu();

std::vector<double> b(n, 1.0f);  // RHS
std::vector<double> x(n, 0.0f);  // Initial solution

// Solve Ax = b
SolveStats stats = solver.solve(A, b, x);
```

### AMG Solver

The AMG solver is based on the [amgcl](https://github.com/ddemidov/amgcl) library.

```cpp
#include "amg_solver.h"
#include "amg_config.h"
#include "precision_traits.h"

// Configure solver
AMGConfig config;
config.max_iterations = 1000;
config.tolerance = 1e-8;
config.precision = Precision::Float64;

// Create solver (runtime precision selection)
AMGSolverFloat solver(config);  // or AMGSolverDouble

// Prepare matrix and RHS
SparseMatrix<Precision::Float64> A(rows, cols, nnz);

std::vector<double> b(n, 1.0f);  // RHS
std::vector<double> x(n, 0.0f);  // Initial solution

// Solve Ax = b
SolveStats stats = solver.solve(A, b, x);
```

### Type Aliases

```cpp
// PCG type aliases
using PCGSolverFloat = PCGSolver<Precision::Float32>;
using SparseMatrixFloat = SparseMatrix<Precision::Float32>;

PCGSolverFloat solver(config);
SparseMatrixFloat A(rows, cols, nnz);

// AMG type aliases
using AMGSolverFloat = AMGSolver<Precision::Float32>;

AMGSolverFloat solver(config);
```

## Architecture

### Layered Architecture

```
+-------------------------------------+
|         Application Layer           |
|  - examples/main.cpp                |
|  - test/combined_test.cpp           |
+-------------------------------------+
                  |
+-------------------------------------+
|         Solver Layer                |
|  - PCGSolver (CPU/GPU unified)      |
|  - AMGSolver (based on amgcl)       |
|  - PreconditionerBase<VectorType>   |
+-------------------------------------+
                  |
+-------------------------------------+
|        Operation Layer              |
|  - SparseMatrix (CSR format)        |
|  - GPUVector (RAII, move semantics) |
|  - CUBLASWrapper (RAII wrapper)     |
|  - CUSparseWrapper (RAII wrapper)   |
|  - CPUOps (namespace implementation)|
+-------------------------------------+
                  |
+-------------------------------------+
|        Library Layer                |
|  - cuBLAS / cuSPARSE / CUDA         |
|  - amgcl (AMG implementation)       |
|  - STL (CPU backend)                |
+-------------------------------------+
```

## Performance Tuning

### AMG Parameter Tuning

```cpp
AMGConfig config;

// Accelerate convergence (increase smoothing steps)
config.pre_smooth_steps = 3;
config.post_smooth_steps = 3;

// Improve accuracy (reduce coarse grid threshold, use direct solver)
config.coarse_grid_size = 1000;
config.direct_coarse = true;

// Use PCG-preconditioned AMG (recommended)
config.use_pcg = true;

// Adjust damping factor
config.damping_factor = 0.72;  // Damped Jacobi parameter
```

## Extension Guide

### Adding New Preconditioners

1. Inherit from `PreconditionerBase<Precision, VectorType>` template base class
2. Implement `setup()` and `apply()` methods
3. Set custom preconditioner using `std::shared_ptr`

```cpp
// GPU preconditioner example
class MyGPUPreconditioner : public PreconditionerBase<Precision::Float32, GPUVector<Precision::Float32>> {
public:
    void setup(const SparseMatrix<Precision::Float32>& A) override;
    void apply(const GPUVector<Precision::Float32>& r, GPUVector<Precision::Float32>& z) const override;
};

// Set preconditioner
PCGSolver<Precision::Float32> solver(config);
solver.set_preconditioner(std::make_shared<MyGPUPreconditioner>());
```

### Adding New Compute Backends

1. Implement backend-specific vector operations
2. Add new option using `Backend` enum
3. Add branch in `solve()` method
