cmake_minimum_required(VERSION 3.20)

# CUDA 路径（可选，用于提示 CMake 查找 CUDA）
if(DEFINED ENV{CUDA_PATH} AND NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER $ENV{CUDA_PATH}/bin/nvcc CACHE FILEPATH "CUDA compiler")
endif()

# 尝试常见的 CUDA 安装路径
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    foreach(CUDA_VER 12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8)
        if(EXISTS /usr/local/cuda-${CUDA_VER}/bin/nvcc)
            set(CMAKE_CUDA_COMPILER /usr/local/cuda-${CUDA_VER}/bin/nvcc CACHE FILEPATH "CUDA compiler" FORCE)
            break()
        endif()
    endforeach()
endif()

# CUDA 架构（必须在 project() 之前设置）
set(CMAKE_CUDA_ARCHITECTURES 75)

project(linear_solver VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# 查找 CUDA 库
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: ${CUDAToolkit_VERSION} at ${CUDAToolkit_ROOT_DIR}")
else()
    message(WARNING "CUDAToolkit not found. CUDA library linking may fail.")
    message(STATUS "To fix, set CUDAToolkit_ROOT or ensure CUDA is in PATH")
endif()

# 选项
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)

# 库源文件
set(LIB_SOURCES
    src/sparse_utils.cpp
    src/preconditioner.cpp
    src/pcg_solver.cpp
    src/amg_relaxation.cpp
    src/amg_coarsening.cpp
    src/amg_interpolation.cpp
    src/amg_hierarchy.cpp
    src/amg_solver.cpp
)

# 创建静态库
add_library(linear_solver STATIC ${LIB_SOURCES})

# 公共头文件目录
target_include_directories(linear_solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CUDA 相关设置（如果找到 CUDA）
if(CUDAToolkit_FOUND)
    # CUDA 头文件目录（PUBLIC：因为公共头文件包含 CUDA 头文件）
    target_include_directories(linear_solver PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    # 链接 CUDA 库
    target_link_libraries(linear_solver PUBLIC
        CUDA::cusparse
        CUDA::cublas
        CUDA::cudart
    )

    # CUDA 属性
    set_target_properties(linear_solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    # 备选：直接链接库名称
    target_link_libraries(linear_solver PUBLIC
        cusparse
        cublas
        cudart
    )

    set_target_properties(linear_solver PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# 添加子目录
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
