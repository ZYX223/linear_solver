cmake_minimum_required(VERSION 3.20)
project(pcg_example CXX)

# C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/../include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# 链接目录
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 主项目源文件
set(PCG_SOURCES
    ../src/sparse_utils.cpp
    ../src/preconditioner.cpp
    ../src/pcg_solver.cpp
)

# 测试程序
add_executable(main main.cpp ${PCG_SOURCES})

# 链接库
target_link_libraries(main
    cusparse
    cublas
    cudart
)

# CUDA编译选项
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
