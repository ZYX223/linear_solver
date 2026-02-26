#pragma once
#ifndef __CPU_ONLY__
#include <cuda_runtime.h>
#include <cusparse.h>
#endif
#include <type_traits>

// ============================================================================
// 精度类型枚举
// ============================================================================

enum class Precision { Float32, Float64 };

// ============================================================================
// CUDA数据类型映射
// ============================================================================

template<Precision P> struct CudaDataType;
template<> struct CudaDataType<Precision::Float32> {
    static constexpr cudaDataType_t value = CUDA_R_32F;
};
template<> struct CudaDataType<Precision::Float64> {
    static constexpr cudaDataType_t value = CUDA_R_64F;
};

// ============================================================================
// C++标量类型映射
// ============================================================================

template<Precision P> struct ScalarType;
template<> struct ScalarType<Precision::Float32> { using type = float; };
template<> struct ScalarType<Precision::Float64> { using type = double; };

// ============================================================================
// 常量定义
// ============================================================================

template<Precision P>
struct ScalarConstants {
    static typename ScalarType<P>::type one();
    static typename ScalarType<P>::type zero();
};

template<>
inline float ScalarConstants<Precision::Float32>::one() { return 1.0f; }

template<>
inline float ScalarConstants<Precision::Float32>::zero() { return 0.0f; }

template<>
inline double ScalarConstants<Precision::Float64>::one() { return 1.0; }

template<>
inline double ScalarConstants<Precision::Float64>::zero() { return 0.0; }

// ============================================================================
// 类型别名模板
// ============================================================================

template<Precision P>
using ScalarT = typename ScalarType<P>::type;