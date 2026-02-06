#!/bin/bash

# PCG Solver 测试脚本

echo "=========================================="
echo "PCG Solver 综合性能测试"
echo "=========================================="

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <矩阵文件> <右端项文件>"
    echo ""
    echo "示例:"
    echo "  $0 matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401"
    exit 1
fi

MATRIX_FILE=$1
RHS_FILE=$2

# 检查文件是否存在
if [ ! -f "$MATRIX_FILE" ]; then
    echo "错误: 找不到矩阵文件: $MATRIX_FILE"
    exit 1
fi

if [ ! -f "$RHS_FILE" ]; then
    echo "错误: 找不到右端项文件: $RHS_FILE"
    exit 1
fi

# 检查可执行文件
if [ ! -f "./build/main" ]; then
    echo "错误: 找不到可执行文件 main!"
    echo "请运行: cd build && cmake .. && make"
    exit 1
fi

# 运行测试
echo ""
echo "运行 PCG 求解器..."
echo "  矩阵: $MATRIX_FILE"
echo "  右端项: $RHS_FILE"
echo ""

./build/main "$MATRIX_FILE" "$RHS_FILE"

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
