#!/bin/bash

# 求解器综合测试脚本

echo "=========================================="
echo "求解器综合性能测试"
echo "=========================================="

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <矩阵文件> <右端项文件> [精度]"
    echo ""
    echo "参数:"
    echo "  矩阵文件: 矩阵文件路径"
    echo "  右端项文件: 右端项文件路径"
    echo "  精度: 可选，float 或 double (默认: 两者都测试)"
    echo ""
    echo "示例:"
    echo "  $0 matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401"
    echo "  $0 matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 float"
    echo "  $0 matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 double"
    exit 1
fi

MATRIX_FILE=$1
RHS_FILE=$2
PRECISION=${3:-both}  # 默认测试两种精度

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
echo "测试配置:"
echo "  矩阵: $MATRIX_FILE"
echo "  右端项: $RHS_FILE"
echo "  精度: $PRECISION"
echo ""

# 根据精度参数运行测试
if [ "$PRECISION" = "float" ]; then
    echo "=========================================="
    echo "单精度测试"
    echo "=========================================="
    ./build/main "$MATRIX_FILE" "$RHS_FILE" float
elif [ "$PRECISION" = "double" ]; then
    echo "=========================================="
    echo "双精度测试"
    echo "=========================================="
    ./build/main "$MATRIX_FILE" "$RHS_FILE" double
else
    echo "=========================================="
    echo "单精度测试"
    echo "=========================================="
    ./build/main "$MATRIX_FILE" "$RHS_FILE" float

    echo ""
    echo ""
    echo "=========================================="
    echo "双精度测试"
    echo "=========================================="
    ./build/main "$MATRIX_FILE" "$RHS_FILE" double
fi

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
