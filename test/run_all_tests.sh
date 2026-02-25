#!/bin/bash

# ============================================================================
# 线性求解器综合测试脚本
# 测试 matrices_florida 目录下的所有矩阵，生成测试报告
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_EXEC="$PROJECT_DIR/build/test/combined_test"
MATRICES_DIR="$SCRIPT_DIR/matrices_florida"
RESULTS_DIR="$SCRIPT_DIR/test_results_$(date +%Y%m%d_%H%M%S)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查可执行文件
if [ ! -f "$TEST_EXEC" ]; then
    echo -e "${RED}错误: 找不到测试程序 $TEST_EXEC${NC}"
    echo "请先构建项目:"
    echo "  cd $PROJECT_DIR/build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 检查矩阵目录
if [ ! -d "$MATRICES_DIR" ]; then
    echo -e "${RED}错误: 找不到矩阵目录 $MATRICES_DIR${NC}"
    exit 1
fi

# 创建结果目录
mkdir -p "$RESULTS_DIR"
REPORT_FILE="$RESULTS_DIR/test_report.txt"
CSV_FILE="$RESULTS_DIR/results.csv"

# 获取矩阵文件列表（按大小排序，小的先测试）
MATRIX_FILES=$(find "$MATRICES_DIR" -name "*.mtx" -exec ls -la {} \; 2>/dev/null | sort -k5 -n | awk '{print $NF}')

if [ -z "$MATRIX_FILES" ]; then
    echo -e "${RED}错误: 未找到 .mtx 矩阵文件${NC}"
    exit 1
fi

MATRIX_COUNT=$(echo "$MATRIX_FILES" | wc -l)

# ============================================================================
# 打印报告头
# ============================================================================
print_header() {
    echo "============================================================================"
    echo "             线性求解器综合测试报告"
    echo "============================================================================"
    echo ""
    echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "矩阵目录: $MATRICES_DIR"
    echo "矩阵数量: $MATRIX_COUNT"
    echo "精度测试: float + double (--both)"
    echo "结果目录: $RESULTS_DIR"
    echo ""
    echo "============================================================================"
    echo ""
}

# 写入 CSV 文件头
init_csv() {
    echo "Matrix,Rows,NNZ,Symmetric,Precision,Method,Iters,RelResidual,Time,Converged" > "$CSV_FILE"
}

# ============================================================================
# 提取矩阵信息
# ============================================================================
extract_matrix_info() {
    local mtx_file="$1"
    local rows=$(grep -v '^%' "$mtx_file" | head -1 | awk '{print $1}')
    local nnz=$(grep -v '^%' "$mtx_file" | head -1 | awk '{print $3}')
    local symmetric="No"
    if head -20 "$mtx_file" | grep -qi "symmetric"; then
        symmetric="Yes"
    fi
    echo "$rows $nnz $symmetric"
}

# ============================================================================
# 提取测试结果
# ============================================================================
extract_results() {
    local output="$1"
    local matrix_name="$2"
    local rows="$3"
    local nnz="$4"
    local symmetric="$5"
    local precision="$6"

    # 从输出中提取各方法的结果
    echo "$output" | grep -E "^\s*(PCG\+ILU|PCG\+IC|PCG\+AMG|AMG|amgcl)" | while read line; do
        local method=$(echo "$line" | awk '{print $1}' | sed 's/\s//g')
        local iters=$(echo "$line" | awk '{print $2}')
        local rel_res=$(echo "$line" | awk '{print $3}')
        local time=$(echo "$line" | awk '{print $4}')
        local converged=$(echo "$line" | awk '{print $5}')

        echo "$matrix_name,$rows,$nnz,$symmetric,$precision,$method,$iters,$rel_res,$time,$converged"
    done
}

# ============================================================================
# 主测试循环
# ============================================================================
print_header | tee "$REPORT_FILE"
init_csv

current=0
failed=0

for mtx_file in $MATRIX_FILES; do
    current=$((current + 1))
    matrix_name=$(basename "$mtx_file" .mtx)

    # 提取矩阵信息
    info=$(extract_matrix_info "$mtx_file")
    rows=$(echo "$info" | awk '{print $1}')
    nnz=$(echo "$info" | awk '{print $2}')
    symmetric=$(echo "$info" | awk '{print $3}')

    echo -e "${BLUE}[$current/$MATRIX_COUNT]${NC} 测试: $matrix_name (${rows}x${rows}, nnz=$nnz, 对称=$symmetric)"
    echo "------------------------------------------------------------------------" >> "$REPORT_FILE"
    echo "[$(date '+%H:%M:%S')] 测试矩阵: $matrix_name (${rows}x${rows}, nnz=$nnz)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"

    # 运行测试
    output_file="$RESULTS_DIR/${matrix_name}_output.txt"

    # 超时设置：大矩阵给更多时间
    if [ "$rows" -gt 50000 ]; then
        timeout=600
    elif [ "$rows" -gt 10000 ]; then
        timeout=300
    else
        timeout=120
    fi

    start_time=$(date +%s.%N)

    if timeout $timeout "$TEST_EXEC" "$mtx_file" --both > "$output_file" 2>&1; then
        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)

        echo -e "  ${GREEN}完成${NC} (耗时: ${elapsed}s)"

        # 提取并显示结果
        cat "$output_file" >> "$REPORT_FILE"

        # 提取 float 结果
        extract_results "$(sed -n '/float 精度/,/double 精度/p' "$output_file")" \
            "$matrix_name" "$rows" "$nnz" "$symmetric" "float" >> "$CSV_FILE"

        # 提取 double 结果
        extract_results "$(sed -n '/double 精度/,$ p' "$output_file")" \
            "$matrix_name" "$rows" "$nnz" "$symmetric" "double" >> "$CSV_FILE"

        # 显示简要结果
        echo "" >> "$REPORT_FILE"
        grep -E "^\s*(PCG\+ILU|PCG\+IC|PCG\+AMG|AMG|amgcl)" "$output_file" | head -10 | while read line; do
            echo "  $line"
        done
        echo "" >> "$REPORT_FILE"
    else
        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)
        failed=$((failed + 1))

        echo -e "  ${RED}失败或超时${NC} (耗时: ${elapsed}s)"
        echo "[ERROR] 测试失败或超时" >> "$REPORT_FILE"
        cat "$output_file" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi

    echo ""
done

echo ""
echo -e "${GREEN}测试完成！结果保存在: $RESULTS_DIR${NC}"
