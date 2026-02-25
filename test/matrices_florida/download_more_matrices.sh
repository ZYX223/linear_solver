#!/bin/bash
# 下载多种类型的中型测试矩阵

MATRICES=(
    # 对称正定矩阵
    "HB/bcsstk16|对称正定|10974"
    "HB/bcsstk17|对称正定|10974"  
    "HB/s3rmq4m1|对称正定|5489"
    "HB/s2rmq4m1|对称正定|5489"
    
    # 非对称矩阵
    "Boeing/bcsstk20|非对称|485"
    "Boeing/bcsstk21|非对称|3600"
    "Rothberg/cavity1|流体/非对称|3174"
    "Rothberg/cavity2|流体/非对称|3202"
    "Boeing/ex5|非对称|2924"
    "GHS_indef/cont-201|非对称|160"
    
    # 结构/力学
    "HB/nos4|结构|100"
    "HB/nos5|结构|506"
    "HB/nos6|结构|675"
    
    # 其他
    "Simon/af23560|热传导|23560"
    "DRIVCAV/cavity3|流体|525"
)

echo "开始下载矩阵到当前目录..."
echo ""

for matrix in "${MATRICES[@]}"; do
    IFS='|' read -r group_name type size <<< "$matrix"
    name=$(basename "$group_name")
    url="https://sparse.tamu.edu/MM/${group_name}.mtx"
    output="${name}.mtx"
    
    if [ -f "$output" ]; then
        echo "跳过 (已存在): $name"
        continue
    fi
    
    echo "下载: $name ($type, ~$size 行)"
    if curl -f -s -o "$output" "$url"; then
        actual_size=$(du -h "$output" | cut -f1)
        echo "  成功: $actual_size"
    else
        echo "  失败: $name"
        rm -f "$output"
    fi
done

echo ""
echo "下载完成！当前矩阵列表:"
ls -lh *.mtx
