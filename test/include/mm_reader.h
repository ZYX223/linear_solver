#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>

/**
 * @brief Matrix Market 格式矩阵数据结构
 */
struct MMMatrix {
    int rows = 0;          // 行数
    int cols = 0;          // 列数
    int nnz = 0;           // 非零元素数
    bool symmetric = false; // 是否对称矩阵
    bool pattern = false;   // 是否只有模式（没有数值）
    std::vector<std::tuple<int, int, double>> entries;  // (row, col, value) 0-based
};

/**
 * @brief 读取 Matrix Market 格式稀疏矩阵文件
 *
 * @param filename .mtx 文件路径
 * @return MMMatrix 矩阵数据
 * @throws std::runtime_error 如果文件读取失败或格式错误
 */
MMMatrix readMatrixMarket(const std::string& filename);

/**
 * @brief 将 MMMatrix 转换为 CSR 格式的数组
 *
 * @param mm Matrix Market 矩阵
 * @param row_ptr 输出: CSR 行指针数组
 * @param col_ind 输出: CSR 列索引数组
 * @param values 输出: CSR 值数组
 */
void convertToCSR(const MMMatrix& mm,
                  std::vector<int>& row_ptr,
                  std::vector<int>& col_ind,
                  std::vector<double>& values);

/**
 * @brief 从 SuiteSparse Matrix Collection 下载矩阵
 *
 * @param matrix_name 矩阵名称（如 "poisson3dB"）
 * @param group 矩阵组（如 "Williams"）
 * @param output_dir 输出目录
 * @return std::string 下载的文件路径
 */
std::string downloadFloridaMatrix(const std::string& matrix_name,
                                   const std::string& group,
                                   const std::string& output_dir = "./matrices");

/**
 * @brief 尝试读取 Matrix Market 格式的 RHS 向量文件
 *
 * @param matrix_file 矩阵文件路径，函数会尝试查找对应的 RHS 文件
 * @param rhs 输出: RHS 向量
 * @return bool 如果找到并成功读取 RHS 文件返回 true，否则返回 false
 */
bool tryReadRHS(const std::string& matrix_file, std::vector<double>& rhs);

/**
 * @brief 读取 Matrix Market 格式的向量文件
 *
 * @param filename .mtx 向量文件路径
 * @return std::vector<double> 向量数据
 * @throws std::runtime_error 如果文件读取失败或格式错误
 */
std::vector<double> readVectorMarket(const std::string& filename);
