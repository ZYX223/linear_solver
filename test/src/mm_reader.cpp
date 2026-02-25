#include "mm_reader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstring>

namespace fs = std::filesystem;

MMMatrix readMatrixMarket(const std::string& filename) {
    MMMatrix mm;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::string line;
    bool header_found = false;
    int entry_count = 0;

    while (std::getline(file, line)) {
        // 跳过注释行和空行
        if (line.empty() || line[0] == '%') {
            // 解析头部信息
            if (line.length() > 1 && line[0] == '%') {
                // 检查是否对称矩阵
                std::string lower = line;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("symmetric") != std::string::npos) {
                    mm.symmetric = true;
                }
                if (lower.find("pattern") != std::string::npos) {
                    mm.pattern = true;
                }
            }
            continue;
        }

        std::istringstream iss(line);

        if (!header_found) {
            // 读取矩阵维度: 行数 列数 非零元素数
            if (!(iss >> mm.rows >> mm.cols >> mm.nnz)) {
                throw std::runtime_error("矩阵维度行格式错误");
            }
            header_found = true;

            // 预分配空间
            mm.entries.reserve(mm.nnz);
        } else {
            // 读取非零元素: 行 列 值
            int r, c;
            double v = 1.0;  // pattern 模式默认为1

            if (mm.pattern) {
                if (!(iss >> r >> c)) {
                    continue;  // 跳过格式错误的行
                }
            } else {
                if (!(iss >> r >> c >> v)) {
                    continue;  // 跳过格式错误的行
                }
            }

            // 转换为 0-based 索引
            mm.entries.emplace_back(r - 1, c - 1, v);
            entry_count++;

            // 如果是对称矩阵，添加对称元素
            if (mm.symmetric && r != c) {
                mm.entries.emplace_back(c - 1, r - 1, v);
                entry_count++;
            }
        }
    }

    // 更新实际非零元素数
    if (mm.symmetric) {
        mm.nnz = entry_count;
    }

    std::cout << "读取矩阵: " << mm.rows << " x " << mm.cols
              << ", nnz = " << mm.nnz
              << (mm.symmetric ? " [对称]" : "")
              << (mm.pattern ? " [模式]" : "") << std::endl;

    return mm;
}

void convertToCSR(const MMMatrix& mm,
                  std::vector<int>& row_ptr,
                  std::vector<int>& col_ind,
                  std::vector<double>& values) {

    // 首先按行排序
    std::vector<std::tuple<int, int, double>> sorted_entries = mm.entries;
    std::sort(sorted_entries.begin(), sorted_entries.end(),
              [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b)) {
                      return std::get<0>(a) < std::get<0>(b);
                  }
                  return std::get<1>(a) < std::get<1>(b);
              });

    // 统计每行非零元素数
    std::vector<int> row_counts(mm.rows + 1, 0);
    for (const auto& [r, c, v] : sorted_entries) {
        row_counts[r + 1]++;
    }

    // 计算行指针（前缀和）
    row_ptr.resize(mm.rows + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < mm.rows; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i + 1];
    }

    // 填充列索引和值
    col_ind.resize(mm.nnz);
    values.resize(mm.nnz);
    std::vector<int> current_row(mm.rows, 0);

    for (const auto& [r, c, v] : sorted_entries) {
        int idx = row_ptr[r] + current_row[r];
        col_ind[idx] = c;
        values[idx] = v;
        current_row[r]++;
    }
}

std::string downloadFloridaMatrix(const std::string& matrix_name,
                                   const std::string& group,
                                   const std::string& output_dir) {
    // 创建输出目录
    fs::create_directories(output_dir);

    std::string filename = output_dir + "/" + matrix_name + ".mtx";
    std::string url = "https://sparse.tamu.edu/MM/" + group + "/" + matrix_name + ".mtx";

    std::cout << "正在下载矩阵: " << matrix_name << std::endl;
    std::cout << "URL: " << url << std::endl;

    // 使用 curl 下载
    std::string cmd = "curl -s -o " + filename + " " + url;
    int ret = std::system(cmd.c_str());

    if (ret != 0) {
        throw std::runtime_error("下载失败: " + matrix_name);
    }

    std::cout << "已下载到: " << filename << std::endl;
    return filename;
}

std::vector<double> readVectorMarket(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::string line;
    bool header_found = false;
    bool is_sparse = false;  // 是否是稀疏格式
    int rows = 0, cols = 0, nnz = 0;
    std::vector<double> vec;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream iss(line);

        if (!header_found) {
            // 第一行：可能是 "size" 或 "rows cols nnz"
            int vals = 0;
            iss >> rows >> cols >> nnz;
            vals = (iss.fail() ? 1 : 3);

            if (vals == 3) {
                // 稀疏格式: rows cols nnz
                is_sparse = true;
                vec.resize(rows, 0.0);
                header_found = true;
            } else if (vals == 1) {
                // 密集格式: size
                iss.clear();
                iss.seekg(0);
                iss >> rows;
                vec.reserve(rows);
                header_found = true;
            } else {
                throw std::runtime_error("无法识别的向量格式");
            }
        } else {
            if (is_sparse) {
                // 稀疏格式: row col value
                int r, c;
                double val;
                if (iss >> r >> c >> val) {
                    if (r >= 1 && r <= rows) {
                        vec[r-1] = val;  // 转为 0-based
                    }
                }
            } else {
                // 密集格式: value
                double val;
                if (iss >> val) {
                    vec.push_back(val);
                }
            }
        }
    }

    if (static_cast<int>(vec.size()) != rows) {
        throw std::runtime_error("向量大小不匹配: 期望 " + std::to_string(rows) +
                               ", 实际 " + std::to_string(vec.size()));
    }

    std::cout << "读取" << (is_sparse ? "稀疏" : "密集") << "向量: " << vec.size() << " 个元素" << std::endl;
    return vec;
}

bool tryReadRHS(const std::string& matrix_file, std::vector<double>& rhs) {
    // 尝试多种可能的 RHS 文件命名
    std::vector<std::string> possible_names;

    // 提取基质文件的基本信息
    fs::path mat_path(matrix_file);
    std::string base_name = mat_path.stem().string();  // 不带扩展名的文件名
    std::string dir = mat_path.parent_path().string();

    // 可能的 RHS 文件名
    possible_names.push_back(dir + "/" + base_name + "_b.mtx");
    possible_names.push_back(dir + "/" + base_name + "_rhs.mtx");
    possible_names.push_back(dir + "/" + base_name + "_rhs1.mtx");
    possible_names.push_back(dir + "/b_" + base_name + ".mtx");
    possible_names.push_back(dir + "/rhs_" + base_name + ".mtx");
    possible_names.push_back(matrix_file + ".rhs");  // 支持 .mtx.rhs 格式

    // 尝试读取
    for (const auto& rhs_file : possible_names) {
        if (fs::exists(rhs_file)) {
            std::cout << "找到 RHS 文件: " << rhs_file << std::endl;
            try {
                // 检查是否是 .rhs 格式（简单文本格式）
                if (rhs_file.find(".rhs") != std::string::npos) {
                    // 读取 .rhs 格式
                    std::ifstream rhs_stream(rhs_file);
                    std::string line;
                    bool header = true;

                    while (std::getline(rhs_stream, line)) {
                        if (line.empty() || line[0] == '%') continue;
                        std::istringstream iss(line);
                        if (header) {
                            int size;
                            iss >> size;
                            rhs.reserve(size);
                            header = false;
                        } else {
                            double val;
                            iss >> val;
                            rhs.push_back(val);
                        }
                    }
                    std::cout << "成功读取 " << rhs.size() << " 个 RHS 值" << std::endl;
                    return true;
                } else {
                    // 读取 Matrix Market 格式
                    rhs = readVectorMarket(rhs_file);
                    return true;
                }
            } catch (const std::exception& e) {
                std::cerr << "读取 RHS 文件失败: " << e.what() << std::endl;
            }
        }
    }

    std::cout << "未找到 RHS 文件，将使用 x_exact = [1,1,...,1]^T 生成" << std::endl;
    return false;
}
