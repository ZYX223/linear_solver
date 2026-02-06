#include "../include/pcg_solver.h"
#include "../include/preconditioner.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

// ============================================================================
// 工具函数
// ============================================================================

// 读取矩阵文件（格式：行 列 值，第一行是：行数 列数 非零元数）
SparseMatrix* read_matrix_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开矩阵文件: " << filename << std::endl;
        exit(1);
    }

    int rows, cols, nnz;
    file >> rows >> cols >> nnz;

    std::cout << "  矩阵信息: " << rows << " x " << cols << ", nnz = " << nnz << std::endl;

    SparseMatrix* A = new SparseMatrix(rows, cols, nnz);

    // 读取矩阵数据（三元组格式）
    std::vector<std::tuple<int, int, float>> triples;
    for (int i = 0; i < nnz; i++) {
        int row, col;
        float val;
        file >> row >> col >> val;
        triples.push_back(std::make_tuple(row, col, val));
    }
    file.close();

    // 转换为 CSR 格式
    std::vector<int> row_count(rows, 0);
    for (const auto& t : triples) {
        row_count[std::get<0>(t)]++;
    }

    A->row_ptr[0] = 0;
    for (int i = 0; i < rows; i++) {
        A->row_ptr[i + 1] = A->row_ptr[i] + row_count[i];
    }

    // 填充 col_ind 和 values
    std::vector<int> current_row(rows, 0);
    for (const auto& t : triples) {
        int row = std::get<0>(t);
        int col = std::get<1>(t);
        float val = std::get<2>(t);

        int idx = A->row_ptr[row] + current_row[row];
        A->col_ind[idx] = col;
        A->values[idx] = val;
        current_row[row]++;
    }

    A->upload_to_gpu();
    return A;
}

// 读取右端项文件（第一行是向量大小，后面是值）
std::vector<float> read_rhs_file(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开右端项文件: " << filename << std::endl;
        exit(1);
    }

    file >> n;
    std::cout << "  右端项大小: " << n << std::endl;

    std::vector<float> b(n);
    for (int i = 0; i < n; i++) {
        file >> b[i];
    }
    file.close();

    return b;
}

// 打印统计信息
void print_stats_line(const std::string& label, const SolveStats& stats) {
    std::cout << "  " << std::setw(25) << std::left << label
              << " | " << std::setw(6) << stats.iterations
              << " | " << std::scientific << std::setw(10) << stats.final_residual
              << " | " << std::fixed << std::setw(10) << std::setprecision(4) << stats.solve_time
              << " | " << std::setw(10) << (stats.converged ? "Yes" : "No")
              << " |\n";
}

// 打印表格分隔线
void print_separator() {
    std::cout << "  " << std::string(85, '-') << "\n";
}

// 打印表格头
void print_table_header() {
    print_separator();
    std::cout << "  " << std::setw(25) << std::left << "Method"
              << " | " << std::setw(6) << "Iters"
              << " | " << std::setw(10) << "Residual"
              << " | " << std::setw(10) << "Time (s)"
              << " | " << std::setw(10) << "Converged"
              << " |\n";
    print_separator();
}

// 保存解向量到文件
void save_solution(const std::string& filename, const std::vector<float>& x) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "  警告: 无法创建解向量文件: " << filename << std::endl;
        return;
    }

    // 写入向量大小
    file << x.size() << "\n";

    // 写入解向量值
    for (size_t i = 0; i < x.size(); i++) {
        file << x[i] << "\n";
    }

    file.close();
    std::cout << "  解向量已保存到: " << filename << std::endl;
}

// 运行测试
void run_test(const std::string& name, Backend backend, bool use_ilu,
              const SparseMatrix& A, const std::vector<float>& b, int n,
              const std::string& output_prefix = "") {
    PCGConfig config;
    config.max_iterations = 1000;
    config.tolerance = 1e-6f;
    config.use_preconditioner = use_ilu;
    config.backend = backend;

    std::vector<float> x(n, 0.0f);
    PCGSolver solver(config);

    SolveStats stats = solver.solve(A, b, x);
    print_stats_line(name, stats);

    // 保存解向量
    if (!output_prefix.empty()) {
        std::string filename = output_prefix + "_" + name + ".txt";
        // 替换空格和括号，使文件名更友好
        std::replace(filename.begin(), filename.end(), ' ', '_');
        filename.erase(std::remove(filename.begin(), filename.end(), '('), filename.end());
        filename.erase(std::remove(filename.begin(), filename.end(), ')'), filename.end());
        save_solution(filename, x);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <矩阵文件> <右端项文件>" << std::endl;
        std::cerr << "示例: " << argv[0] << " matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401" << std::endl;
        return 1;
    }

    std::string matrix_file = argv[1];
    std::string rhs_file = argv[2];

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  PCG 求解器综合测试\n";
    std::cout << "========================================\n\n";

    // 读取矩阵和右端项
    std::cout << "[1/3] 读取矩阵文件: " << matrix_file << "\n";
    SparseMatrix* A = read_matrix_file(matrix_file);

    std::cout << "\n[2/3] 读取右端项文件: " << rhs_file << "\n";
    int n;
    std::vector<float> b = read_rhs_file(rhs_file, n);

    std::cout << "\n[3/3] 运行测试...\n";
    print_table_header();

    // 生成输出文件前缀（基于矩阵文件名）
    std::string output_prefix = matrix_file;

    run_test("CPU (无预处理)", BACKEND_CPU, false, *A, b, n, output_prefix);
    run_test("CPU + ILU(0)", BACKEND_CPU, true, *A, b, n, output_prefix);
    run_test("GPU (无预处理)", BACKEND_GPU, false, *A, b, n, output_prefix);
    run_test("GPU + ILU(0)", BACKEND_GPU, true, *A, b, n, output_prefix);

    print_separator();

    // 清理
    delete A;

    return 0;
}
