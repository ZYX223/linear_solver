#include "../include/solvers.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

// ============================================================================
// 模板化的工具函数
// ============================================================================

template<Precision P>
std::unique_ptr<SparseMatrix<P>> read_matrix_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开矩阵文件: " << filename << std::endl;
        exit(1);
    }

    int rows, cols, nnz;
    file >> rows >> cols >> nnz;

    auto A = std::make_unique<SparseMatrix<P>>(rows, cols, nnz);

    using Scalar = typename SparseMatrix<P>::Scalar;

    std::vector<std::tuple<int, int, Scalar>> triples;
    for (int i = 0; i < nnz; i++) {
        int row, col;
        Scalar val;
        file >> row >> col >> val;
        triples.push_back(std::make_tuple(row, col, val));
    }
    file.close();

    std::vector<int> row_count(rows, 0);
    for (const auto& t : triples) {
        row_count[std::get<0>(t)]++;
    }

    A->row_ptr[0] = 0;
    for (int i = 0; i < rows; i++) {
        A->row_ptr[i + 1] = A->row_ptr[i] + row_count[i];
    }

    std::vector<int> current_row(rows, 0);
    for (const auto& t : triples) {
        int row = std::get<0>(t);
        int col = std::get<1>(t);
        Scalar val = std::get<2>(t);

        int idx = A->row_ptr[row] + current_row[row];
        A->col_ind[idx] = col;
        A->values[idx] = val;
        current_row[row]++;
    }

    A->upload_to_gpu();
    return A;
}

template<Precision P>
std::vector<typename ScalarType<P>::type> read_rhs_file(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开右端项文件: " << filename << std::endl;
        exit(1);
    }

    file >> n;

    using Scalar = typename ScalarType<P>::type;
    std::vector<Scalar> b(n);
    for (int i = 0; i < n; i++) {
        file >> b[i];
    }
    file.close();

    return b;
}

// 打印统计信息
void print_stats_line(const std::string& label, const SolveStats& stats) {
    std::cout << "  " << std::left << std::setw(24) << label
              << " | " << std::right << std::setw(7) << stats.iterations
              << " | " << std::scientific << std::setprecision(2) << std::setw(11) << stats.final_residual
              << " | " << std::fixed << std::setprecision(4) << std::setw(10) << stats.solve_time
              << " | " << std::setw(10) << std::right << (stats.converged ? "Yes" : "No")
              << " |" << std::endl;
}

// 打印表格分隔线
void print_separator() {
    std::cout << "  " << std::string(84, '-') << std::endl;
}

// 打印表格头
void print_table_header() {
    print_separator();
    std::cout << "  " << std::left << std::setw(24) << "Method"
              << " | " << std::right << std::setw(7) << "Iters"
              << " | " << std::setw(11) << "Residual"
              << " | " << std::setw(10) << "Time (s)"
              << " | " << std::setw(10) << "Converged"
              << " |" << std::endl;
    print_separator();
}

template<Precision P>
void save_solution(const std::string& filename, const std::vector<typename ScalarType<P>::type>& x) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "  警告: 无法创建解向量文件: " << filename << std::endl;
        return;
    }

    file << x.size() << "\n";

    for (size_t i = 0; i < x.size(); i++) {
        file << x[i] << "\n";
    }

    file.close();
}

template<Precision P>
void run_pcg_test(const std::string& name, Backend backend, bool use_ilu,
                  const SparseMatrix<P>& A, const std::vector<typename ScalarType<P>::type>& b, int n,
                  const std::string& output_dir) {
    PCGConfig config;
    config.max_iterations = 1000;
    config.tolerance = 1e-6;
    config.use_preconditioner = use_ilu;
    config.backend = backend;

    using Scalar = typename ScalarType<P>::type;
    std::vector<Scalar> x(n, ScalarConstants<P>::zero());

    PCGSolver<P> solver(config);
    SolveStats stats = solver.solve(A, b, x);
    print_stats_line(name, stats);

    // 保存解向量
    std::string precision_str = (P == Precision::Float32) ? "float" : "double";
    std::string backend_str = (backend == BACKEND_CPU) ? "cpu" : "gpu";
    std::string filename = output_dir + "/solution_pcg_" + backend_str + "_" + precision_str + ".dat";
    save_solution<P>(filename, x);
}

template<Precision P>
void run_amg_test(const std::string& name,
                  const SparseMatrix<P>& A,
                  const std::vector<typename ScalarType<P>::type>& b, int n,
                  const std::string& output_dir) {
    AMGConfig config;
    config.max_iterations = 100;
    config.tolerance = 1e-6;
    config.precision = P;
    config.max_levels = 10;
    config.coarse_grid_size = 50;
    config.pre_smooth_steps = 1;
    config.post_smooth_steps = 1;

    using Scalar = typename ScalarType<P>::type;
    std::vector<Scalar> x(n, ScalarConstants<P>::zero());

    AMGSolver<P> solver(config);
    SolveStats stats = solver.solve(A, b, x);
    print_stats_line(name, stats);

    // 保存解向量
    std::string precision_str = (P == Precision::Float32) ? "float" : "double";
    std::string filename = output_dir + "/solution_amg_cpu_" + precision_str + ".dat";
    save_solution<P>(filename, x);
}

template<Precision P>
void run_all_tests(const SparseMatrix<P>& A, const std::vector<typename ScalarType<P>::type>& b, int n,
                   const std::string& output_dir) {
    run_pcg_test<P>("PCG (CPU)", BACKEND_CPU, true, A, b, n, output_dir);
    run_pcg_test<P>("PCG (GPU)", BACKEND_GPU, true, A, b, n, output_dir);
    run_amg_test<P>("AMG (CPU)", A, b, n, output_dir);
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <矩阵文件> <右端项文件> [精度]\n";
        std::cerr << "精度选项: float, double (默认: float)\n";
        std::cerr << "示例: " << argv[0] << " matrix_poisson_P1_14401 matrix_poisson_P1rhs_14401 float\n";
        return 1;
    }

    std::string matrix_file = argv[1];
    std::string rhs_file = argv[2];
    std::string precision_str = (argc > 3) ? argv[3] : "float";

    Precision prec;
    if (precision_str == "double") {
        prec = Precision::Float64;
        std::cout << "使用双精度 (double)\n";
    } else {
        prec = Precision::Float32;
        std::cout << "使用单精度 (float)\n";
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " 求解器综合测试\n";
    std::cout << "========================================\n\n";

    std::cout << "[1/3] 读取矩阵文件: " << matrix_file << "\n";
    std::cout << "[2/3] 读取右端项文件: " << rhs_file << "\n";

    // 提取输出目录（与矩阵文件同级）
    size_t last_slash = matrix_file.find_last_of('/');
    std::string output_dir = (last_slash != std::string::npos) ?
                             matrix_file.substr(0, last_slash) : ".";

    // 根据精度分发到不同的实现
    if (prec == Precision::Float32) {
        auto A = read_matrix_file<Precision::Float32>(matrix_file);
        int n;
        std::vector<float> b = read_rhs_file<Precision::Float32>(rhs_file, n);

        std::cout << "[3/3] 运行测试...\n";
        std::cout << "\n问题规模: " << A->rows << " × " << A->cols << ", nnz = " << A->nnz << "\n";
        std::cout << "右端项大小: " << n << "\n";
        print_table_header();

        run_all_tests<Precision::Float32>(*A, b, n, output_dir);

        // 智能指针自动清理,无需手动 delete
    } else {
        auto A = read_matrix_file<Precision::Float64>(matrix_file);
        int n;
        std::vector<double> b = read_rhs_file<Precision::Float64>(rhs_file, n);

        std::cout << "[3/3] 运行测试...\n";
        std::cout << "\n问题规模: " << A->rows << " × " << A->cols << ", nnz = " << A->nnz << "\n";
        std::cout << "右端项大小: " << n << "\n";
        print_table_header();

        run_all_tests<Precision::Float64>(*A, b, n, output_dir);

        // 智能指针自动清理,无需手动 delete
    }

    print_separator();

    return 0;
}
