#include <vector>
#include <Eigen/Dense>  // Eigenライブラリを使用して行列計算を行います

// 確率密度関数の内積を計算する関数
double dot_function(const std::function<double(double)>& p1, const std::function<double(double)>& p2, double start, double end, int div) {
    double step = (end - start) / div;
    double integral = 0.0;
    for (int i = 0; i < div; ++i) {
        double x = start + i * step;
        integral += p1(x) * p2(x);
    }
    return integral * step;
}

// 技術行列Aの計算
Eigen::MatrixXd calculate_technique_matrix(const std::vector<std::function<double(double)>>& pdfs, double start, double end, int div) {
    int n = pdfs.size();
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = dot_function(pdfs[i], pdfs[j], start, end, div);
        }
    }
    return A;
}

// 寄与ベクトルbの計算
Eigen::VectorXd calculate_contribution_vector(const std::vector<std::function<double(double)>>& pdfs, const std::function<double(double)>& f, double start, double end, int div) {
    int n = pdfs.size();
    Eigen::VectorXd b(n);
    for (int i = 0; i < n; ++i) {
        b(i) = dot_function(pdfs[i], f, start, end, div);
    }
    return b;
}

// オプティマル重みの計算
std::vector<double> optimal_weights(const std::vector<std::function<double(double)>>& pdfs, const std::function<double(double)>& f, double start, double end, int div) {
    int n = pdfs.size();
    Eigen::MatrixXd A = calculate_technique_matrix(pdfs, start, end, div);
    Eigen::VectorXd b = calculate_contribution_vector(pdfs, f, start, end, div);

    // 連立方程式Aα=bを解く
    Eigen::VectorXd alpha = A.colPivHouseholderQr().solve(b);

    std::vector<double> weights(n);
    for (int i = 0; i < n; ++i) {
        weights[i] = alpha(i);
    }

    return weights;
}

// サンプルごとの最終的な重み計算
double compute_final_weight(double p_i, const std::vector<double>& alphas, const std::vector<std::function<double(double)>>& pdfs, double x) {
    double numerator = p_i;
    double denominator = 0.0;
    for (int j = 0; j < pdfs.size(); ++j) {
        denominator += pdfs[j](x);
    }
    double weight = numerator / denominator;
    return weight;
}
