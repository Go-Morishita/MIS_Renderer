#include <vector>
#include <Eigen/Dense>  // Eigen���C�u�������g�p���čs��v�Z���s���܂�

// �m�����x�֐��̓��ς��v�Z����֐�
double dot_function(const std::function<double(double)>& p1, const std::function<double(double)>& p2, double start, double end, int div) {
    double step = (end - start) / div;
    double integral = 0.0;
    for (int i = 0; i < div; ++i) {
        double x = start + i * step;
        integral += p1(x) * p2(x);
    }
    return integral * step;
}

// �Z�p�s��A�̌v�Z
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

// ��^�x�N�g��b�̌v�Z
Eigen::VectorXd calculate_contribution_vector(const std::vector<std::function<double(double)>>& pdfs, const std::function<double(double)>& f, double start, double end, int div) {
    int n = pdfs.size();
    Eigen::VectorXd b(n);
    for (int i = 0; i < n; ++i) {
        b(i) = dot_function(pdfs[i], f, start, end, div);
    }
    return b;
}

// �I�v�e�B�}���d�݂̌v�Z
std::vector<double> optimal_weights(const std::vector<std::function<double(double)>>& pdfs, const std::function<double(double)>& f, double start, double end, int div) {
    int n = pdfs.size();
    Eigen::MatrixXd A = calculate_technique_matrix(pdfs, start, end, div);
    Eigen::VectorXd b = calculate_contribution_vector(pdfs, f, start, end, div);

    // �A��������A��=b������
    Eigen::VectorXd alpha = A.colPivHouseholderQr().solve(b);

    std::vector<double> weights(n);
    for (int i = 0; i < n; ++i) {
        weights[i] = alpha(i);
    }

    return weights;
}

// �T���v�����Ƃ̍ŏI�I�ȏd�݌v�Z
double compute_final_weight(double p_i, const std::vector<double>& alphas, const std::vector<std::function<double(double)>>& pdfs, double x) {
    double numerator = p_i;
    double denominator = 0.0;
    for (int j = 0; j < pdfs.size(); ++j) {
        denominator += pdfs[j](x);
    }
    double weight = numerator / denominator;
    return weight;
}
