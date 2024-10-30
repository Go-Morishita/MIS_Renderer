#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_DONT_VECTORIZE

#include "optimal_heuristic.h"

#include <iostream>
#include <functional>
#include <vector>
#include <cmath>

#define start 0
#define end 1
#define div 1000

double dot_function(const std::function<double(double)>& f1, const std::function<double(double)>& f2)
{
	double step = (end - start) / div;
	double integral = 0.0;
	for (int i = 0; i < div; i++) {
		double x = start + i * step;
		integral += f1(x) * f2(x);
	}

	return integral * step;
}

Eigen::Matrix2d A_matrix(const std::function<double(double)>& p1, const std::function<double(double)>& p2,const double x)
{
	Eigen::Matrix2d A(2, 2);

	std::vector <std::function<double(double)>> pdfs = { p1, p2 };

	const double denominator = p1(x) + p2(x);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			auto term = [pdfs, j, denominator](double x) {
				return pdfs[j](x) / denominator;
				};
			auto pdf = [pdfs, i](double x) {
				return pdfs[i](x);
				};
			A(i, j) = dot_function(pdf, term);
		}
	}

	return A;
}

Eigen::Vector2d b_vector(const std::function<double(double)>& f,const std::function<double(double)>& p1, const std::function<double(double)>& p2, const double x)
{
	Eigen::Vector2d b;

	std::vector <std::function<double(double)>> pdf = { p1, p2 };

	for (int i = 0; i < 2; i++) {
		auto term = [pdf, i](double x) {
			return pdf[i](x) / (pdf[0](x) + pdf[1](x));
			};
		auto func = [f](double x) { return f(x); };
		b(i) = dot_function(func, term);
	}

	return b;
}
