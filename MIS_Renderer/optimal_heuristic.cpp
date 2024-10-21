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

	std::vector <std::function<double(double)>> pdf = { p1, p2 };

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			auto term = [pdf, j](double x) {
				return pdf[j](x) / (pdf[0](x) + pdf[1](x));
				};
			A(i, j) = dot_function(pdf[i], term);
		}
	}

	return A;
}
