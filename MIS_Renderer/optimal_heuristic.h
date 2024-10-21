#ifndef OPTIMAL_HEURISTIC_H
#define OPTIMAL_HEURISTIC_H

#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_DONT_VECTORIZE

#include <Eigen/Dense>

double dot_function(const std::function<double(double)>& f1, const std::function<double(double)>& f2, double start, double end, int div);

#endif // OPTIMAL_HEURISTIC_H
