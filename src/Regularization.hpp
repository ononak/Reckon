
#ifndef REGULARIZATION_HPP
#define REGULARIZATION_HPP

#include "ILinearAlgebra.hpp"
#include <tuple>

namespace regu {

// loop control params
constexpr double tolerance = 1e-3;
constexpr int MAX_ITER = 500;
/**
 * @brief Solver for min_x {|y-Ax|_2 + lambda^2*|Lx|_2}
 *
 * @param y
 * @param A
 * @param L
 * @param lambda
 * @return Vec
 */
Vec solve(Vec y, Mat A, Mat L, double lambda);

/**
 * @brief Solver for min_x {|y-Ax|_p + lambda^2|Lx|_q}  where 0< p <= 2 0 < q <=
 * 2 A GENERALIZED KRYLOV SUBSPACE METHOD FOR Lp-Lq MINIMIZATION
 * DOI:10.1137/140967982
 * @param y noisy measurement
 * @param A linear tranfer matrix
 * @param L regularization matrix
 * @param lambda regularization parameter
 * @param p vector norm for residual
 * @param q vector norm for contraint
 * @return std::tuple<Vec, double, double> that contains the estimation of x ,
 * |Lx|_q, and |y-Ax|_p
 */
std::tuple<Vec, double, double> solve(Vec y, Mat A, Mat L, double lambda,
                                      double p, double q);

} // namespace regu

#endif