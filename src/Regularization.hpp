
#ifndef REGULARIZATION_HPP
#define REGULARIZATION_HPP

#include "ILinearAlgebra.hpp"
#include <__tuple_dir/tuple_element.h>
#include <tuple>

namespace sci {

// loop control params
constexpr double tolerance = 1e-3;
constexpr int MAX_ITER = 500;

/**
 * @brief Solver for optimization problem min_x {|y-Ax|_p + lambda^2|Lx|_q}
 * where 0< p <= 2 0 < q <= 2 A GENERALIZED KRYLOV SUBSPACE METHOD FOR Lp-Lq
 * MINIMIZATION DOI:10.1137/140967982
 */
class LpLqRegularization {

public:
  /**
   * @brief Construct a new Lp Lq Regularization object
   *
   * @param forwardM linear tranfer matrix
   * @param regularizationM regularization matrix
   */
  LpLqRegularization(const Mat &forwardM, const Mat &regularizationM);
  ~LpLqRegularization();

  /**
   * @brief Solve optimization problem min_x {|y-Ax|_p + lambda^2|Lx|_q}
   *
   * @param y noisy measurement
   * @param lambda regularization parameter
   * @param pNorm vector norm for residual
   * @param qNorm vector norm for contsraint
   * @return std::tuple<Result, Vec, double, double> that contains the
   * estimation of x , |Lx|_q, and |y-Ax|_p
   */
  std::tuple<Result, Vec, double, double>
  solve(const Vec &y, double lambda, double pNorm = 2, double qNorm = 2) const;

protected:
  /**
   * @brief Solve optimization problem min_x {|y-Ax|_2 + lambda^2|Lx|_2}
   */
  std::tuple<Result, Vec> solveTikhonov(const Vec &y, const Mat &fm,
                                        const Mat &rm, double lambda) const;

private:
  LpLqRegularization(const LpLqRegularization &other) = delete;
  LpLqRegularization(LpLqRegularization &&other) = delete;
  LpLqRegularization &operator=(const LpLqRegularization &other) = delete;

private:
  /**
   * @brief Forward transfer matrix which tranfers input to measurement
   */
  Mat forwardMatrix;
  /**
   * @brief Regularization matrix, which express the prior information about the
   * unknown state variable vector
   */
  Mat regularizationMatrix;
};
} // namespace sci

#endif