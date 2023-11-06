//
//  LinAlgebraDef.hpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 25.01.2022.
//

#ifndef I_LINEAR_ALGEBRA_HPP
#define I_LINEAR_ALGEBRA_HPP

#include "SciTool.hpp"
#include <armadillo>

namespace sci {

using Vec = arma::vec;
using Mat = arma::mat;

/**
 * @brief
 *
 * @tparam T
 * @param v1
 * @param v2
 * @return true
 * @return false
 */
template <typename T> bool operator==(const T &v1, const T &v2) {
  return v1 == v2;
}

/**
 * @brief Generate unit matrix
 *
 * @param nrow
 * @param ncol
 * @return Mat
 */
Mat makeEye(unsigned long long nrow, unsigned long long ncol);

/**
 * @brief  Compute matrix norm
 *
 * @param x
 * @param p
 * @return double
 */
double computeNorm(const Vec &x, const unsigned int p = 2);

/**
 * @brief Compute the non-integer norm of the matrix
 *
 * @param x
 * @param p
 * @return double
 */
double computeNorm(const Vec &x, const double p);

/**
 * @brief eps vector
 *
 * @param x
 * @return Vec
 */
Vec makeEps(const Vec &x);

/**
 * @brief
 *
 * @param x
 * @return Vec
 */
Vec makeAbs(const Vec &x);

/**
 * @brief Make diagonal matrix
 *
 * @param x
 * @return Mat
 */
Mat diagonalize(const Vec &x);

/**
 * @brief Merge matrices vertically
 *
 * @param X
 * @param Y
 * @return Mat
 */
Mat joinVertical(const Mat &X, const Mat &Y);

/**
 * @brief Economical qr decomposition
 *
 * @param U
 * @param R
 * @param A
 * @return true
 * @return false
 */
bool computeQrEconDecomposition(Mat &U, Mat &R, const Mat &A);

/**
 * @brief
 *
 * @param U
 * @param s
 * @param V
 * @param A
 * @return true
 * @return false
 */
bool computeSvd(Mat &U, Vec &s, Mat &V, const Mat &A);

/**
 * @brief solve the system of linear equation
 *
 * @param Lhs
 * @param rhs
 * @return Vec
 */
Vec solveLinearSystem(const Mat &Lhs, const Vec &rhs);

/**
 * @brief
 *
 * @param M
 * @return Mat
 */
Mat inverse(const Mat &M);

}; // namespace sci

#endif
