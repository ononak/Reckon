//
//  LinAlgebraDef.cpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 25.01.2022.
//

#include "LinAlgebraDef.hpp"

namespace regu {

Mat la_eye(unsigned long long nrow, unsigned long long ncol) {

  return Mat(nrow, ncol, arma::fill::eye);
};

double la_norm(const Vec &x, const unsigned int p) { return arma::norm(x, p); };

double la_norm(const Vec &x, const double p) {

  double sum = 0;
  for (auto i = 0; i < x.n_rows; i++) {
    sum += pow(abs(x[i]), p);
  }

  return pow(sum, 1 / p);
};

Vec la_eps(const Vec &x) { return arma::eps(x); };

Vec la_abs(const Vec &x) { return arma::abs(x); };

Mat la_diag(const Vec &x) { return arma::diagmat(x); };

Mat la_join_vertical(const Mat &X, const Mat &Y) {

  return arma::join_vert(X, Y);
};

bool la_qr_econ(Mat &U, Mat &R, const Mat &A) {

  return arma::qr_econ(U, R, A);
};

bool la_svd(Mat &U, Vec &s, Mat &V, const Mat &A) {
  return arma::svd(U, s, V, A);
};

Vec la_solve(const Mat &Lhs, const Vec &rhs) {
  return arma::solve(Lhs, rhs,
                     arma::solve_opts::fast + arma::solve_opts::no_approx);
};

}; // namespace regu
