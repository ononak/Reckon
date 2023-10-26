//
//  LinAlgebraDef.cpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 25.01.2022.
//

#include "ILinearAlgebra.hpp"

namespace regu {

/*----------------------------------------------------------------*/
template <> bool operator==<Vec>(const Vec &v1, const Vec &v2) {

  bool a = arma::approx_equal(v1, v2, "absdiff", 0.002);
  return a;
}
/*----------------------------------------------------------------*/
template <> bool operator==<Mat>(const Mat &v1, const Mat &v2) {

  return arma::approx_equal(v1, v2, "absdiff", 0.002);
}
/*----------------------------------------------------------------*/
Mat makeEye(unsigned long long nrow, unsigned long long ncol) {

  return Mat(nrow, ncol, arma::fill::eye);
};

/*----------------------------------------------------------------*/
double computeNorm(const Vec &x, const unsigned int p) {
  return arma::norm(x, p);
};

/*----------------------------------------------------------------*/
double computeNorm(const Vec &x, const double p) {

  double sum = 0;
  for (auto i = 0; i < x.n_rows; i++) {
    sum += pow(abs(x[i]), p);
  }
  return pow(sum, 1 / p);
};

/*----------------------------------------------------------------*/
Vec makeEps(const Vec &x) { return arma::eps(x); };

/*----------------------------------------------------------------*/
Vec makeAbs(const Vec &x) { return arma::abs(x); };

/*----------------------------------------------------------------*/
Mat diagonalize(const Vec &x) { return arma::diagmat(x); };

/*----------------------------------------------------------------*/
Mat joinVertical(const Mat &X, const Mat &Y) { return arma::join_vert(X, Y); };

/*----------------------------------------------------------------*/
bool computeQrEconDecomposition(Mat &U, Mat &R, const Mat &A) {

  return arma::qr_econ(U, R, A);
};

/*----------------------------------------------------------------*/
bool computeSvd(Mat &U, Vec &s, Mat &V, const Mat &A) {
  return arma::svd(U, s, V, A);
};

/*----------------------------------------------------------------*/
Vec solveLinearSystem(const Mat &Lhs, const Vec &rhs) {
  return arma::solve(Lhs, rhs,
                     arma::solve_opts::fast + arma::solve_opts::no_approx);
};

}; // namespace regu
