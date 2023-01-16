//
//  LinAlgebraDef.hpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 25.01.2022.
//

#ifndef LinAlgebraDef_hpp
#define LinAlgebraDef_hpp

#include <armadillo>

namespace regu {

using Vec = arma::vec;
using Mat = arma::mat;

Mat la_eye(unsigned long long nrow, unsigned long long ncol);
/**
 vector norm
 */
double la_norm(const Vec &x, const unsigned int p = 2);
double la_norm(const Vec &x, const double p);

Vec la_eps(const Vec &x);
Vec la_abs(const Vec &x);

/**
 make diagonal matrix
 */
Mat la_diag(const Vec &x);

/**
 vertically merge  matrices
 */
Mat la_join_vertical(const Mat &X, const Mat &Y);

/**
 economical qr decomposition
 */
bool la_qr_econ(Mat &U, Mat &R, const Mat &A);
/**
 svd decomposition
 */
bool la_svd(Mat &U, Vec &s, Mat &V, const Mat &A);

/**
 solve the system of linear equation
 */
Vec la_solve(const Mat &Lhs, const Vec &rhs);

}; // namespace regu

#endif /* LinAlgebraDef_hpp */
