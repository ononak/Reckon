
#ifndef Regularization_hpp
#define Regularization_hpp

#include "AlgebraInterface.hpp"
#include <tuple>

namespace regu {

// Solver for min_x {|y-Ax|_2 + |Lx|_2}
Vec solveL2L2(Vec y, Mat A, Mat L, double lambda);

/**
 Solver for min_x {|y-Ax|_p + |Lx|_q}  where 0< p <= 2 0 < q <= 2
 % Reference: A GENERALIZED KRYLOV SUBSPACE METHOD FOR Lp-Lq MINIMIZATION
 % A. LANZA, S. MORIGI, L. REICHEL, AND F. SGALLARI
 % DOI:10.1137/140967982
 */
std::tuple<Vec, double, double> solve(Vec y, Mat A, Mat L, double lambda,
                                      double pval, double qval);


}

#endif