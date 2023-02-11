#include "Regularization.hpp"

namespace regu {

Vec solveL2L2(Vec y, Mat A, Mat L, double lambdasqr) {

  // construct augmented matrix [A;L]
  auto AugA = la_join_vertical(A, L);

  // compute qr decomposition
  Mat Q, Q1, R;
  if (!la_qr_econ(Q, R, AugA)) {
    std::cout << "qr failed" << std::endl;
  }
  Q1 = Q.rows(0, A.n_rows - 1);

  // compute svd
  Mat U1, V1;
  Vec s1;
  if (!la_svd(U1, s1, V1, Q1)) {
    std::cout << "svd failed" << std::endl;
  }

  // solve the system of linear equation to find the unknown vector x
  Mat S1 = la_diag(s1);
  S1.reshape(A.n_rows, A.n_cols);
  Vec rhs = S1.t() * U1.t() * y;
  Mat In = la_eye(A.n_cols, A.n_cols);
  Mat Lhs = (lambdasqr * In + (1 - lambdasqr) * S1.t() * S1);
  Vec b = la_solve(Lhs, rhs);
  auto xlhs = (V1.t() * R);
  Vec x_reg = la_solve(xlhs, b);

  return x_reg;
}


std::tuple<Vec, double, double> solve(Vec y, Mat A, Mat L, double lambda,
                                      double p, double q)
{
 // init loop control params
  double tolerance = 1e-3;
  double diff = 1e10;
  int iteration = 0;
  const int MAX_ITER = 500;

  // square of regularization parameter
  auto lambda2 = lambda * lambda;

  // initial estiMation
  Vec x_est = solveL2L2(y, A, L, lambda2);

  // iterate solution till converge
  while ((iteration <= MAX_ITER) && (diff >= tolerance)) {
    // Compute weight Matrices
    Vec v = la_abs(A * x_est - y);
    Vec z = la_abs(L * x_est);

    auto elementwise_pow = [](Vec &source, double pw) -> Vec {
      Vec retval(source.size());

      for (int i = 0; i < source.size(); i++) {
        retval(i) =
            pow(abs(source(i)) + std::numeric_limits<double>::epsilon(), pw);
      }
      retval += la_eps(retval);
      return retval;
    };

    Vec wf = elementwise_pow(v, (p - 2) / 2);
    Vec wr = elementwise_pow(z, (q - 2) / 2);

    auto WF = la_diag(wf);
    auto WR = la_diag(wr);
    auto AF = WF * A;
    auto LR = WR * L;
    auto bF = WF * y;

    Vec x_est_new = solveL2L2(bF, AF, LR, lambda2);
    diff = la_norm(x_est_new - x_est) / la_norm(x_est_new);
    x_est = x_est_new;

    iteration++;
  }

  double constraint_norm = la_norm(L * x_est, q);
  double residual_norm = la_norm(y - A * x_est, p);

  return std::make_tuple(x_est, constraint_norm, residual_norm);
}
}