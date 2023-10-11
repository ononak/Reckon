#include "Regularization.hpp"

namespace regu {

Vec solve(Vec y, Mat A, Mat L, double lambda) {

  // square of regularization parameter
  double lambdasqr = lambda * lambda;

  // construct augmented matrix [A;L]
  auto AugA = joinVertical(A, L);

  // compute qr decomposition
  Mat Q, Q1, R;
  if (!computeQrEconDecomposition(Q, R, AugA)) {
    std::cout << "qr failed" << std::endl;
  }
  Q1 = Q.rows(0, A.n_rows - 1);

  // compute svd
  Mat U1, V1;
  Vec s1;
  if (!computeSvd(U1, s1, V1, Q1)) {
    std::cout << "svd failed" << std::endl;
  }

  // solve the system of linear equation to find the unknown vector x
  auto S1 = diagonalize(s1);
  S1.reshape(A.n_rows, A.n_cols);
  Vec rhs = S1.t() * U1.t() * y;
  auto In = makeEye(A.n_cols, A.n_cols);
  Mat Lhs = (lambdasqr * In + (1 - lambdasqr) * S1.t() * S1);
  auto b = solveLinearSystem(Lhs, rhs);
  auto xlhs = (V1.t() * R);
  auto x_reg = solveLinearSystem(xlhs, b);

  return x_reg;
}

std::tuple<Vec, double, double> solve(Vec y, Mat A, Mat L, double lambda,
                                      double p, double q) {
  int iteration = 0;
  double diff = 1e10;
  // initial estimation ıs L2L2 solution
  auto xEstimated = solve(y, A, L, lambda);

  // iterate solution till converge
  while ((iteration <= MAX_ITER) && (diff >= tolerance)) {
    // Compute weight Matrices
    auto v = makeAbs(A * xEstimated - y);
    auto z = makeAbs(L * xEstimated);

    auto elementwise_pow = [](Vec &source, double pw) -> Vec {
      Vec retval(source.size());

      for (int i = 0; i < source.size(); i++) {
        retval(i) =
            pow(abs(source(i)) + std::numeric_limits<double>::epsilon(), pw);
      }
      retval += makeEps(retval);
      return retval;
    };

    auto wf = elementwise_pow(v, (p - 2) / 2);
    auto wr = elementwise_pow(z, (q - 2) / 2);

    auto WF = diagonalize(wf);
    auto WR = diagonalize(wr);
    auto AF = WF * A;
    auto LR = WR * L;
    auto bF = WF * y;

    auto xEstimatedNew = solve(bF, AF, LR, lambda);
    diff = computeNorm(xEstimatedNew - xEstimated) / computeNorm(xEstimatedNew);
    xEstimated = xEstimatedNew;

    iteration++;
  }

  auto constraint_norm = computeNorm(L * xEstimated, q);
  auto residual_norm = computeNorm(y - A * xEstimated, p);

  return std::make_tuple(xEstimated, constraint_norm, residual_norm);
}
} // namespace regu