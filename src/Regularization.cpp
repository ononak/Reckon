#include "Regularization.hpp"

namespace regu {

std::tuple<bool, Vec> solve(Vec y, Mat A, Mat L, double lambda) {

  bool result = true;
  Vec xEstimated;
  double lambdasqr = lambda * lambda;

  if ((A.n_cols != L.n_cols) || (y.n_rows != A.n_rows)) {
    result = false;
  }

  if (result) {
    // construct augmented matrix [A;L]
    auto AugA = joinVertical(A, L);

    // compute qr decomposition
    Mat Q, R;
    result = computeQrEconDecomposition(Q, R, AugA);
    if (result) {
      Mat Q1, U1, V1;
      Vec s1;
      Q1 = Q.rows(0, A.n_rows - 1);
      if (!computeSvd(U1, s1, V1, Q1)) {
        result = false;
      }

      if (result) {
        // solve the system of linear equation to find the unknown vector x
        auto S1 = diagonalize(s1);
        S1.reshape(A.n_rows, A.n_cols);
        Vec rhs = S1.t() * U1.t() * y;
        auto In = makeEye(A.n_cols, A.n_cols);
        Mat Lhs = (lambdasqr * In + (1 - lambdasqr) * S1.t() * S1);

        try {
          auto b = solveLinearSystem(Lhs, rhs);
          auto xlhs = (V1.t() * R);
          xEstimated = solveLinearSystem(xlhs, b);
        } catch (...) {
          result = false;
        }
      }
    }
  }
  return std::tuple(result, xEstimated);
}

std::tuple<bool, Vec, double, double> solve(Vec y, Mat A, Mat L, double lambda,
                                            double p, double q) {
  int iteration = 0;
  double diff = 1e10;

  // initial estimation ıs L2L2 solution
  auto [result, xEstimated] = solve(y, A, L, lambda);

  // iterate solution till converge
  while ((iteration <= MAX_ITER) && (diff >= tolerance) && (result == true)) {
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

    auto [result, xEstimatedNew] = solve(bF, AF, LR, lambda);
    auto estimatedNorm = computeNorm(xEstimatedNew);
    if (estimatedNorm > 0) {
      diff = computeNorm(xEstimatedNew - xEstimated) / estimatedNorm;
      xEstimated = xEstimatedNew;
    } else {
      result = false;
    }
    iteration++;
  }

  double constraint_norm, residual_norm;
  if (result) {
    constraint_norm = computeNorm(L * xEstimated, q);
    residual_norm = computeNorm(y - A * xEstimated, p);
  }

  return std::make_tuple(result, xEstimated, constraint_norm, residual_norm);
}
} // namespace regu