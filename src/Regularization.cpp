#include "Regularization.hpp"
#include <utility>

namespace sci {

LpLqRegularization::LpLqRegularization(Mat &forwardM, Mat &regularizationM)
    : forwardMatrix(forwardM), regularizationMatrix(regularizationM) {}

LpLqRegularization::LpLqRegularization(const LpLqRegularization &other) {
  this->forwardMatrix = other.forwardMatrix;
  this->regularizationMatrix = other.regularizationMatrix;
}

LpLqRegularization::LpLqRegularization(LpLqRegularization &&other)
    : forwardMatrix(std::move(other.forwardMatrix)),
      regularizationMatrix(std::move(other.regularizationMatrix)) {}

LpLqRegularization &
LpLqRegularization::operator=(const LpLqRegularization &other) {
  if (this != &other) {
    this->forwardMatrix = other.forwardMatrix;
    this->regularizationMatrix = other.regularizationMatrix;
  }
  return *this;
}

std::tuple<Result, Vec, double, double>
LpLqRegularization::solve(Vec &y, double lambda, double p, double q) {
  int iteration = 0;
  double diff = 1e10;

  // initial estimation ıs L2L2 solution
  auto [result, xEstimated] =
      this->solveTikhonov(y, forwardMatrix, regularizationMatrix, lambda);

  // iterate solution till converge
  while ((iteration <= MAX_ITER) && (diff >= tolerance) &&
         (result == Result::OK)) {
    // Compute weight Matrices
    auto v = makeAbs(forwardMatrix * xEstimated - y);
    auto z = makeAbs(regularizationMatrix * xEstimated);

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
    Mat AF = WF * forwardMatrix;
    Mat LR = WR * regularizationMatrix;
    Vec bF = WF * y;

    auto [result, xEstimatedNew] = solveTikhonov(bF, AF, LR, lambda);
    auto estimatedNorm = computeNorm(xEstimatedNew);
    if (estimatedNorm > 0) {
      diff = computeNorm(xEstimatedNew - xEstimated) / estimatedNorm;
      xEstimated = xEstimatedNew;
    } else {
      result = Result::NOK;
    }
    iteration++;
  }

  double constraint_norm, residual_norm;
  if (result == Result::OK) {
    constraint_norm = computeNorm(regularizationMatrix * xEstimated, q);
    residual_norm = computeNorm(y - forwardMatrix * xEstimated, p);
  }

  return std::make_tuple(result, xEstimated, constraint_norm, residual_norm);
}

std::tuple<Result, Vec>
LpLqRegularization::solveTikhonov(Vec &y, Mat &fm, Mat &rm, double lambda) {
  Result result = Result::OK;
  Vec xEstimated;
  double lambdasqr = lambda * lambda;

  if ((fm.n_cols != rm.n_cols) || (y.n_rows != fm.n_rows)) {
    result = Result::NOK;
  }

  if (result == Result::OK) {
    // construct augmented matrix [A;L]
    auto AugA = joinVertical(fm, rm);

    // compute qr decomposition
    Mat Q, R;
    auto qrResult = computeQrEconDecomposition(Q, R, AugA);
    if (qrResult) {
      Mat Q1, U1, V1;
      Vec s1;
      Q1 = Q.rows(0, fm.n_rows - 1);
      if (!computeSvd(U1, s1, V1, Q1)) {
        result = Result::UNEXPECTED_ERROR;
      }

      if (result == Result::OK) {
        // solve the system of linear equation to find the unknown vector x
        auto S1 = diagonalize(s1);
        S1.reshape(fm.n_rows, fm.n_cols);
        Vec rhs = S1.t() * U1.t() * y;
        auto In = makeEye(fm.n_cols, fm.n_cols);
        Mat Lhs = (lambdasqr * In + (1 - lambdasqr) * S1.t() * S1);

        try {
          auto b = solveLinearSystem(Lhs, rhs);
          auto xlhs = (V1.t() * R);
          xEstimated = solveLinearSystem(xlhs, b);
        } catch (...) {
          result = Result::UNEXPECTED_ERROR;
        }
      }
    }
  }
  return std::tuple(result, xEstimated);
}

} // namespace sci