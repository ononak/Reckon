#include "KalmanFilter.hpp"
#include "ILinearAlgebra.hpp"
#include "SciTool.hpp"
#include <iostream>

namespace sci {

KalmanFilter::KalmanFilter(const Mat &stateTransitionOperator,
                           const Mat &controlInputOperator,
                           const Mat &observationOperator,
                           const Mat &processCovariance,
                           const Mat &measurementCovariance,
                           const Mat &initialEstimationCovariance,
                           const Vec &x0)
    : F(stateTransitionOperator), B(controlInputOperator),
      H(observationOperator), Q(processCovariance), R(measurementCovariance),
      P(initialEstimationCovariance), Xestimated(x0) {

  sizeX = F.n_rows;
  sizeY = H.n_rows;
  I = makeEye(sizeX, sizeX);
  K.eye(sizeX, sizeY);
}

KalmanFilter::~KalmanFilter() {}

std::tuple<Result, KalmanOutput> KalmanFilter::predict(const Vec &y,
                                                       const Vec &u) {
  Result ret = Result::NOK;
  Vec yfiltered;
  Mat yP;

  if ((B.n_cols == u.n_rows) && (sizeY == y.n_rows)) {

    try {
      measurementUpdate(y);
      yfiltered = H * Xestimated; // filtered
      yP = H * P * H.t();         // error covariance

      timeUpdate(u);
      ret = Result::OK;

    } catch (...) {
      ret = Result::UNEXPECTED_ERROR;
    }
  }
  return std::tuple<Result, KalmanOutput>(ret, {Xestimated, P, yfiltered, yP});
}

void KalmanFilter::measurementUpdate(const Vec &measurement) {

  Mat prefitCovariances = H * P * H.t() + R;
  K = P * H.t() * inverse(prefitCovariances);
  Xestimated += K * (measurement - H * Xestimated);
  P = (I - K * H) * P;
}
void KalmanFilter::timeUpdate(const Vec &u) {

  Xestimated = F * Xestimated + B * u;
  P = F * P * F.t() + Q;
}

} // namespace sci