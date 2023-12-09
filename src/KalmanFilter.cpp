#include "KalmanFilter.hpp"
#include "ILinearAlgebra.hpp"
#include "SciTool.hpp"

namespace sci {

KalmanFilter::KalmanFilter(const Mat &F, const Mat &B, const Mat &H,
                           const Mat &Q, const Mat &R, const Mat &P0,
                           const Vec &x0)
    : stateTransitionOperator(F), controlInputOperator(B),
      observationOperator(H), processCovariance(Q), measurementCovariance(R),
      estimationCovariance(P0), estimatedState(x0) {

  nOfState = stateTransitionOperator.n_rows;
  nOfObservedState = observationOperator.n_rows;
  kalmanGain.eye(nOfState, nOfObservedState);
}

KalmanFilter::~KalmanFilter() {}

bool KalmanFilter::checkConsistency() {

  // check consistency with state size
  if ((nOfState != stateTransitionOperator.n_cols) ||
      (nOfState != controlInputOperator.n_rows) ||
      (nOfState != estimationCovariance.n_rows) ||
      (nOfState != estimationCovariance.n_cols) ||
      (nOfState != processCovariance.n_rows) ||
      (nOfState != processCovariance.n_cols) ||
      (nOfState != observationOperator.n_cols) ||
      (nOfState != kalmanGain.n_rows)) {
    return false;
  }

  // check consistency with measurement
  if ((nOfObservedState != measurementCovariance.n_cols) ||
      (nOfObservedState != observationOperator.n_rows) ||
      (nOfObservedState != kalmanGain.n_cols)) {
    return false;
  }

  return true;
}

std::tuple<Result, KalmanOutput> KalmanFilter::predict(const Vec &measurement,
                                                       const Vec &input) {
  Result ret = Result::NOK;
  if ((controlInputOperator.n_cols == input.n_rows) &&
      (nOfObservedState == measurement.n_rows)) {

    try {
      /*
       * priori state and uncertanity estimation
       */
      Vec xkkm1 = stateTransitionOperator * estimatedState +
                  controlInputOperator * input;
      Mat pkkm1 = stateTransitionOperator * estimationCovariance *
                      stateTransitionOperator.t() +
                  processCovariance;

      /*Correction with measurement*/
      update(measurement, xkkm1, pkkm1);
      ret = Result::OK;

    } catch (...) {
      ret = Result::UNEXPECTED_ERROR;
    }
  }
  return std::tuple<Result, KalmanOutput>(
      ret, {estimatedState, estimationCovariance});
}

void KalmanFilter::update(const Vec &measurement,
                          const Vec &prioriStateEstimate,
                          const Mat &prioriStateCovariance) {

  Vec prefitResiduals = measurement - observationOperator * prioriStateEstimate;
  Mat prefitCovariances =
      observationOperator * prioriStateCovariance * observationOperator.t() +
      measurementCovariance;
  kalmanGain = prioriStateCovariance * observationOperator.t() *
               inverse(prefitCovariances);
  estimatedState = prioriStateEstimate + kalmanGain * prefitResiduals;
  estimationCovariance =
      (makeEye(nOfState, nOfState) - kalmanGain * observationOperator) *
      estimationCovariance;
}

} // namespace sci