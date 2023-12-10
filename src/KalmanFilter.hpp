#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include "ILinearAlgebra.hpp"

namespace sci {

struct KalmanOutput {
  Vec X; // estimated state
  Mat P; // state covariance
  Vec Y; // filtered measurement
  Mat EP; // Error covariance
};

/**
 * @brief Linear Kalman filter
 *
 * state vector - > x_k = F*x_k-1 + B*u_k + w_k  where w_k = N(0, Q)
 * measurement vector -> y_k = H*y_k-1 + v_k where v_k = N(0,R)
 */
class KalmanFilter {

public:
  /**
   * @brief Construct a new Kalman Filter object
   *
   * @param stateTransitionOperator  F
   * @param controlInputOperator   B
   * @param observationOperator  H
   * @param processCovariance  Q
   * @param measurementCovariance R
   * @param initialEstimationCovariance P0
   * @param initialState x0
   */
  KalmanFilter(const Mat &stateTransitionOperator,
               const Mat &controlInputOperator, const Mat &observationOperator,
               const Mat &processCovariance, const Mat &measurementCovariance,
               const Mat &initialEstimationCovariance, const Vec &initialState);

  virtual ~KalmanFilter();

  /**
   * @brief Predict sate and estimation uncertanity
   * @param measurement y
   * @param input u
   * @return std::tuple<Result, KalmanOutput>
   */
  std::tuple<Result, KalmanOutput> predict(const Vec &y, const Vec &u);

protected:
void measurementUpdate(const Vec &measurement);
void timeUpdate(const Vec &u);

private:
  KalmanFilter(const KalmanFilter &other) = delete;
  KalmanFilter(const KalmanFilter &&other) = delete;
  KalmanFilter &operator=(const KalmanFilter &other) = delete;

private:
  const Mat F;
  const Mat B;
  const Mat H;
  const Mat Q;
  const Mat R;
  Mat I;
  Mat K; // Kalman gain
  Mat P; // estimation covariance
  Vec Xestimated;
  int sizeX;
  int sizeY;
};

} // namespace sci
#endif