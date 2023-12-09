#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include "ILinearAlgebra.hpp"

namespace sci {

struct KalmanOutput {
  Vec stateEstimation;
  Mat estimationCovariances;
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
   * @brief Kalman Filter object
   *
   * @param F state stransition matrix
   * @param B input control matrix
   * @param H observation matrix
   * @param Q process noise covariance matrix
   * @param R measurement noise covariance matrix
   * @param P0 initial estimate covariance
   * @param x0 initial state
   */
  KalmanFilter(const Mat &F, const Mat &B, const Mat &H, const Mat &Q,
               const Mat &R, const Mat &P0, const Vec &x0);
  virtual ~KalmanFilter();
  /**
   * @brief Predict sate and estimation uncertanity
   *
   * @param measurement
   * @param input
   * @return std::tuple<Result, KalmanOutput>
   */
  std::tuple<Result, KalmanOutput> predict(const Vec &measurement,
                                           const Vec &input);

  /**
   * @brief Check input consistency
   * 
   * @return true 
   * @return false 
   */
  bool checkConsistency();
protected:
  /**
   * @brief Correction with measurement. Update gain, state and uncertanity
   * (corection step)
   *
   */
  void update(const Vec &measurement, const Vec &prioriStateEstimate,
              const Mat &prioriStateCovariance);

private:
  KalmanFilter(const KalmanFilter &other) = delete;
  KalmanFilter(const KalmanFilter &&other) = delete;
  KalmanFilter &operator=(const KalmanFilter &other) = delete;

private:
  Mat stateTransitionOperator;
  Mat controlInputOperator;
  Mat observationOperator;
  Mat processCovariance;
  Mat measurementCovariance;
  Mat kalmanGain;
  Mat estimationCovariance;
  Vec estimatedState;
  int nOfState;
  int nOfObservedState;
};

} // namespace sci
#endif