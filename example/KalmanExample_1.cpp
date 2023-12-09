#include "../src/KalmanFilter.hpp"
#include <matplot/matplot.h>
using namespace sci;

struct kalmanTestData {

  const int numberOfSamples = 100;
  const int stateDimension = 2;
  const int observationDimension = 1;

  /**
   * linear system chareacteristics
   * x(k+1) = Ax(k) + Bu + w(k)
   * y(k+1) = Hx(k+1) + v(k)
   */
  double varProcessNoise = 0.1;
  double stdProcessNoise = sqrt(varProcessNoise);
  double muProcessNoise = 0;
  double varObservationNoise = 2;
  double stdObservationNoise = sqrt(varObservationNoise);
  double muObservationNoise = 0;

  Mat A = {{1.9223, -0.9604}, {1, 0}};
  Vec B = {0, 0};
  Mat H = {1, 0};
  Mat R = {varObservationNoise};
  Mat Q = {{varProcessNoise, 0}, {0, varProcessNoise}};
  Vec x0 = {3, -3};
  Mat P0 = {{1, 0}, {0, 1}};
  Vec u = {0};
};

void test() {

  kalmanTestData testData;

  std::cout << "Simulate real system states and observations " << std::endl;

  std::cout << "Generating Process and Observation noise" << std::endl;
  Mat observationNoise =
      arma::randn(testData.observationDimension, testData.numberOfSamples,
                  arma::distr_param(testData.muObservationNoise,
                                    testData.stdObservationNoise));
  Mat processNoise = arma::randn(
      testData.stateDimension, testData.numberOfSamples,
      arma::distr_param(testData.muProcessNoise, testData.stdProcessNoise));

  observationNoise.save("obs.mat", arma::raw_ascii);
  processNoise.save("pros.mat", arma::raw_ascii);

  std::cout << "COMPLETED" << std::endl;

  std::cout << "Calculating real system states and measurements " << std::endl;

  Mat X(testData.stateDimension, testData.numberOfSamples, arma::fill::zeros);
  Mat Y(testData.observationDimension, testData.numberOfSamples,
        arma::fill::zeros);

  X.col(0) = {0.1, 0.5};
  Y.col(0) = testData.H * X.col(0);

  for (int i = 1; i < testData.numberOfSamples; i++) {
    X.col(i) = testData.A * X.col(i - 1) + processNoise.col(i);
    Y.col(i) = testData.H * X.col(i - 1) + observationNoise.col(i);
  }
  std::cout << "COMPLETED" << std::endl;
  std::cout << "Kalman filtering " << std::endl;

  Mat Xe(testData.stateDimension, testData.numberOfSamples + 1,
         arma::fill::zeros);
  Xe.col(0) = testData.x0;
  Mat Ye(testData.observationDimension, testData.numberOfSamples,
         arma::fill::zeros);

  KalmanFilter tracker(testData.A, testData.B, testData.H, testData.Q,
                       testData.R, testData.P0, testData.x0);
  for (int i = 1; i < testData.numberOfSamples; i++) {

    std::cout << ". ";
    auto [ret, output] = tracker.predict(Y.col(i), testData.u);
    Xe.col(i) = output.stateEstimation;
    Ye.col(i - 1) = testData.H * Xe.col(i);

    if(i%25 == 0)
      std::cout << std::endl;
  }
  std::cout << " OK" << std::endl;

  // Plot results
  auto h1 = matplot::figure();
  matplot::title("Real observation & Estimated observation");
  matplot::hold(matplot::on);
  matplot::plot(Y)->line_width(2);
  matplot::plot(Ye)->line_width(2);
  ::matplot::legend({"Real", "Estimate"});

  auto h2 = matplot::figure();
  matplot::title("Real state 1 & Estimated state 1");
  matplot::hold(matplot::on);
  Mat xtrue_0 = X.row(0).t();
  Mat xest_0 = Xe.row(0).t();
  matplot::plot(xtrue_0)->line_width(2);
  matplot::plot(xest_0)->line_width(2);
  ::matplot::legend({"Real", "Estimate"});

  auto h3 = matplot::figure();
  matplot::title("Real state 2 & Estimated state 2");
  matplot::hold(matplot::on);
  Mat xtrue_1 = X.row(1).t();
  Mat xest_1 = Xe.row(1).t();
  matplot::plot(xtrue_1)->line_width(2);
  matplot::plot(xest_1)->line_width(2);
  ::matplot::legend({"Real", "Estimate"});

  char key;
  std::cin >> key;
}

int main(int argc, const char *argv[]) {

  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;
  std::cout << "Kalman Filtering......: " << tmp << std::endl;
  test();
  return 0;
}
