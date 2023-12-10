#include "../src/KalmanFilter.hpp"
#include <matplot/matplot.h>
using namespace sci;
struct LinearSystem {

  const int nSample = 100;
  const int sizeX = 3;
  const int sizeY = 1;
  /**
   * linear system chareacteristics
   * x(k+1) = Ax(k) + Bu + w(k)
   * y(k+1) = Hx(k+1) + v(k)
   */
  double varNoiseProcess = 0.001;
  double stdNoiseProcess = sqrt(varNoiseProcess);
  double muNoiseProcess = 0;
  double varNoiseY = 1;
  double stdNoiseY = sqrt(varNoiseY);
  double munoiseY = 0;

  Mat A = {{1.1269, -0.4940, 0.1129}, {1, 0, 0}, {0, 1, 0}};
  Vec B = {-0.3832, 0.5919, 0.5191};
  Mat H = {1, 0, 0};
  Mat R = {varNoiseY};
  Mat Q = {{varNoiseProcess, 0, 0},
           {0, varNoiseProcess, 0},
           {0, 0, varNoiseProcess}};
  Vec x0 = {0, 0, 0};
  Mat P0 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
};

void plotResults(const Vec y, const Vec ynoisy, const Vec yfiltered,
                 const Vec &yCov) {
  // Plot results
  auto h1 = matplot::figure();
  matplot::title("System output Y");
  matplot::hold(matplot::on);
  matplot::plot(y)->line_width(2);
  matplot::plot(ynoisy)->line_width(2);
  matplot::plot(yfiltered)->line_width(2);
  ::matplot::legend({"True", "Real observation", "Filtered observation"});

  auto h2 = matplot::figure();
  matplot::title("Error covariance");
  matplot::hold(matplot::on);
  Vec errorTrue = y - ynoisy;
  Vec errorFiltered = y - yfiltered;

  matplot::plot(errorTrue)->line_width(2);
  matplot::plot(errorFiltered)->line_width(2);
  ::matplot::legend({"Error before Kalman", "Error after Kalman"});

  auto h3 = matplot::figure();
  matplot::title("Error covariance");
  matplot::hold(matplot::on);
  matplot::plot(yCov)->line_width(2);

  char key;
  std::cin >> key;
}

void test() {

  LinearSystem lsystem;

  // measurement noise
  Mat noiseY =
      arma::randn(lsystem.sizeY, lsystem.nSample,
                  arma::distr_param(lsystem.munoiseY, lsystem.stdNoiseY));

  // process noise
  Mat noiseProcess = arma::randn(
      lsystem.sizeX, lsystem.nSample,
      arma::distr_param(lsystem.muNoiseProcess, lsystem.stdNoiseProcess));

  // Prepare Real system data
  Mat X(lsystem.sizeX, lsystem.nSample, arma::fill::zeros);
  Mat Y(lsystem.sizeY, lsystem.nSample, arma::fill::zeros);
  Mat Ynoisy(lsystem.sizeY, lsystem.nSample, arma::fill::zeros);
  Mat U(1, lsystem.nSample, arma::fill::zeros);

  X.col(0) = lsystem.x0;
  Y.col(0) = lsystem.H * X.col(0);
  Ynoisy.col(0) = Y.col(0) + noiseY.col(0);

  for (int i = 1; i < lsystem.nSample; i++) {
    U.col(i) = sin(i / 5);
    X.col(i) =
        lsystem.A * X.col(i - 1) + lsystem.B * U.col(i) + noiseProcess.col(i);
    Y.col(i) = lsystem.H * X.col(i - 1);
    Ynoisy.col(i) = Y.col(i) + noiseY.col(i);
  }

  // Kalman filtering
  Vec yfiltered(lsystem.nSample, arma::fill::zeros);
  Vec yCov(lsystem.nSample, arma::fill::zeros);
  KalmanFilter tracker(lsystem.A, lsystem.B, lsystem.H, lsystem.Q, lsystem.R,
                       lsystem.P0, lsystem.x0);
  for (int i = 0; i < lsystem.nSample; i++) {

    std::cout << ". ";
    auto [ret, output] = tracker.predict(Ynoisy.col(i), U.col(i));
    yfiltered(i) = output.Y(0, 0);
    yCov(i) = output.EP(0, 0);
    if (i % 50 == 0)
      std::cout << std::endl;
  }
  std::cout << " OK" << std::endl;

  plotResults(Y.row(0).t(), Ynoisy.row(0).t(), yfiltered, yCov);
}

int main(int argc, const char *argv[]) {

  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;
  std::cout << "Kalman Filtering......: " << tmp << std::endl;
  test();
  return 0;
}