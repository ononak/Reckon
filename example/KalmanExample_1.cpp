#include "../src/KalmanFilter.hpp"
#include <matplot/matplot.h>
using namespace sci;
struct LinearSystem {

  const int nSample = 200;
  const int sizeX = 2;
  const int sizeY = 1;
  /**
   * linear system chareacteristics
   * x(k+1) = Ax(k) + Bu + w(k)
   * y(k+1) = Hx(k+1) + v(k)
   */
  double varNoiseProcess = .1;
  double stdNoiseProcess = sqrt(varNoiseProcess);
  double muNoiseProcess = 0;
  double varNoiseY = 2;
  double stdNoiseY = sqrt(varNoiseY);
  double munoiseY = 0;

  Mat A = {{1.9223, -0.9604}, {1, 0}};
  Vec B = {0, 0};
  Mat H = {.5, .3};
  Mat R = {varNoiseY};
  Mat Q = {{varNoiseProcess, 0}, {0, varNoiseProcess}};
  Vec x0 = {.1, .5};
  Mat P0 = {{1, 0}, {0, 1}};
  Vec u = {0};
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

  std::cout << "Generating Process and Observation noise" << std::endl;
  // Mat noiseY;
  // noiseY.load("onoise.mat",arma::raw_ascii);

  Mat noiseY =
      arma::randn(lsystem.sizeY, lsystem.nSample,
                  arma::distr_param(lsystem.munoiseY, lsystem.stdNoiseY));
  // Mat noiseProcess;
  // noiseProcess.load("pnoise.mat",arma::raw_ascii);
  Mat noiseProcess = arma::randn(
      lsystem.sizeX, lsystem.nSample,
      arma::distr_param(lsystem.muNoiseProcess, lsystem.stdNoiseProcess));

  std::cout << "Calculating real system states and measurements " << std::endl;

  Mat X(lsystem.sizeX, lsystem.nSample, arma::fill::zeros);
  Mat Y(lsystem.sizeY, lsystem.nSample, arma::fill::zeros);
  Mat Ynoisy(lsystem.sizeY, lsystem.nSample, arma::fill::zeros);

  X.col(0) = {0.1, 0.5};
  Y.col(0) = lsystem.H * X.col(0);

  for (int i = 1; i < lsystem.nSample; i++) {
    X.col(i) = lsystem.A * X.col(i - 1) + noiseProcess.col(i);
    Y.col(i) = lsystem.H * X.col(i - 1);
    Ynoisy.col(i) = Y.col(i) + noiseY.col(i);
  }

  std::cout << "Kalman filtering " << std::endl;

  Vec yfiltered(lsystem.nSample, arma::fill::zeros);
  Vec yCov(lsystem.nSample, arma::fill::zeros);
  KalmanFilter tracker(lsystem.A, lsystem.B, lsystem.H, lsystem.Q, lsystem.R,
                       lsystem.P0, lsystem.x0);
  for (int i = 0; i < lsystem.nSample; i++) {

    std::cout << ". ";
    auto [ret, output] = tracker.predict(Ynoisy.col(i), lsystem.u);
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
