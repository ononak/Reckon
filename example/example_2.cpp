//
//  main.cpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 30.10.2021.
//

#include "../src/Regularization.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <matplot/matplot.h>
#include <string>
#include <unistd.h>

using namespace std::chrono;
using namespace regu;

void test() {

  Mat forwardMatrix;
  Vec timeInst, measurement, trueData;

  /*Read test data*/
  auto ret = forwardMatrix.load("../test_data_1/fmat.mat", arma::raw_ascii);
  ret = timeInst.load("../test_data_1/time.mat", arma::raw_ascii);
  ret = measurement.load("../test_data_1/measurement.mat", arma::raw_ascii);
  ret = trueData.load("../test_data_1/true.mat", arma::raw_ascii);

  auto h1 = matplot::figure();
  matplot::plot(timeInst, measurement)->line_width(2);
  matplot::title("Noisy measurement");
  matplot::xlabel("time");
  matplot::ylabel("Seismic Reflector Amplitude");

  auto h2 = matplot::figure();
  matplot::plot(timeInst, trueData)->line_width(2);
  matplot::title("True model");
  matplot::xlabel("time");
  matplot::ylabel("Seismic Reflector Amplitude");

  /*Create default regularization matrix*/
  Mat regularizationMatrix;
  int sz = (int)forwardMatrix.n_cols;
  regularizationMatrix.eye(sz, sz);

  double pArray[] = {2, 2};
  double qArray[] = {2, 1.5};
  double lambdaArray[] = {0.32, 0.09};

  std::tuple<Vec, double, double> ret1 =
      solve(measurement, forwardMatrix, regularizationMatrix, 0.0032, 2, 1);
  Vec estimation = std::get<0>(ret1);
  auto h3 = matplot::figure();
  matplot::title("L2L1 model");
  matplot::plot(timeInst, estimation)->line_width(2);

  auto h4 = matplot::figure();
  matplot::hold(matplot::on);
  for (int i = 0; i < 2; i++) {
    std::tuple<Vec, double, double> ret2 =
        solve(measurement, forwardMatrix, regularizationMatrix, lambdaArray[i],
              pArray[i], qArray[i]);
    Vec estimation = std::get<0>(ret2);
    matplot::plot(timeInst, estimation)->line_width(2);
  }
  ::matplot::legend({"L2L2", "L2L1.5"});
  char key;
  std::cin >> key;
}

int main(int argc, const char *argv[]) {

  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;
  test();
  return 0;
}
