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

  Mat regularizationMatrix, forwardMatrix;
  Vec timeInst, measurement, trueData;

  auto ret =
      regularizationMatrix.load("../test_data_3/D1.mat", arma::raw_ascii);
  ret = forwardMatrix.load("../test_data_3/G.mat", arma::raw_ascii);
  ret = timeInst.load("../test_data_3/t.Mat", arma::raw_ascii);
  ret =
      measurement.load("../test_data_3/noisy_observation.mat", arma::raw_ascii);
  ret = trueData.load("../test_data_3/m_true.mat", arma::raw_ascii);

  auto h1 = matplot::figure();
  matplot::plot(timeInst, measurement)->line_width(2);
  matplot::title("Noisy measurement");
  matplot::xlabel("time");
  matplot::ylabel("mV");

  auto h2 = matplot::figure();
  matplot::plot(timeInst, trueData)->line_width(2);
  matplot::title("True data & Estimations");
  matplot::xlabel("time");
  matplot::ylabel("mV");

  matplot::hold(matplot::on);

  double pArray[] = {2, 2};
  double qArray[] = {2,  1};
  double lambdaArray[] = {6,  3};

  for (int i = 0; i < 2; i++) {
    std::tuple<bool, Vec, double, double> ret =
        solve(measurement, forwardMatrix, regularizationMatrix, lambdaArray[i],
              pArray[i], qArray[i]);
    Vec estimation = std::get<1>(ret);
    matplot::plot(timeInst, estimation)->line_width(2);
  }
  ::matplot::legend({"True", "L2L2",  "L2L1"});
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
