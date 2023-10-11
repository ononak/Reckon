//
//  main.cpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 30.10.2021.
//

#include <cmath>
#include <iostream>

#include "../src/Regularization.hpp"
#include <chrono>

#include "gnuplot_i.hpp"
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace std::chrono;
using namespace regu;

void test() {

  std::cout << "L1 regularization runnings" << std::endl;

  Mat forward_matrix;
  Vec time_inst, measurement, true_data;

  auto ret = forward_matrix.load("../test_data_1/fmat.mat", arma::raw_ascii);
  ret = time_inst.load("../test_data_1/time.mat", arma::raw_ascii);
  ret = measurement.load("../test_data_1/measurement.mat", arma::raw_ascii);
  ret = true_data.load("../test_data_1/true.mat", arma::raw_ascii);

  Gnuplot pl1("lines");
  Gnuplot pl2("lines");

  pl1.set_title("True Model");
  pl1.plot_xy(time_inst, true_data, "True Model");
  pl1.set_xlabel("Time (s)");
  pl1.set_ylabel("Seismic Reflector Amplitude");

  pl2.set_title("Noisy measurements");
  pl2.plot_xy(time_inst, measurement, "Noisy measurements");
  pl2.set_xlabel("Time (s)");
  pl2.set_ylabel("Seismic Amplitude");

  Mat L;
  int sz = (int)forward_matrix.n_cols;
  L.eye(sz, sz);

  std::tuple<Vec, double, double> ret1 =
      solve(measurement, forward_matrix, L, 0.0032, 2, 1);
  Vec l1_estimation = std::get<0>(ret1);

  Gnuplot pl4("lines");
  pl4.plot_xy(time_inst, l1_estimation, "L1 estimation");
  pl4.set_xlabel("Time(s)");
  pl4.set_ylabel("Seismic Reflector Amplitude");

  std::cout << "Will continue to run in 10 seconds" << std::endl;
  sleep(10);
}

int main(int argc, const char *argv[]) {

  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;

  Gnuplot::set_terminal_std("qt");
  test();
  return 0;
}
