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

void test_1() {

 std::cout << "LpLq p = 2 q = 0.2 regularization runnings" << std::endl;

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
  pl1.plot_xy(time_inst, true_data, "True Model");

  pl2.set_title("Noisy measurements");
  pl2.plot_xy(time_inst, measurement, "Noisy measurements");
  pl2.set_xlabel("Time (s)");
  pl2.set_ylabel("Seismic Amplitude");

  Mat L;
  int sz = (int)forward_matrix.n_cols;
  L.eye(sz, sz);
  std::tuple<Vec, double, double> ret2 = solve(measurement, forward_matrix, L, 0.01,2,0.2);
  Vec l2_estimation = std::get<0>(ret2);

  Gnuplot pl3("lines");
  pl3.plot_xy(time_inst, l2_estimation, "L(2)L(0.2) estimation");
  pl3.set_xlabel("Time(s)");
  pl3.set_ylabel("Seismic Reflector Amplitude");

  sleep(10);
}


int main(int argc, const char *argv[]) {

  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;
  Gnuplot::set_terminal_std("qt");
  test_1();

  return 0;
}
