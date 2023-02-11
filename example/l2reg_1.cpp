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

#include "gnuplot-cpp/gnuplot_i.hpp"
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace std::chrono;
using namespace regu;

void test() {

  std::cout << "L2 regularization runnings" << std::endl;

  Mat regularization_matrix, forward_matrix;
  Vec time_inst, measurement, true_data;

  auto ret =
      regularization_matrix.load("../test_data_3/D1.mat", arma::raw_ascii);
  ret = forward_matrix.load("../test_data_3/G.mat", arma::raw_ascii);
  ret = time_inst.load("../test_data_3/t.Mat", arma::raw_ascii);
  ret =
      measurement.load("../test_data_3/noisy_observation.mat", arma::raw_ascii);
  ret = true_data.load("../test_data_3/m_true.mat", arma::raw_ascii);

  Gnuplot pl1("lines");

  pl1.set_title("Noisy measurements");
  pl1.plot_xy(time_inst, measurement, "Noisy measurements");
  pl1.set_xlabel("Time (s)");
  pl1.set_ylabel("nm(t)");

  std::tuple<Vec, double, double> ret2 =
      solve(measurement, forward_matrix, regularization_matrix, 10, 2, 2);
  Vec l2_estimation = std::get<0>(ret2);

  Gnuplot pl2("lines");
  pl2.plot_xy(time_inst, true_data, "True Model");
  pl2.plot_xy(time_inst, l2_estimation, "1th-order Tikhonov");
  pl2.set_xlabel("Time(s)");
  pl2.set_ylabel("Denoised signal");
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
