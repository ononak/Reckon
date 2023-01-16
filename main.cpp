//
//  main.cpp
//  ToolRegu
//
//  Created by Önder Nazım Onak on 30.10.2021.
//

#include <cmath>
#include <iostream>

#include "LpLqRegulizer.hpp"
#include <chrono>

#include "gnuplot-cpp/gnuplot_i.hpp"
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace std::chrono;
using namespace regu;

void test_1() {

  std::cout << "Tes 1 runnings" << std::endl;

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

  std::unique_ptr<Regularization> tik = std::make_unique<LpLqRegulizer>(2, .2);
  std::unique_ptr<Regularization> lp = std::make_unique<LpLqRegulizer>(2, 1);

  Mat L;
  int sz = (int)forward_matrix.n_cols;
  L.eye(sz, sz);
  std::tuple<Vec, double, double> ret2 =
      tik->solve(measurement, forward_matrix, L, 0.01);
  Vec l2_estimation = std::get<0>(ret2);

  Gnuplot pl3("lines");
  pl3.plot_xy(time_inst, l2_estimation, "Tikhonov estimation");
  pl3.set_xlabel("Time(s)");
  pl3.set_ylabel("Seismic Reflector Amplitude");

  std::tuple<Vec, double, double> ret1 =
      lp->solve(measurement, forward_matrix, L, 0.0032);
  Vec l1_estimation = std::get<0>(ret1);

  Gnuplot pl4("lines");
  pl4.plot_xy(time_inst, l1_estimation, "L1 estimation");
  pl4.set_xlabel("Time(s)");
  pl4.set_ylabel("Seismic Reflector Amplitude");
}

void test_2() {

  std::cout << "Tes 2 runnings" << std::endl;

  Mat A, L, Y;

  auto ret = A.load("../test_data_2/A.mat", arma::raw_ascii);
  ret = L.load("../test_data_2/L.mat", arma::raw_ascii);
  ret = Y.load("../test_data_2/y.mat", arma::raw_ascii);

  Mat x_estimated_Tikh(A.n_cols, Y.n_cols, arma::fill::zeros);
  Mat x_estimated_TV(A.n_cols, Y.n_cols, arma::fill::zeros);

  std::unique_ptr<Regularization> tik = std::make_unique<LpLqRegulizer>(2, 2);
  std::unique_ptr<Regularization> lp = std::make_unique<LpLqRegulizer>(2, 1);

  for (int i = 10; i < 11; i++) {

    Vec y = Y.col(i);
    std::tuple<Vec, double, double> retTk = tik->solve(y, A, L, 0.02);
    std::tuple<Vec, double, double> retTV = lp->solve(y, A, L, 0.001);
    x_estimated_Tikh.col(i) = std::get<0>(retTk);
    x_estimated_TV.col(i) = std::get<0>(retTV);
    Vec x2 = std::get<0>(retTk);
    Vec x1 = std::get<0>(retTV);

    Gnuplot pl1("lines");
    pl1.plot_x(x2);
    pl1.plot_x(x1);
  }
}

void test_3() {

  std::cout << "Tes 3 runnings" << std::endl;

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

  std::unique_ptr<Regularization> tik = std::make_unique<LpLqRegulizer>(2, 2);
  std::unique_ptr<Regularization> lp = std::make_unique<LpLqRegulizer>(2, 1);

  std::tuple<Vec, double, double> ret2 =
      tik->solve(measurement, forward_matrix, regularization_matrix, 10);
  Vec l2_estimation = std::get<0>(ret2);

  Gnuplot pl2("lines");
  pl2.plot_xy(time_inst, true_data, "True Model");
  pl2.plot_xy(time_inst, l2_estimation, "1th-order Tikhonov");
  pl2.set_xlabel("Time(s)");
  pl2.set_ylabel("Denoised signal");

  std::tuple<Vec, double, double> ret1 =
      lp->solve(measurement, forward_matrix, regularization_matrix, 3);
  Vec l1_estimation = std::get<0>(ret1);

  pl2.plot_xy(time_inst, l1_estimation, "Total Variation");
}

int main(int argc, const char *argv[]) {

  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;

  Gnuplot::set_terminal_std("qt");

  test_1();
  std::cout << "Will continue to run in 10 seconds" << std::endl;
  sleep(10);

  test_2();
  std::cout << "Will continue to run in 10 seconds" << std::endl;
  sleep(10);

  test_3();
  std::cout << "Will continue to run in 10 seconds" << std::endl;
  sleep(10);

  return 0;
}
