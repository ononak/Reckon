#include "../src/Regularization.hpp"
#include <gtest/gtest.h>

using namespace regu;
;
TEST(RegularizationTest, InputCompatiplityTestCase) {

  Vec measurements;
  Mat transferMatrix, regularizationMatrix;

  measurements.zeros(100);
  transferMatrix.eye(100, 100);
  regularizationMatrix.eye(100, 100);

  {
    auto [ret, estimate] =
        solve(measurements, transferMatrix, regularizationMatrix, 1);
    EXPECT_TRUE(ret);
  }

  measurements.resize(10);
  {
    auto [ret, estimate] =
        solve(measurements, transferMatrix, regularizationMatrix, 1);
    EXPECT_FALSE(ret);
  }

  measurements.resize(100);
  regularizationMatrix.resize(100, 90);
  {
    auto [ret, estimate] =
        solve(measurements, transferMatrix, regularizationMatrix, 1);
    EXPECT_FALSE(ret);
  }
}

TEST(RegularizationTest, SimpleEstimationTestCase) {
  Vec measurements;
  Mat transferMatrix, regularizationMatrix;

  measurements.ones(100);
  transferMatrix.eye(100, 100);
  regularizationMatrix.eye(100, 100);

  {
    auto [ret, estimate] =
        solve(measurements, transferMatrix, regularizationMatrix, 0);
    EXPECT_TRUE(ret);
    EXPECT_TRUE(measurements == estimate);
  }

  {
    auto [ret, estimate] =
        solve(measurements, transferMatrix, regularizationMatrix, 10);
    EXPECT_TRUE(ret);
    EXPECT_FALSE(measurements == estimate);
  }

  {
    double multiplier = 5.0;
    auto [ret, estimate] =
        solve(measurements, multiplier * transferMatrix, regularizationMatrix, 0);
    EXPECT_TRUE(ret);
    Vec expected = measurements / multiplier ;
    EXPECT_TRUE(expected == estimate);
  }
}