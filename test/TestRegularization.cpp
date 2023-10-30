#include "../src/Regularization.hpp"
#include <gtest/gtest.h>

using namespace sci;
;
TEST(RegularizationTest, InputCompatiplityTestCase) {

  Vec measurements;
  Mat transferMatrix, regularizationMatrix;

  measurements.zeros(100);
  transferMatrix.eye(100, 100);
  regularizationMatrix.eye(100, 100);
  LpLqRegularization regularization(transferMatrix, regularizationMatrix);

  {
    auto [ret, estimate, res1, res2] =
        regularization.solve(measurements, 1, 2, 2);
    EXPECT_EQ(ret, Result::OK);
  }

  measurements.resize(10);
  {
    auto [ret, estimate, res1, res2] =
        regularization.solve(measurements, 1, 2, 2);
    EXPECT_EQ(ret, Result::NOK);
  }

  measurements.resize(100);
  regularizationMatrix.resize(100, 90);
  LpLqRegularization otherRegularization(transferMatrix, regularizationMatrix);
  {
    auto [ret, estimate, re1, res2] =
        otherRegularization.solve(measurements, 1);
    EXPECT_EQ(ret, Result::NOK);
  }
}

TEST(RegularizationTest, SimpleEstimationTestCase) {
  Vec measurements;
  Mat transferMatrix, regularizationMatrix;

  measurements.ones(100);
  transferMatrix.eye(100, 100);
  regularizationMatrix.eye(100, 100);
  LpLqRegularization regularization(transferMatrix, regularizationMatrix);
  {
    auto [ret, estimate, res1, res2] = regularization.solve(measurements, 0);
    EXPECT_EQ(ret, Result::OK);
    EXPECT_TRUE(measurements == estimate);
  }

  {
    auto [ret, estimate, reg1, reg2] = regularization.solve(measurements, 10);
    EXPECT_EQ(ret, Result::OK);
    EXPECT_FALSE(measurements == estimate);
  }

  {
    double multiplier = 5.0;
    Mat newTransferMatrix = multiplier * transferMatrix;
    LpLqRegularization otherRegularization(newTransferMatrix,
                                           regularizationMatrix);
    auto [ret, estimate, reg1, reg2] =
        otherRegularization.solve(measurements, 0);
    EXPECT_EQ(ret, Result::OK);
    Vec expected = measurements / multiplier;
    EXPECT_TRUE(expected == estimate);
  }
}