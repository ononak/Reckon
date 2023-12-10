#include "../src/KalmanFilter.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace sci;

class KalmanTest : public testing::Test {
protected:
  void SetUp() override {

    x0 = Vec(2);
    x0(1) = 1;
    x0(1) = 1;

    F = 2 * makeEye(2, 2);
    B = Mat(2, 2, arma::fill::zeros);
    H = makeEye(2, 2);
    Q = makeEye(2, 2);
    R = makeEye(2, 2);
    P0 = makeEye(2, 2);
    x0 = Vec(2, arma::fill::ones);
    u = Vec(2, arma::fill::zeros);
    y = Vec(2, arma::fill::ones);

    Kf = std::make_unique<KalmanFilter>(F, B, H, Q, R, P0, x0);
    KfInconsistend =
        std::make_unique<KalmanFilter>(makeEye(3, 2), B, H, Q, R, P0, x0);
  };

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  ~KalmanTest() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  Mat F, B, H, Q, R, P0;
  Vec x0, u, y;
  std::unique_ptr<KalmanFilter> Kf, KfInconsistend;
};

TEST_F(KalmanTest, TestInconsistendLinearSystem) {

  auto result = KfInconsistend->predict(y, u);
  auto retval = std::get<0>(result);
  EXPECT_EQ(Result::UNEXPECTED_ERROR, retval);
}

TEST_F(KalmanTest, TestPrediction) {

  auto result = Kf->predict(y, u);
  auto retval = std::get<0>(result);
  EXPECT_EQ(Result::OK, retval);
}