#include "INNC/INNC.hpp"
#include "INNC/utils/utils.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>

std::int8_t data_i8_1[6] = {1, -3, -5, 7, -9, 11};
std::int16_t data_i16_1[6] = {0, -2, 4, 6, 8, -10};
std::string output_i8_1 = "[[1, -3, -5], [7, -9, 11]]";
std::int16_t data_i16_2[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

INNC::types all_type[6] = {INNC::i8,  INNC::i16, INNC::i32,
                           INNC::i64, INNC::f32, INNC::f64};

constexpr double epsilon = 1e-6;

#define ASSERT_STRICT_APPROX(lhs, rhs)                                         \
  static_assert(std::is_same_v<decltype(lhs), INNC::Tensor> &&                 \
                    std::is_same_v<decltype(rhs), INNC::Tensor>,               \
                "This macro only compares between `Tensor`s");                 \
  ASSERT_EQ((lhs).size(), (rhs).size());                                       \
  ASSERT_EQ((lhs).type(), (rhs).type());                                       \
  ASSERT_TRUE((((lhs) - (rhs)).abs() < epsilon).all());

TEST(basic, initialization) {
  auto a = INNC::zeros({1}, INNC::i16);
  ASSERT_EQ(a.to_string(), "[0]");
  a = INNC::zeros({1}, INNC::i32);
  ASSERT_EQ(a.to_string(), "[0]");
  a = INNC::zeros({1, 1}, INNC::f32);
  ASSERT_EQ(a.to_string(), "[[" + std::to_string(float(0)) + "]]");
  a = INNC::zeros({1, 2, 3}, INNC::i8);
  ASSERT_EQ(a.to_string(), "[[[0, 0, 0], [0, 0, 0]]]");
  ASSERT_EQ(a.numel(), 6);
  a = INNC::ones({1, 2, 3}, INNC::i32);
  ASSERT_EQ(a.to_string(), "[[[1, 1, 1], [1, 1, 1]]]");
  a = INNC::ones({2, 1, 1}, INNC::f32);
  std::string f32_one_str = std::to_string(double(1));
  ASSERT_EQ(a.to_string(),
            "[[[" + f32_one_str + "]], [[" + f32_one_str + "]]]");
  a = INNC::zeros({3, 2, 5}, INNC::i8);
  auto b = INNC::zeros_like(a);
  ASSERT_EQ(a.to_string(), b.to_string());
  a = INNC::ones({4, 5, 6}, INNC::i8);
  b = INNC::ones_like(a);
  ASSERT_EQ(a.to_string(), b.to_string());
  a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.to_string(), output_i8_1);
  a = 3;
  ASSERT_EQ(a.to_string(), "3");
  a = -3.14;
  ASSERT_EQ(a.to_string(), std::to_string(double(-3.14)));
  b = a;
  ASSERT_EQ(b.to_string(), std::to_string(double(-3.14)));
  INNC::Tensor c(float(2.71828));
  ASSERT_EQ(c.to_string(), std::to_string(float(2.71828)));
  b = INNC::zeros_like(a);
  ASSERT_EQ(b.to_string(), std::to_string(double(0)));
  b = INNC::ones_like(a);
  ASSERT_EQ(b.to_string(), std::to_string(double(1)));
  a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.to_string(), output_i8_1);
}

TEST(basic, type) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  auto b = a.type(INNC::i64);
  ASSERT_EQ(b.to_string(), output_i8_1);
  a = INNC::from_blob(data_i8_1 + 1, INNC::SizeVec{1}, INNC::i8);
  a = a.type(INNC::f64);
  ASSERT_EQ(a.to_string(), "[" + std::to_string(-3.0) + "]");
  a = a.type(INNC::i8);
  ASSERT_EQ(a.to_string(), "[-3]");
}

TEST(basic, compare) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::i64);
  auto b = a < 1;
  char rst_d0[6] = {0, 1, 1, 0, 1, 0};
  auto rst = INNC::from_blob(rst_d0, {2, 3}, INNC::i8);
  ASSERT_FALSE(b.all());
  ASSERT_TRUE((b == rst).all());
  b = a < 100;
  ASSERT_TRUE((b == INNC::ones_like(a).type(INNC::i8)).all());
  b = a > 1;
  char rst_d1[6] = {0, 0, 0, 1, 0, 1};
  ASSERT_TRUE((b == INNC::from_blob(rst_d1, {2, 3}, INNC::i8)).all());
  b = a.type(INNC::f64);
  ASSERT_TRUE((a == b).all());
  char rst_d2[6] = {1, 0, 0, 0, 0, 0};
  ASSERT_TRUE(((a == 1) == INNC::from_blob(rst_d2, {2, 3}, INNC::i8)).all());
  char rst_d3[6] = {1, 0, 0, 1, 0, 1};
  ASSERT_TRUE(
      ((a.type(INNC::f32) >= 1) == INNC::from_blob(rst_d3, {2, 3}, INNC::i8))
          .all());
  char rst_d4[6] = {1, 1, 1, 0, 1, 0};
  ASSERT_TRUE(((a <= 1) == INNC::from_blob(rst_d4, {2, 3}, INNC::i8)).all());
  char rst_d5[6] = {0, 1, 1, 1, 1, 1};
  ASSERT_TRUE(
      ((a.type(INNC::i16) != 1) == INNC::from_blob(rst_d5, {2, 3}, INNC::i8))
          .all());
}

TEST(arithmetic, add) {
  auto a = INNC::ones({2, 3}, INNC::i16);
  auto b = INNC::ones({2, 3}, INNC::i32);
  ASSERT_EQ((a + b).type(), INNC::i32);
  ASSERT_EQ((a + (a + b)).to_string(), "[[3, 3, 3], [3, 3, 3]]");
  ASSERT_EQ((a + b).to_string(), "[[2, 2, 2], [2, 2, 2]]");
  a = 2;
  b = 4;
  ASSERT_EQ((a + b).to_string(), "6");
  a = INNC::ones({1}, INNC::i64);
  b = INNC::ones({1}, INNC::i32);
  ASSERT_EQ((a + b).to_string(), "[2]");
  a = a["4:5"];
  ASSERT_THROW(a + b, std::runtime_error);
  a = INNC::from_blob(data_i8_1, {3, 1, 2, 1}, INNC::i8);
  b = INNC::from_blob(data_i16_1, {3, 1, 2}, INNC::i16);
  char rst_d[3][3][2][2] = {
      {{{1, -1}, {-3, -5}}, {{5, 7}, {1, 3}}, {{9, -9}, {5, -13}}},
      {{{-5, -7}, {7, 5}}, {{-1, 1}, {11, 13}}, {{3, -15}, {15, -3}}},
      {{{-9, -11}, {11, 9}}, {{-5, -3}, {15, 17}}, {{-1, -19}, {19, 1}}}};
  auto c = a + b;
  auto rst = INNC::from_blob(rst_d, {3, 3, 2, 2}, INNC::i8).type(c.type());
  ASSERT_EQ(c.to_string(), rst.to_string());
}

TEST(arithmetic, substract) {
  auto a = INNC::ones({2, 3}, INNC::i16);
  auto b = INNC::ones({2, 3}, INNC::i32);
  ASSERT_EQ((a - b).to_string(), INNC::zeros_like(b).to_string());
  b = a - 1;
  ASSERT_EQ(b.to_string(), INNC::zeros_like(b).to_string());
  a = INNC::from_blob(data_i8_1, {3, 1, 2, 1}, INNC::i8);
  b = INNC::from_blob(data_i16_1, {3, 1, 2}, INNC::i16);
  char rst_d[3][3][2][2] = {
      {{{1, 3}, {-3, -1}}, {{-3, -5}, {-7, -9}}, {{-7, 11}, {-11, 7}}},
      {{{-5, -3}, {7, 9}}, {{-9, -11}, {3, 1}}, {{-13, 5}, {-1, 17}}},
      {{{-9, -7}, {11, 13}}, {{-13, -15}, {7, 5}}, {{-17, 1}, {3, 21}}}};
  auto c = a - b;
  auto rst = INNC::from_blob(rst_d, {3, 3, 2, 2}, INNC::i8).type(c.type());
  ASSERT_EQ(c.to_string(), rst.to_string());
  a = INNC::from_blob(data_i8_1, {3, 2}, INNC::i8);
  // std::int8_t data_i8_1[6] = {1, -3, -5, 7, -9, 11};
  char rst_d1[3][2] = {{-1, 3}, {5, -7}, {9, -11}};
  rst = INNC::from_blob(rst_d1, {3, 2}, INNC::i8);
  ASSERT_EQ((-a).to_string(), rst.type(a.type()).to_string());
}

TEST(arithmetic, multiplication) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  auto b = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16);
  ASSERT_EQ((a * b).to_string(), "[[0, 6, -20], [42, -72, -110]]");
  ASSERT_EQ((a * (a * b)).to_string(), "[[0, -18, 100], [294, 648, -1210]]");
  a = 2;
  b = 4;
  ASSERT_EQ((a * b).to_string(), "8");
  a = INNC::ones({1}, INNC::i64);
  b = INNC::ones({1}, INNC::i32);
  ASSERT_EQ((a + b).to_string(), "[2]");
  a = INNC::from_blob(data_i8_1, {3, 1, 2, 1}, INNC::i8);
  b = INNC::from_blob(data_i16_1, {3, 1, 2}, INNC::i16);
  char rst_d[3][3][2][2] = {
      {{{0, -2}, {0, 6}}, {{4, 6}, {-12, -18}}, {{8, -10}, {-24, 30}}},
      {{{0, 10}, {0, -14}}, {{-20, -30}, {28, 42}}, {{-40, 50}, {56, -70}}},
      {{{0, 18}, {0, -22}}, {{-36, -54}, {44, 66}}, {{-72, 90}, {88, -110}}}};
  auto c = a * b;
  auto rst = INNC::from_blob(rst_d, {3, 3, 2, 2}, INNC::i8).type(c.type());
  ASSERT_EQ(c.to_string(), rst.to_string());
}

TEST(arithmetic, divide) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  auto b = a / 3;
  std::int8_t rst_d0[2][3] = {{0, -1, -1}, {2, -3, 3}};
  auto rst = INNC::from_blob(rst_d0, {2, 3}, INNC::i8);
  ASSERT_EQ(b.to_string(), rst.to_string());
  a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f64);
  b = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16);
  ASSERT_EQ((b / a * a).to_string(), b.type(INNC::f64).to_string());
  a = INNC::from_blob(data_i8_1, {3, 1, 2, 1}, INNC::i8);
  b = INNC::from_blob(data_i16_1, {3, 1, 2}, INNC::i16);
  char rst_d1[3][3][2][2] = {
      {{{0, -2}, {0, 0}}, {{4, 6}, {-1, -2}}, {{8, -10}, {-2, 3}}},
      {{{0, 0}, {0, 0}}, {{0, -1}, {0, 0}}, {{-1, 2}, {1, -1}}},
      {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}, {{0, 1}, {0, 0}}}};
  auto c = b / a;
  rst = INNC::from_blob(rst_d1, {3, 3, 2, 2}, INNC::i8).type(c.type());
  ASSERT_EQ(c.to_string(), rst.to_string());
  ASSERT_STRICT_APPROX(c, rst);
}

TEST(arithmetic, sum) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.sum().to_string(), std::to_string(std::int8_t(2)));
  a = INNC::from_blob(data_i16_1, {3, 2}, INNC::i16);
  ASSERT_EQ(a.sum().to_string(), std::to_string(std::int16_t(6)));
  a = std::int16_t(3);
  ASSERT_EQ(a.sum().to_string(), std::to_string(std::int16_t(3)));
  a = INNC::ones({1}, INNC::i16);
  ASSERT_EQ(a.sum().to_string(), std::to_string(std::int16_t(1)));
}

TEST(arithmetic, max) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.max().to_string(), std::to_string(std::int8_t(11)));
  a = INNC::from_blob(data_i16_1, {3, 1, 2}, INNC::i16);
  ASSERT_EQ(a.max().to_string(), std::to_string(std::int16_t(8)));
  a = std::int16_t(3);
  ASSERT_EQ(a.max().to_string(), std::to_string(std::int16_t(3)));
  a = INNC::ones({1}, INNC::i16);
  ASSERT_EQ(a.sum().to_string(), std::to_string(std::int16_t(1)));
  a = a["3:4"];
  a.max();
}

TEST(index, slice) {
  // [[0, 1, 2],
  //  [3, 4, 5],
  //  [6, 7, 8],
  //  [9,10,11]]
  auto a = INNC::from_blob(data_i16_2, {4, 3}, INNC::i16);
  auto b = a[""];
  ASSERT_EQ(b.size().to_string(), "[4, 3]");
  b = a[","];
  ASSERT_EQ(b.size().to_string(), "[4, 3]");
  b = a[":,::"];
  ASSERT_EQ(b.size().to_string(), "[4, 3]");
  b = a[":,"];
  ASSERT_EQ(b.size().to_string(), "[4, 3]");
  b = a["1, 1"];
  ASSERT_EQ(b.size().to_string(), "[]");
  ASSERT_EQ(b.to_string(), "4");
  b = a["2,:"];
  ASSERT_EQ(b.size().to_string(), "[3]");
  ASSERT_EQ(b.to_string(), "[6, 7, 8]");
  b = a[":,-1"];
  ASSERT_EQ(b.size().to_string(), "[4]");
  ASSERT_EQ(b.to_string(), "[2, 5, 8, 11]");
  b = a["-4:-1:2,-1:0:-2"];
  ASSERT_EQ(b.size().to_string(), "[2, 1]");
  ASSERT_EQ(b.to_string(), "[[2], [8]]");
  b = a["-99: 99 :2, 99 : -99:-2"];
  ASSERT_EQ(b.size().to_string(), "[2, 2]");
  ASSERT_EQ(b.to_string(), "[[2, 0], [8, 6]]");
  b = a["::2, ::-2"];
  ASSERT_EQ(b.size().to_string(), "[2, 2]");
  ASSERT_EQ(b.to_string(), "[[2, 0], [8, 6]]");
  b = a["99::2, :-99:-2"];
  ASSERT_EQ(b.size().to_string(), "[0, 2]");
  ASSERT_EQ(b.to_string(), "[]");
  b = b["-2:,1"];
  ASSERT_EQ(b.size().to_string(), "[0]");
  ASSERT_EQ(b.to_string(), "[]");
  b = a["-5::-2,:-4:2"];
  ASSERT_EQ(b.size().to_string(), "[0, 0]");
  ASSERT_EQ(b.to_string(), "[]");
  b = a[":3:-2,2:1:2"];
  ASSERT_EQ(b.size().to_string(), "[0, 0]");
  ASSERT_EQ(b.to_string(), "[]");
  ASSERT_THROW(a[" - 1, 2"], std::invalid_argument);
  ASSERT_THROW(a["99"], std::runtime_error);
}

TEST(index, transpose) {
  auto a = INNC::from_blob(data_i16_2, {4, 3}, INNC::i16);
  auto b = INNC::transpose(a, 0, 1);
  ASSERT_EQ(b.to_string(), "[[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]");
  a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  b = a.transpose(a, 0, 1);
  ASSERT_EQ(b.to_string(), "[[1, 7], [-3, -9], [-5, 11]]");
  a = INNC::from_blob(data_i16_2, {2, 3, 2}, INNC::i16);
  b = INNC::transpose(a, 0, 2);
  ASSERT_EQ(b.to_string(),
            "[[[0, 6], [2, 8], [4, 10]], [[1, 7], [3, 9], [5, 11]]]");
  a = INNC::from_blob(data_i8_1, {3, 2}, INNC::i8).type(INNC::f64);
  a = a.transpose(0, 1);
  double f64_arr[6] = {1, -5, -9, -3, 7, 11};
  b = INNC::from_blob(f64_arr, {2, 3}, INNC::f64);
  ASSERT_EQ(a.to_string(), b.to_string());
  ASSERT_EQ(a.transpose(0, 1).transpose(0, 1).to_string(), a.to_string());

  a = INNC::from_blob(data_i16_2, {4, 3}, INNC::i16);
  b = INNC::transpose(a, 0, 1);
  ASSERT_EQ(b.to_string(), "[[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]");
  a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  b = a.permute(a, {0, 1});
  ASSERT_EQ(b.to_string(), "[[1, 7], [-3, -9], [-5, 11]]");
  a = INNC::from_blob(data_i16_2, {2, 3, 2}, INNC::i16);
  b = INNC::Tensor::permute(a, {0, 2});
  ASSERT_EQ(b.to_string(),
            "[[[0, 6], [2, 8], [4, 10]], [[1, 7], [3, 9], [5, 11]]]");
  a = INNC::from_blob(data_i8_1, {3, 2}, INNC::i8).type(INNC::f64);
  a = a.permute({0, 1});
  b = INNC::from_blob(f64_arr, {2, 3}, INNC::f64);
  ASSERT_EQ(a.to_string(), b.to_string());
  ASSERT_EQ(a.permute({0, 1}).permute({0, 1}).to_string(), a.to_string());
  a = INNC::from_blob(data_i16_2, {1, 3, 4}, INNC::i16);
  b = INNC::Tensor::permute(a, {2, 0, 1});
  std::int16_t rst1[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  ASSERT_STRICT_APPROX(b, INNC::from_blob(rst1, {4, 1, 3}, INNC::i16));
}

TEST(index, reshape) {
  auto a = INNC::from_blob(data_i16_2, {3, 4}, INNC::i16);
  auto b = INNC::reshape(a, {4, 3});
  ASSERT_EQ(b.to_string(), "[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]");
  b = a.reshape({2, 6});
  ASSERT_EQ(b.to_string(), "[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]");
  auto c = INNC::zeros({12}, INNC::i16);
  b = a.reshape_as(c);
  ASSERT_EQ(b.to_string(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
  b = a.reshape({2, -1});
  ASSERT_EQ(b.to_string(), "[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]");
  ASSERT_THROW(INNC::reshape(a, {6, 3}), std::runtime_error);
  ASSERT_THROW(INNC::reshape(a, {4, -1, -1}), std::runtime_error);
  ASSERT_THROW(INNC::reshape(a, {7, -1}), std::runtime_error);
  ASSERT_THROW(INNC::reshape(a, {}), std::runtime_error);
  a = INNC::from_blob(data_i16_2, {4, 3}, INNC::i16);
  b = a["::2, :"];
  ASSERT_EQ(b.to_string(), "[[0, 1, 2], [6, 7, 8]]");
  c = b.reshape({6});
  ASSERT_EQ(c.to_string(), "[0, 1, 2, 6, 7, 8]");
  a = INNC::ones({}, INNC::i8);
  b = INNC::reshape(a, {1});
  ASSERT_EQ(b.to_string(), "[1]");
  b = INNC::reshape(a, {1, 1, 1});
  ASSERT_EQ(b.to_string(), "[[[1]]]");
  a = INNC::ones({1}, INNC::i8);
  b = INNC::reshape(a, {});
  ASSERT_EQ(b.to_string(), "1");
  float data[15] = {0.1, 1.2, 3.4, 5.6, 7.8, 2.2, 3.1, 4.5, 6.7, 8.9, 4.9, 5.2, 6.3, 7.4, 8.5};
  a = INNC::from_blob(data, {3, 5}, INNC::f32);
  ASSERT_STRICT_APPROX(a.view({5,3}), INNC::from_blob(data, {5, 3}, INNC::f32));
  ASSERT_STRICT_APPROX(a.view({1, -1}), INNC::from_blob(data, {1, 15}, INNC::f32));
}

TEST(autograd, add) {
  auto a = INNC::ones({2, 3}, INNC::f64);
  auto b = INNC::ones({2, 3}, INNC::f32);
  a.requires_grad(true);
  ASSERT_EQ(a.grad().to_string(), "[]");
  ASSERT_EQ(b.grad().to_string(), "[]");
  auto c = a + (b + a);
  ASSERT_TRUE(c.requires_grad());
  auto d = b + (a + b);
  ASSERT_EQ(c.grad().to_string(), "[]");
  c.sum().backward();
  ASSERT_TRUE(c.requires_grad());
  ASSERT_EQ(a.grad().to_string(), (a + a).to_string());
  ASSERT_EQ(b.grad().to_string(), "[]");
  b.requires_grad(true);
  d.sum().backward();

  a = INNC::ones({1}, INNC::f32);
  b = INNC::ones({1}, INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  c = a + a;
  d = c + c;
  d.sum().backward();
  ASSERT_EQ(a.grad().to_string(), "[" + std::to_string(4.0) + "]");

  a = float(1);
  b = double(1);
  a.requires_grad(true);
  b.requires_grad(true);
  c = a + (b + a);
  c = c + b;
  d = c + c + a;
  d.sum().backward();
  ASSERT_EQ(a.grad().to_string(), INNC::Tensor(float(5)).to_string());
  ASSERT_EQ(b.grad().to_string(), INNC::Tensor(double(4)).to_string());
  a.zero_grad();
  ASSERT_EQ(a.grad().to_string(), INNC::Tensor(float(0)).to_string());
}

TEST(autograd, substract) {
  auto a = INNC::ones({2, 3}, INNC::f64);
  auto b = INNC::ones({2, 1}, INNC::f32);
  a.requires_grad(true);
  b.requires_grad(true);
  (a - b).sum().backward();
  ASSERT_STRICT_APPROX(a.grad(), a);
  ASSERT_STRICT_APPROX(b.grad(), b * (-3));
  b = double(1);
  b.requires_grad(true);
  (a - b).sum().backward();
  ASSERT_STRICT_APPROX(b.grad(), b * (-6));
  b = double(5);
  b.requires_grad(true);
  (a - b + b).sum().backward();
  ASSERT_STRICT_APPROX(b.grad(), INNC::zeros_like(b));
}

TEST(autograd, multiplication) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f32);
  auto b = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  auto c = a * b;
  c.sum().backward();
  ASSERT_STRICT_APPROX(a.grad(), b.type(a.type()));
  ASSERT_STRICT_APPROX(b.grad(), a.type(b.type()));
  a.zero_grad();
  c = a * a;
  c.sum().backward();
  ASSERT_STRICT_APPROX(a.grad(), a + a);
  a.zero_grad();
  b.requires_grad(false);
  (a * (a * b) * b).sum().backward();
  ASSERT_STRICT_APPROX(a.grad(), ((a + a) * b * b).type(a.type()));
  a.zero_grad();
  b = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8);
  (a * b).sum().backward();
  ASSERT_STRICT_APPROX(a.grad(), b.type(a.type()));
  a.zero_grad();
  b = double(2);
  b.requires_grad(true);
  (a * b).sum().backward();
  ASSERT_STRICT_APPROX(a.grad(), INNC::zeros_like(a) + 2);
  ASSERT_STRICT_APPROX(b.grad(), INNC::zeros_like(b) + 2);
}

TEST(autograd, divide) {
  auto a = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f32);
  auto b = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  (b / a).sum().backward();
  ASSERT_STRICT_APPROX(b.grad(), (1 / a).type(b.type()));
  auto c = -b;
  ASSERT_STRICT_APPROX(a.grad(), (-b / a / a).type(a.type()));
  b = double(5);
  b.requires_grad(true);
  a.zero_grad();
  (b / a * a).sum().backward();
  ASSERT_STRICT_APPROX(b.grad(), INNC::Tensor(6).type(b.type()));
  ASSERT_STRICT_APPROX(a.grad(), INNC::zeros_like(a));
}

TEST(autograd, slice) {
  // [[0, 1, 2],
  //  [3, 4, 5],
  //  [6, 7, 8],
  //  [9,10,11]]
  auto a = INNC::from_blob(data_i16_2, {4, 3}, INNC::i16).type(INNC::f32);
  a.requires_grad(true);
  auto b = a[""];
  b = a[":,"];
  b.sum().backward();
  ASSERT_EQ(a.grad().to_string(), INNC::ones({4, 3}, INNC::f32).to_string());
  a.zero_grad();
  b = a["1, 1"];
  b.sum().backward();
  ASSERT_EQ(b.grad().to_string(), "[]");
  char rst0[4][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}};
  ASSERT_EQ(
      a.grad().to_string(),
      INNC::from_blob(&rst0, {4, 3}, INNC::i8).type(INNC::f32).to_string());
  b = a["2,:"];
  b.sum().backward();
  char rst1[4][3] = {{0, 0, 0}, {0, 1, 0}, {1, 1, 1}, {0, 0, 0}};
  ASSERT_EQ(
      a.grad().to_string(),
      INNC::from_blob(&rst1, {4, 3}, INNC::i8).type(INNC::f32).to_string());
  b = a[":,-1"];
  b.sum().backward();
  char rst2[4][3] = {{0, 0, 1}, {0, 1, 1}, {1, 1, 2}, {0, 0, 1}};
  ASSERT_EQ(
      a.grad().to_string(),
      INNC::from_blob(&rst2, {4, 3}, INNC::i8).type(INNC::f32).to_string());
  b = a["-4:-1:2,-1:0:-2"];
  b.sum().backward();
  char rst3[4][3] = {{0, 0, 2}, {0, 1, 1}, {1, 1, 3}, {0, 0, 1}};
  ASSERT_EQ(
      a.grad().to_string(),
      INNC::from_blob(&rst3, {4, 3}, INNC::i8).type(INNC::f32).to_string());
  b = a["-99: 99 :2, 99 : -99:-2"];
  b.sum().backward();
  char rst4[4][3] = {{1, 0, 3}, {0, 1, 1}, {2, 1, 4}, {0, 0, 1}};
  ASSERT_EQ(
      a.grad().to_string(),
      INNC::from_blob(&rst4, {4, 3}, INNC::i8).type(INNC::f32).to_string());
  b = a["::2, ::-2"];
  char rst5[4][3] = {{1, 0, 3}, {0, 1, 1}, {2, 1, 4}, {0, 0, 1}};
  ASSERT_EQ(
      a.grad().to_string(),
      INNC::from_blob(&rst5, {4, 3}, INNC::i8).type(INNC::f32).to_string());
  a.zero_grad();
  b = a["99::2, :-99:-2"];
  b.sum().backward();
  ASSERT_EQ(a.grad().to_string(), INNC::zeros_like(a).to_string());
}

TEST(autograd, transpose) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f64);
  auto b = INNC::from_blob(data_i8_1, {3, 2}, INNC::i8).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  auto c = b.transpose(0, 1);
  c.retain_grad(true);
  (a * c).sum().backward();
  ASSERT_EQ(a.grad().to_string(), c.to_string());
  ASSERT_EQ(c.grad().to_string(), a.to_string());
  ASSERT_EQ(b.grad().to_string(), a.transpose(0, 1).to_string());
  a.zero_grad();
  a.transpose(0, 1).transpose(0, 1).transpose(0, 1).sum().backward();
  ASSERT_EQ(a.grad().to_string(), INNC::ones_like(a).to_string());
}

TEST(autograd, clone) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f64);
  a.requires_grad(true);
  auto b = a.clone();
  a.sum().backward();
  ASSERT_EQ(a.grad().to_string(), INNC::ones_like(a).to_string());
  ASSERT_EQ(b.grad().to_string(), "[]");
  a.zero_grad();
  b.requires_grad();
  b.sum().backward();
  (a + b).sum().backward();
  auto rst = INNC::ones_like(a);
  rst = rst + rst;
  ASSERT_EQ(a.grad().to_string(), rst.to_string());
  ASSERT_EQ(b.grad().to_string(), INNC::ones_like(b).to_string());
}

TEST(autograd, detach) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f64);
  a.requires_grad(true);
  auto b = a.clone().detach();
  (a + b).sum().backward();
  ASSERT_EQ(a.grad().to_string(), INNC::ones_like(a).to_string());
}

TEST(autograd, contiguous) {
  // [[0, -2,  4],
  //  [6, 8, -10]]
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f64);
  a.requires_grad(true);
  auto b = a.transpose(0, 1);
  b.retain_grad(true);
  ASSERT_EQ(b.is_contiguous(), false);
  auto c = b.contiguous();
  c.retain_grad(true);
  ASSERT_EQ(c.is_contiguous(), true);
  // [[0,  6 ], [4, -10]]
  auto d = c["::2, :"];
  d.retain_grad(true);
  ASSERT_EQ(d.is_contiguous(), false);
  d.sum().backward();
  char rst[2][3] = {{1, 0, 1}, {1, 0, 1}};
  auto rst_tensor = INNC::from_blob(rst, {2, 3}, INNC::i8).type(a.type());
  ASSERT_EQ(d.grad().to_string(), INNC::ones_like(d).to_string());
  ASSERT_EQ(c.grad().to_string(), rst_tensor.transpose(0, 1).to_string());
  ASSERT_EQ(b.grad().to_string(), rst_tensor.transpose(0, 1).to_string());
  ASSERT_EQ(a.grad().to_string(),
            INNC::from_blob(rst, {2, 3}, INNC::i8).type(a.type()).to_string());
}

TEST(autograd, reshape) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f32);
  auto b = INNC::from_blob(data_i8_1, {6}, INNC::i8).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  auto c = b.reshape({2, 3});
  c.retain_grad(true);
  (a * c).sum().backward();
  ASSERT_EQ(a.grad().to_string(), c.type(a.type()).to_string());
  ASSERT_EQ(c.grad().to_string(), a.type(c.type()).to_string());
  ASSERT_EQ(b.grad().to_string(), a.reshape({-1}).type(b.type()).to_string());
  ASSERT_THROW(a.reshape({}), std::runtime_error);
  a = INNC::from_blob(data_i16_2, {4, 3}, INNC::i16).type(INNC::f64);
  a.requires_grad(true);
  b = a["::2, :"];
  std::int8_t rst0[2][3] = {{0, 1, 2}, {6, 7, 8}};
  ASSERT_EQ(
      b.to_string(),
      INNC::from_blob(&rst0, {2, 3}, INNC::i8).type(INNC::f64).to_string());
  c = b.reshape({3, 2});
  c.retain_grad(true);
  b.retain_grad(true);
  auto d = INNC::from_blob(data_i16_1, {3, 2}, INNC::i16).type(INNC::f64);
  d.requires_grad(true);
  (d * c).sum().backward();
  ASSERT_EQ(d.grad().to_string(), c.to_string());
  ASSERT_EQ(c.grad().to_string(), d.to_string());
  ASSERT_EQ(b.grad().to_string(), d.reshape({2, 3}).to_string());
}

TEST(autograd, type) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f32);
  auto b = INNC::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f32);
  a.requires_grad(true);
  ASSERT_THROW(a.type(INNC::i64), std::runtime_error);
  auto c = a.type(INNC::f64);
  c = c * b;
  c.sum().backward();
  ASSERT_EQ(a.grad().to_string(), b.to_string());
}

TEST(autograd, max) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f32);
  a.requires_grad(true);
  a.max().backward();
  std::int8_t rst_d0[2][3] = {{0, 0, 0}, {0, 1, 0}};
  auto rst = INNC::from_blob(rst_d0, {2, 3}, INNC::i8);
  ASSERT_STRICT_APPROX(a.grad(), rst.type(a.type()));
}

TEST(autograd, min) {
  auto a = INNC::from_blob(data_i16_1, {2, 3}, INNC::i16).type(INNC::f32);
  a.requires_grad(true);
  a.min().backward();
  std::int8_t rst_d0[2][3] = {{0, 0, 0}, {0, 0, 1}};
  auto rst = INNC::from_blob(rst_d0, {2, 3}, INNC::i8);
  ASSERT_STRICT_APPROX(a.grad(), rst.type(a.type()));
}

TEST(utils, utils) {
  ASSERT_THROW(INNC::sformat("%ls", "123"), std::runtime_error);
}

TEST(utils, randn) {
  ASSERT_THROW(INNC::Tensor::randn({3, 5}, INNC::i8), std::runtime_error);
}
