#include "INNC/INNC.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

std::int8_t data_i8_1[6] = {1, -3, -5, 7, -9, 11};
std::int16_t data_i16_1[6] = {0, -2, 4, 6, 8, -10};
std::string output_i8_1 = "[[1, -3, -5], [7, -9, 11]]";
std::int16_t data_i16_2[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

std::string typed_output(const std::string &s, INNC::types) {
  // TODO 2
  return "";
}

TEST(basic, initialization) {
  auto a = INNC::Tensor::zeros({1}, INNC::i16);
  ASSERT_EQ(a.to_string(), "[0]");
  a = INNC::Tensor::zeros({1}, INNC::i32);
  ASSERT_EQ(a.to_string(), "[0]");
  a = INNC::Tensor::zeros({1, 1}, INNC::f32);
  ASSERT_EQ(a.to_string(), "[[" + std::to_string(float(0)) + "]]");
  a = INNC::Tensor::zeros({1, 2, 3}, INNC::i8);
  ASSERT_EQ(a.to_string(), "[[[0, 0, 0], [0, 0, 0]]]");
  ASSERT_EQ(a.numel(), 6);
  a = INNC::Tensor::ones({1, 2, 3}, INNC::i32);
  ASSERT_EQ(a.to_string(), "[[[1, 1, 1], [1, 1, 1]]]");
  a = INNC::Tensor::ones({2, 1, 1}, INNC::f32);
  std::string f32_one_str = std::to_string(double(1));
  ASSERT_EQ(a.to_string(),
            "[[[" + f32_one_str + "]], [[" + f32_one_str + "]]]");
  a = INNC::Tensor::zeros({3, 2, 5}, INNC::i8);
  auto b = INNC::Tensor::zeros_like(a);
  ASSERT_EQ(a.to_string(), b.to_string());
  a = INNC::Tensor::ones({4, 5, 6}, INNC::i8);
  b = INNC::Tensor::ones_like(a);
  ASSERT_EQ(a.to_string(), b.to_string());
  a = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.to_string(), output_i8_1);
}

TEST(basic, type) {
  auto a = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.to_string(), output_i8_1);
  auto b = a.type(INNC::i64);
  ASSERT_EQ(b.to_string(), output_i8_1);
  a = INNC::Tensor::from_blob(data_i8_1 + 1, INNC::SizeVec{1}, INNC::i8);
  a = a.type(INNC::f64);
  ASSERT_EQ(a.to_string(), "[" + std::to_string(-3.0) + "]");
  a = a.type(INNC::i8);
  ASSERT_EQ(a.to_string(), "[-3]");
}

TEST(arithmetic, add) {
  auto a = INNC::Tensor::ones({2, 3}, INNC::i16);
  auto b = INNC::Tensor::ones({2, 3}, INNC::i32);
  ASSERT_EQ((a + b).type(), INNC::i32);
  ASSERT_EQ((a + (a + b)).to_string(), "[[3, 3, 3], [3, 3, 3]]");
  ASSERT_EQ((a + b).to_string(), "[[2, 2, 2], [2, 2, 2]]");
}

TEST(index, slice) {
  // [[0, 1, 2],
  //  [3, 4, 5],
  //  [6, 7, 8],
  //  [9,10,11]]
  auto a = INNC::Tensor::from_blob(data_i16_2, {4, 3}, INNC::i16);
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
}

TEST(index, reshape) {
  // [[0, 1, 2],
  //  [3, 4, 5],
  //  [6, 7, 8],
  //  [9,10,11]]
  auto a = INNC::Tensor::from_blob(data_i16_2, {4, 3}, INNC::i16);
  // auto b = INNC::Tensor::reshape(a, {3, 4});
  // ASSERT_EQ(b.size().to_string(), "[3, 4]");
  // ASSERT_EQ(b.to_string(), "[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]");
  auto b = INNC::Tensor::reshape(a, {3, 2, 2});
  ASSERT_EQ(b.size().to_string(), "[3, 2, 2]");
  b = INNC::Tensor::reshape(a, {-1});
  ASSERT_EQ(b.size().to_string(), "[12]");
  ASSERT_EQ(b.to_string(), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
  b = INNC::Tensor::reshape(a, {3, -1});
  ASSERT_EQ(b.size().to_string(), "[3, 4]");
  ASSERT_EQ(b.to_string(), "[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]");
  a = INNC::Tensor::ones({4, 3}, INNC::f32);
  b = INNC::Tensor::reshape(a, {3, 4});
  ASSERT_EQ(b.size().to_string(), "[3, 4]");
  std::string f32_one_str = std::to_string(double(1));
  ASSERT_EQ(b.to_string(), "[[" + f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + "]]");
  b = a.reshape({2, 6});
  ASSERT_EQ(b.size().to_string(), "[2, 6]");
  ASSERT_EQ(b.to_string(), "[[" + f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + ", " +
                               f32_one_str + ", " + f32_one_str + "]]");
  b = a.reshape_as(INNC::Tensor::ones({6, 2}, INNC::f32));
  ASSERT_EQ(b.to_string(), "[[" + f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + "], [" +
                               f32_one_str + ", " + f32_one_str + "]]");
}

TEST(index, transpose) {
  auto a = INNC::Tensor::from_blob(data_i16_2, {4, 3}, INNC::i16);
  auto b = INNC::Tensor::transpose(a, 0, 1);
  ASSERT_EQ(b.to_string(), "[[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]");
  a = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8);
  b = INNC::Tensor::transpose(a, 0, 1);
  ASSERT_EQ(b.to_string(), "[[1, 7], [-3, -9], [-5, 11]]");
  a = INNC::Tensor::from_blob(data_i16_2, {2, 3, 2}, INNC::i16);
  b = INNC::Tensor::transpose(a, 0, 2);
  ASSERT_EQ(b.to_string(),
            "[[[0, 6], [2, 8], [4, 10]], [[1, 7], [3, 9], [5, 11]]]");
  a = INNC::Tensor::from_blob(data_i8_1, {3, 2}, INNC::i8).type(INNC::f64);
  b = INNC::Tensor::transpose(a, 0, 1);
  ASSERT_EQ(b.to_string(), "[[" + std::to_string(double(1)) + ", " +
                               std::to_string(double(-5)) + ", " +
                               std::to_string(double(-9)) + "], [" +
                               std::to_string(double(-3)) + ", " +
                               std::to_string(double(7)) + ", " +
                               std::to_string(double(11)) + "]]");
}

TEST(arithmetic, mul) {
  auto a = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8);
  auto b = INNC::Tensor::from_blob(data_i16_1, {2, 3}, INNC::i16);
  ASSERT_EQ((a * b).to_string(), "[[0, 6, -20], [42, -72, -110]]");
  ASSERT_EQ((a * (a * b)).to_string(), "[[0, -18, 100], [294, 648, -1210]]");
}

TEST(arithmetic, sum) {
  auto a = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8);
  ASSERT_EQ(a.sum().to_string(), "[2]");
  a = INNC::Tensor::from_blob(data_i16_1, {3, 2}, INNC::i16);
  ASSERT_EQ(a.sum().to_string(), "[6]");
}

TEST(autograd, add) {
  auto a = INNC::Tensor::ones({2, 3}, INNC::f64);
  auto b = INNC::Tensor::ones({2, 3}, INNC::f32);
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

  a = INNC::Tensor::ones({1}, INNC::f32);
  b = INNC::Tensor::ones({1}, INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  c = a + a;
  d = c + c;
  d.sum().backward();
  ASSERT_EQ(a.grad().to_string(), "[" + std::to_string(4.0) + "]");

  a = INNC::Tensor::ones({1}, INNC::f32);
  b = INNC::Tensor::ones({1}, INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  c = a + (b + a);
  c = c + b;
  d = c + c + a;
  d.sum().backward();
  ASSERT_EQ(a.grad().to_string(), "[" + std::to_string(5.0) + "]");
  ASSERT_EQ(b.grad().to_string(), "[" + std::to_string(4.0) + "]");
  a.zero_grad();
  ASSERT_EQ(a.grad().to_string(), "[" + std::to_string(0.0) + "]");
}

TEST(autograd, mul) {
  auto a = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f32);
  auto b = INNC::Tensor::from_blob(data_i8_1, {2, 3}, INNC::i8).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  auto c = a * b;
  c.sum().backward();
  ASSERT_EQ(a.grad().to_string(), b.to_string());
  ASSERT_EQ(b.grad().to_string(), a.to_string());
}

TEST(autograd, reshape) {
  auto a =
      INNC::Tensor::from_blob(data_i16_1, {2, 3}, INNC::i8).type(INNC::f64);
  auto b = INNC::Tensor::from_blob(data_i8_1, {6}, INNC::i8).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  auto d = INNC::Tensor::reshape(b, {2, 3});
  d.requires_grad(true);

  auto c = a * d;
  c.sum().backward();
  ASSERT_EQ(a.grad().to_string(), d.to_string());
  ASSERT_EQ(b.grad().to_string(),
            "[" + std::to_string(double(0)) + ", " + std::to_string(double(0)) +
                ", " + std::to_string(double(-2)) + ", " +
                std::to_string(double(-1)) + ", " + std::to_string(double(4)) +
                ", " + std::to_string(double(0)) + "]");
  // ASSERT_EQ(d.grad().to_string(), a.to_string());
}

TEST(autograd, tranpose) {
  auto a =
      INNC::Tensor::from_blob(data_i16_1, {2, 3}, INNC::i8).type(INNC::f64);
  auto b = INNC::Tensor::from_blob(data_i8_1, {3, 2}, INNC::i8).type(INNC::f64);
  a.requires_grad(true);
  b.requires_grad(true);
  auto d = INNC::Tensor::transpose(b, 0, 1);
  d.requires_grad(true);
  auto c = a * d;
  c.sum().backward();
  ASSERT_EQ(a.grad().to_string(), d.to_string());
  ASSERT_EQ(b.grad().to_string(), "[[" + std::to_string(double(0)) + ", " +
                                      std::to_string(double(-1)) + "], [" +
                                      std::to_string(double(0)) + ", " +
                                      std::to_string(double(4)) + "], [" +
                                      std::to_string(double(-2)) + ", " +
                                      std::to_string(double(0)) + "]]");
  // ASSERT_EQ(d.grad().to_string(), a.to_string());
}