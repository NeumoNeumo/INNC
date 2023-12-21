#include "INNC/INNC.hpp"
#include <gtest/gtest.h>
#include <iostream>

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
}

TEST(arithmetic, add) {
  auto a = INNC::Tensor::ones({2, 3}, INNC::i16);
  auto b = INNC::Tensor::ones({2, 3}, INNC::i32);
  ASSERT_EQ((a + b).type(), INNC::i32);
  ASSERT_EQ((a + a).to_string(), "[[2, 2, 2], [2, 2, 2]]");
  ASSERT_EQ((a + b).to_string(), "[[2, 2, 2], [2, 2, 2]]");
}

TEST(autograd, add) {
  auto a = INNC::Tensor::ones({2, 3}, INNC::i16);
  auto b = INNC::Tensor::ones({2, 3}, INNC::f32);
  a.requires_grad(true);
  auto c = a + b;
  ASSERT_EQ(a.grad().to_string(), "Tensor not exist.");
  ASSERT_EQ(b.grad().to_string(), "Tensor not exist.");
  ASSERT_EQ(c.grad().to_string(), "Tensor not exist.");
  c.backward();
  ASSERT_TRUE(c.requires_grad());
  ASSERT_EQ(a.grad().to_string(), a.to_string());
  ASSERT_EQ(b.grad().to_string(), "Tensor not exist.");
  ASSERT_EQ(c.grad().to_string(), "Tensor not exist.");
}
