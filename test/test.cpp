#include "INNC/INNC.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(basic, initialization) {
  INNC::Tensor a = INNC::Tensor::zeros({1}, INNC::i16);
  ASSERT_EQ(a.to_string(), "[0]");
  a = INNC::Tensor::zeros({1}, INNC::i32);
  ASSERT_EQ(a.to_string(), "[0]");
  a = INNC::Tensor::zeros({1,1}, INNC::f32);
  ASSERT_EQ(a.to_string(), "[[" + std::to_string(float(0)) + "]]");
  a = INNC::Tensor::zeros({1,2,3}, INNC::i8);
  ASSERT_EQ(a.to_string(), "[[[0, 0, 0], [0, 0, 0]]]");
  ASSERT_EQ(a.elem_num(), 6);
  a = INNC::Tensor::ones({1,2,3}, INNC::i32);
  ASSERT_EQ(a.to_string(), "[[[1, 1, 1], [1, 1, 1]]]");
  a = INNC::Tensor::ones({2,1,1}, INNC::f32);
  std::string f32_one_str = std::to_string(double(1));
  ASSERT_EQ(a.to_string(), "[[[" + f32_one_str + "]], [[" + f32_one_str+ "]]]");
}

TEST(arithmetic, add){
  auto a = INNC::Tensor::ones({2,3}, INNC::i16);
  auto b = INNC::Tensor::ones({2,3}, INNC::i32);
  ASSERT_EQ((a + a).to_string(), "[[2, 2, 2], [2, 2, 2]]");
  ASSERT_EQ((a + b).to_string(), "[[2, 2, 2], [2, 2, 2]]");
}
