#pragma once

#include "tensor.hpp"

namespace INNC {
const auto zeros = Tensor::zeros;
const auto zeros_like = Tensor::zeros_like;
const auto ones = Tensor::ones;
const auto ones_like = Tensor::ones_like;
const auto from_blob = Tensor::from_blob;
Tensor (*const transpose)(const Tensor &, std::size_t,
                          std::size_t) = Tensor::transpose;
Tensor (*const reshape)(const Tensor &, const SignedVec &) = Tensor::reshape;
} // namespace INNC
