#include "INNC/types.hpp"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace INNC {

std::string to_string(types t) { return type_to_string_aux_arr[t]; }

std::string innc_type_to_string(NumericType auto num, types t) {
  switch (t) {
  case i8:
    return std::to_string(*static_cast<std::int8_t *>(num));
  case i16:
    return std::to_string(*static_cast<std::int16_t *>(num));
  case i32:
    return std::to_string(*static_cast<std::int32_t *>(num));
  case i64:
    return std::to_string(*static_cast<std::int64_t *>(num));
  case f32:
    return std::to_string(*static_cast<float *>(num));
  case f64:
    return std::to_string(*static_cast<double *>(num));
  default:
    std::cerr << "type not available" << std::endl;
    abort();
  }
}

std::string innc_type_to_string(void *ptr, types t) {
  switch (t) {
  case i8:
    return std::to_string(*static_cast<std::int8_t *>(ptr));
  case i16:
    return std::to_string(*static_cast<std::int16_t *>(ptr));
  case i32:
    return std::to_string(*static_cast<std::int32_t *>(ptr));
  case i64:
    return std::to_string(*static_cast<std::int64_t *>(ptr));
  case f32:
    return std::to_string(*static_cast<float *>(ptr));
  case f64:
    return std::to_string(*static_cast<double *>(ptr));
  default:
    std::cerr << "type not available" << std::endl;
    abort();
  }
}

std::string SizeVec::to_string() const noexcept {
  std::string s = "[";
  bool begin = true;
  for (auto i : *this) {
    if (begin) {
      s += std::to_string(i);
      begin = false;
      continue;
    }
    s += ", " + std::to_string(i);
  }
  return s + "]";
}

std::string SignedVec::to_string() const noexcept {
  std::string s = "[";
  bool begin = true;
  for (auto i : *this) {
    if (begin) {
      s += std::to_string(i);
      begin = false;
      continue;
    }
    s += ", " + std::to_string(i);
  }
  return s + "]";
}

SizeVec::SizeVec() = default;
SizeVec::SizeVec(const std::initializer_list<size_t> &init_list)
    : std::vector<size_t>{init_list} {}
SizeVec::SizeVec(is_same_wo_cvref<std::vector<int>> auto &&vec)
    : std::vector<size_t>(std::forward<decltype(vec)>(vec)) {}

SignedVec::SignedVec() = default;
SignedVec::SignedVec(const std::initializer_list<long long> &init_list)
    : std::vector<long long>{init_list} {}
SignedVec::SignedVec(is_same_wo_cvref<std::vector<int>> auto &&vec)
    : std::vector<size_t>(std::forward<decltype(vec)>(vec)) {}
} // namespace INNC
