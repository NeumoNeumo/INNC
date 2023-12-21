#include "INNC/types.hpp"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace INNC {
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
std::ostream &operator<<(std::ostream &o, const SizeVec &sv) noexcept {
  return o << sv.to_string();
}

std::string SizeVec::to_string() const noexcept {
  std::string s = "[";
  bool begin = true;
  for (auto i : *this) {
    if (begin) {
      s += i;
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
SizeVec::SizeVec(is_vec_i auto vec)
    : std::vector<size_t>(std::forward<decltype(vec)>(vec)) {}
} // namespace INNC
