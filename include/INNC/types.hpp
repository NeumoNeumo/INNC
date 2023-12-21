#pragma once

#include <array>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace INNC {
enum types { i8, i16, i32, i64, f32, f64, Count };
constexpr std::array size_of_data_{1, 2, 4, 8, 4, 8};
inline types common_type(types a, types b) { return a >= b ? a : b; }
constexpr unsigned char size_of(types t) { return size_of_data_[t]; }
std::string innc_type_to_string(void *ptr, types t);

template <typename T>
concept is_vec_i = std::is_same_v<std::remove_cvref_t<T>, std::vector<int>>;

class SizeVec : public std::vector<size_t> {
public:
  SizeVec();
  SizeVec(const std::initializer_list<size_t> &init_list);
  SizeVec(is_vec_i auto vec);
  friend std::ostream &operator<<(std::ostream &o, const SizeVec &sv) noexcept;
  std::string to_string() const noexcept;
};

}; // namespace INNC
