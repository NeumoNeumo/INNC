#pragma once

#include <array>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <vector>

namespace INNC {
// Integer types must be place before floating point types. Types on the left
// have to be able to convert to those to the right.
enum types { i8, i16, i32, i64, f32, f64, Count };
constexpr std::array size_of_data_{1, 2, 4, 8, 4, 8};
inline types larger_type(types a, types b) { return a >= b ? a : b; }
constexpr unsigned char size_of(types t) { return size_of_data_[t]; }
std::string innc_type_to_string(void *ptr, types t);

template<typename T1, typename T2>
using larger_t = std::conditional_t<sizeof(T1) >= sizeof(T2), T1, T2>;

template <typename T, typename R>
concept is_same_wo_cvref = std::is_same_v<std::remove_cvref_t<T>, R>;

class SizeVec : public std::vector<size_t> {
public:
  SizeVec();
  SizeVec(const std::initializer_list<size_t> &init_list);
  SizeVec(is_same_wo_cvref<std::vector<int>> auto&& vec);
  friend std::ostream &operator<<(std::ostream &o, const SizeVec &sv) noexcept;
  std::string to_string() const noexcept;
};

}; // namespace INNC
