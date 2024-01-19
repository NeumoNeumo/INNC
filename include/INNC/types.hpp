#pragma once

#include "INNC/utils/traits.hpp"
#include <array>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <vector>

namespace INNC {
// Integer types must be place before floating point types. Types on the left
// have to be able to convert to those to the right. Do not touch these
// hard-coded variables unless you know what you are doing.
// BEGIN hard-coded variables
enum types { i8, i16, i32, i64, f32, f64, Count };
const char *const type_to_string_aux_arr[] = {"i8",  "i16", "i32",
                                              "i64", "f32", "f64"};
constexpr std::array size_of_data_{1, 2, 4, 8, 4, 8};
constexpr size_t float_type_idx_start_ = 4;
// END hard-coded variables

constexpr size_t float_type_n_ = types::Count - float_type_idx_start_;
inline types larger_type(types a, types b) { return a >= b ? a : b; }
constexpr unsigned char size_of(types t) { return size_of_data_[t]; }
inline bool is_int(types t) { return t < float_type_idx_start_; }
inline bool is_float(types t) { return t >= float_type_idx_start_; }
std::string to_string(types t);
std::string innc_type_to_string(NumericType auto num, types t);
std::string innc_type_to_string(void *ptr, types t);

template <typename T1, typename T2>
using size_larger_t = std::conditional_t<sizeof(T1) >= sizeof(T2), T1, T2>;

template <typename T>
concept is_innc_type = std::is_same_v<std::remove_cvref_t<T>, std::int8_t> ||
                       std::is_same_v<std::remove_cvref_t<T>, std::int16_t> ||
                       std::is_same_v<std::remove_cvref_t<T>, std::int32_t> ||
                       std::is_same_v<std::remove_cvref_t<T>, std::int64_t> ||
                       std::is_same_v<std::remove_cvref_t<T>, float> ||
                       std::is_same_v<std::remove_cvref_t<T>, double>;

template <typename L, typename R>
  requires is_innc_type<L> && is_innc_type<R>
struct innc_common_type {
  using type =
      std::conditional_t<std::is_integral_v<L> && std::is_integral_v<R>,
                         std::conditional_t<sizeof(L) >= sizeof(R), L, R>,
                         decltype(L{} + R{})>;
};

template <typename L, typename R>
using innc_common_t = typename innc_common_type<L, R>::type;

class SizeVec : public std::vector<size_t> {
public:
  SizeVec();
  SizeVec(const std::initializer_list<size_t> &init_list);
  SizeVec(is_same_wo_cvref<std::vector<int>> auto &&vec);
  friend std::ostream &operator<<(std::ostream &o, const SizeVec &sv) noexcept;
  std::string to_string() const noexcept;
};

class SignedVec : public std::vector<long long> {
public:
  SignedVec();
  SignedVec(const std::initializer_list<long long> &init_list);
  SignedVec(is_same_wo_cvref<std::vector<int>> auto &&vec);
  friend std::ostream &operator<<(std::ostream &o,
                                  const SignedVec &sv) noexcept;
  std::string to_string() const noexcept;
};

}; // namespace INNC
