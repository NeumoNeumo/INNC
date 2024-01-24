#pragma once
#ifdef _MSC_VER
#error Static reflection is only tested under clang and gcc because \
I don't use MSVC. However, any PR is welcomed.
#endif

static_assert(
    sizeof(float) == 4,
    "[Arch not supported] float size is not 4bytes on this architecture.");
static_assert(
    sizeof(double) == 8,
    "[Arch not supported] double size is not 8bytes on this architecture.");

#include "INNC/utils/traits.hpp"
#include <array>
#include <initializer_list>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace INNC {
// Integer types must be place before floating point types. Types on the left
// have to be able to convert to those to the right. Do not touch these
// hard-coded variables unless you know what you are doing.
// BEGIN hard-coded variables
enum types { i8, i16, i32, i64, f32, f64, Count };
constexpr size_t float_type_idx_start_ = 4;
// END hard-coded variables

template <typename T> struct to_itype_aux;
template <typename T> constexpr types to_ts = to_itype_aux<T>::value;
template <types t> struct to_native_aux;
template <types t> using to_native = to_native_aux<t>::type;

#define REGISTER_TSTYPE(ts_typename, native_typename)                          \
  template <> struct to_itype_aux<native_typename> {                           \
    static constexpr types value = ts_typename;                                \
  };                                                                           \
  template <> struct to_native_aux<ts_typename> {                              \
    using type = native_typename;                                              \
  }

REGISTER_TSTYPE(i8, int8_t);
REGISTER_TSTYPE(i16, int16_t);
REGISTER_TSTYPE(i32, int32_t);
REGISTER_TSTYPE(i64, int64_t);
REGISTER_TSTYPE(f32, float);
REGISTER_TSTYPE(f64, double);

template <size_t... idx> consteval auto size_of_h(std::index_sequence<idx...>) {
  return std::array<size_t, Count>{
      sizeof(to_native<static_cast<types>(idx)>)...};
}

constexpr auto size_of_d = size_of_h(std::make_index_sequence<Count>());

inline constexpr auto size_of(types t) {
  return size_of_d[t];
}

template <types t> consteval auto get_typename() {
  std::string_view sig = __PRETTY_FUNCTION__;
  sig.remove_suffix(1);
  sig.remove_prefix(sig.find('=') + 2);
  return sig;
}

template <std::size_t N> struct static_string {
  consteval static_string(std::string_view sv) noexcept {
    std::copy(sv.begin(), sv.end(), content.begin());
  }
  consteval operator std::string_view() const noexcept {
    return {content.data(), N};
  }

private:
  std::array<char, N + 1> content{};
};

template <types t> consteval auto get_typename_zt() {
  constexpr auto sig = get_typename<t>();
  return static_string<sig.size()>{sig};
}

template<types t>
constexpr auto get_typename_zt_v = get_typename_zt<t>();

template <size_t... IdxSeq>
consteval auto get_types_str(std::index_sequence<IdxSeq...>) {
  return std::array<std::string_view, Count>{
      get_typename_zt_v<static_cast<types>(IdxSeq)>...};
}

constexpr auto types_str = get_types_str(std::make_index_sequence<Count>());

inline constexpr auto to_string(types t) {
  return types_str[t];
}

template <typename T, size_t...> struct is_any_itype_s;

template <typename T, size_t N, size_t... Ss>
struct is_any_itype_s<T, N, Ss...> {
  constexpr static bool value = is_any_itype_s<T, N - 1, N - 1, Ss...>::value;
};

template <typename T, size_t... Ss> struct is_any_itype_s<T, 0, Ss...> {
  constexpr static bool value =
      (std::is_same_v<T, to_native<static_cast<types>(Ss)>> || ...);
};

template <typename T>
concept is_itype = is_any_itype_s<T, Count>::value;

constexpr size_t float_type_n_ = types::Count - float_type_idx_start_;
inline types larger_type(types a, types b) { return a >= b ? a : b; }
inline bool is_int(types t) { return t < float_type_idx_start_; }
inline bool is_float(types t) { return t >= float_type_idx_start_; }
std::string innc_type_to_string(NumericType auto num, types t);
std::string innc_type_to_string(void *ptr, types t);

template <typename T1, typename T2>
using size_larger_t = std::conditional_t<sizeof(T1) >= sizeof(T2), T1, T2>;

template <typename L, typename R>
  requires is_itype<L> && is_itype<R>
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
