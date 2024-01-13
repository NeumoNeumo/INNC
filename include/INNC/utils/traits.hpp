#pragma once
#include <concepts>
#include <type_traits>

namespace INNC {
using size_t = std::size_t;
struct Any {
  template <typename T> operator T();
};

template <typename T> struct to_func_ptr {
  using type = T *;
};

template <typename T> struct to_func_ptr<T *> {
  using type = T *;
};

template <typename T> using to_func_ptr_t = typename to_func_ptr<T>::type;

template <typename T> consteval std::size_t get_param_n(auto &&...args) {
  using R = to_func_ptr_t<T>;
  if constexpr (requires { R{}(args...); })
    return sizeof...(args);
  else
    return get_param_n<R>(args..., Any{});
}

template <typename T, typename ArgType, typename Ret, size_t N>
consteval bool is_valid_func_h_(auto &&...args);

template <typename T, template <typename> typename RetJ>
concept struct_to_concept = RetJ<T>::value;

template <typename T, typename ArgType, template <typename> typename RetJ,
          size_t N>
consteval bool is_valid_func_h_(auto &&...args) {
  using R = to_func_ptr_t<T>;
  if constexpr (N == 0) {
    if constexpr (requires {
                    { R{}(args...) } -> struct_to_concept<RetJ>;
                  })
      return true;
    else
      return false;
  } else {
    return is_valid_func_h_<R, ArgType, RetJ, N - 1>(args..., ArgType{});
  }
}

template <typename T, typename ArgType, typename Ret, size_t N>
consteval bool is_valid_func_h_(auto &&...args) {
  using R = to_func_ptr_t<T>;
  if constexpr (N == 0) {
    if constexpr (requires {
                    { R{}(args...) } -> std::same_as<Ret>;
                  })
      return true;
    else
      return false;
  } else {
    return is_valid_func_h_<R, ArgType, Ret, N - 1>(args..., ArgType{});
  }
}

template <class F> struct return_type;

template <class T, class... A> struct return_type<T (*)(A...)> {
  typedef T type;
};

template <typename T, typename R>
concept is_same_wo_cvref = std::is_same_v<std::remove_cvref_t<T>, R>;

template <typename T>
concept NumericType = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Iterable = requires(T x) {
  ++x.begin();
  x.end();
};
} // namespace INNC
