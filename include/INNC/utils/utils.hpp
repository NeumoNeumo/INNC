#pragma once
#include "INNC/function.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/compile_opt.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace INNC {
std::vector<std::string> ssplit(const std::string &str, const char delimiter);

template <typename... Args>
std::string sformat(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  char *buf = new char[size];
  std::snprintf(buf, size, format.c_str(), args...);
  auto ret = std::string(buf, buf + size - 1);
  delete[] buf;
  return ret;
}

static const char *ws = " \t\n\r\f\v";

inline std::string &ltrim_(std::string &s) {
  return s.erase(0, s.find_first_not_of(ws));
}

inline std::string &rtrim_(std::string &s) {
  return s.erase(s.find_last_not_of(ws) + 1);
}

inline std::string &trim_(std::string &s) { return ltrim_(rtrim_(s)); }

template <typename T> struct is_valid_op_h_ {
  static constexpr size_t param_n = get_param_n<T>();
  static constexpr bool value =
      is_valid_func_h_<T, TensorImpl *, void, param_n>();
};

template <typename T> consteval bool is_valid_forward_func_() {
  using func_t = decltype(T::dispatch);
  constexpr size_t k = get_param_n<func_t>();
  return is_valid_func_h_<func_t, INNC::types, is_valid_op_h_, k>();
}

template <typename ForwardType>
concept is_valid_forward = is_valid_forward_func_<ForwardType>();

template <typename ForwardType>
  requires is_valid_forward<ForwardType>
std::shared_ptr<TensorImpl> apply_no_grad_binary_op(const TensorImpl &lhs,
                                                    const TensorImpl &rhs) {
  types lt = INNC::larger_type(lhs.dtype, rhs.dtype);
  auto ret = INNC::TensorImpl::create(lt, StridedLayout{lhs.view->sizes});
  if (lhs.dtype == lt) {
    ForwardType::dispatch(lhs.dtype, rhs.dtype)(ret.get(), &lhs, &rhs);
  } else {
    ForwardType::dispatch(rhs.dtype, lhs.dtype)(ret.get(), &rhs, &lhs);
  }
  return ret;
}

template <typename ForwardType>
  requires is_valid_forward<ForwardType>
void apply_no_grad_binary_op(TensorImpl &dst, const TensorImpl &lhs,
                             const TensorImpl &rhs) {
  ForwardType::dispatch(dst.dtype, lhs.dtype, rhs.dtype)(&dst, &lhs, &rhs);
}

template <typename ForwardType, typename BackwardType>
  requires is_valid_forward<ForwardType> &&
           std::derived_from<BackwardType, Backward>
std::shared_ptr<TensorImpl>
apply_binary_operator(std::shared_ptr<TensorImpl> lhs,
                      std::shared_ptr<TensorImpl> rhs) {
  check_same_size(*lhs.get(), *rhs.get());
  auto ret = apply_no_grad_binary_op<ForwardType>(*lhs, *rhs);
  if (!lhs->requires_grad && !rhs->requires_grad)
    return ret;
  ret->requires_grad = true;
  ret->grad_fn.reset(new BackwardType(ret.get(), {lhs, rhs}));
  return ret;
}

// TODO 2: fast path, concurrency, iterator, Broadcasting
void for_each_sizevec(const SizeVec &range, auto op) {
  [op, &range]() {
    if (__LIKELY(range.size() != 0)) {
      SizeVec sv;
      auto last_idx = range.size() - 1;
      sv.resize(last_idx + 1, 0);
      while (true) {
        size_t ptr = last_idx;
        while (sv[ptr] == range[ptr]) {
          if (ptr == 0)
            return;
          sv[ptr] = 0;
          ++sv[--ptr];
        }
        op(sv);
        ++sv[last_idx];
      };
    } else {
      op(SizeVec{});
    }
  }();
}

} // namespace INNC
