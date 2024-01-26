#pragma once

#include "INNC/layouts.hpp"
#include "INNC/types.hpp"

namespace INNC {
class TensorImpl;
class TensorInit;
/**
 * @brief ``Tensor`` holds the information of a multi-dimentional matrix. The
 * shape, stride, type is dynamic to this class, which means you might assign
 * the same object of ``Tensor`` with tensors with distinct sizes. Most of the
 * case, the member function of ``Tensor`` returns a new ``Tensor`` object.
 * Inplace operation is not encourged in INNC because it may break the
 * dependency of the backward graph.
 *
 * Examples:
 * \code{.cpp}
 * auto a = INNC::Tensor::zeros({2, 3, 4}, INNC::i16);
 * std::cout << a.to_string() << std::endl;
 * a = INNC::Tensor::ones({4, 3, 2}, INNC::i32);
 * std::cout << a.to_string() << std::endl;
 * \endcode
 *
 */
class Tensor {
private:
  std::shared_ptr<TensorImpl> fptr;
  Tensor(std::unique_ptr<TensorImpl> &&tf);
  Tensor(std::shared_ptr<TensorImpl> tf);

public:
  /**
   * @brief Initialize an empty Tensor.
   *
   */
  Tensor();

  /**
   * @brief The ctor is deleted because of its ambiguous semantics.
   *
   * Example:
   * \code{.cpp}
   * auto a = INNC::Tensor::zeros({2, 3, 4}, INNC::i16);
   * // auto b = INNC::Tensor(a); // Don't do that.
   * auto b = a.clone().detach(); // Use this syntax
   * auto c = a.detach().clone(); // or this syntax instead.
   * \endcode
   *
   */
  Tensor(const Tensor &t) = delete;
  Tensor(std::int8_t a);
  Tensor(std::int16_t a);
  Tensor(std::int32_t a);
  Tensor(std::int64_t a);
  Tensor(float a);
  Tensor(double a);
  Tensor(Tensor &&t);
  Tensor(TensorInit init);
  Tensor &operator=(const Tensor &t);
  Tensor &operator=(Tensor &&t);
  Tensor &operator=(TensorInit init);
  Tensor(const SizeVec &sizes, types dtype);
  ~Tensor();
  void backward();
  std::string to_string() const;
  const SizeVec size() const;
  const SignedVec stride() const;
  static Tensor zeros(const SizeVec &size, types t);
  static Tensor ones(const SizeVec &size, types t);
  static Tensor zeros_like(const Tensor &t);
  static Tensor ones_like(const Tensor &t);
  static Tensor full(const SizeVec &size, std::int64_t num, types dtype);
  static Tensor full(const SizeVec &size, double num, types dtype);
  static Tensor eye(size_t n, types dtype = types::i8);
  static Tensor eye(size_t n, size_t m, types dtype = types::i8);
  static Tensor from_blob(void *data, const SizeVec &sizes, types dtype);
  static Tensor randn(const SizeVec &sizes, types dtype);
  static Tensor randn_like(const Tensor &t);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  Tensor type(types t);
  Tensor operator[](const std::string &slice);
  static Tensor transpose(const Tensor &input, size_t dim0, size_t dim1);
  Tensor transpose(size_t dim0, size_t dim1);
  static Tensor reshape(const Tensor &input, const SignedVec &sizes);
  Tensor reshape(const SignedVec &sizes);
  Tensor reshape_as(const Tensor &input);
  static Tensor cat(const std::vector<Tensor> &input_tensors,
                    const size_t dim = 0);
  Tensor &operator+=(const Tensor &rhs);
  bool requires_grad() const noexcept;
  void requires_grad(bool b);
  bool retain_grad() const noexcept;
  void retain_grad(bool b) noexcept;
  size_t dim() const noexcept;
  Tensor grad() const noexcept;
  Tensor sum() const;
  Tensor mean() const;
  Tensor abs() const;
  Tensor max() const;
  Tensor min() const;
  void zero_grad() const noexcept;
  bool is_contiguous() const noexcept;
  Tensor contiguous() const;
  Tensor clone() const;
  Tensor detach() const;
  bool all() const;
  Tensor operator-();
  Tensor operator+();
  friend Tensor operator+(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator-(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator*(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator/(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator>(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator<(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator>=(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator<=(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator==(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator!=(const Tensor &lhs, const Tensor &rhs);
  friend class Backward;
};

class TensorInit {
  std::unique_ptr<uint8_t[]> data_ptr;
  SizeVec shape;
  types dtype;
  static inline bool is_same_spec(const TensorInit &lhs,
                                  const TensorInit &rhs) noexcept {
    return lhs.shape == rhs.shape && lhs.dtype == rhs.dtype;
  }

  size_t numel() const noexcept {
    size_t n = 1;
    for (auto i : shape)
      n *= i;
    return n;
  }

public:
  template <typename T>
    requires is_itype<T>
  TensorInit(std::initializer_list<T> args)
      : data_ptr(new uint8_t[args.size() * sizeof(T)]), shape{args.size()},
        dtype{to_ts<T>} {
    auto dptr = reinterpret_cast<T *>(data_ptr.get());
    for (auto t : args)
      *dptr++ = t;
  }

  TensorInit(std::initializer_list<TensorInit> args) {
    if (args.size() > 0) {
      shape.resize(1 + args.begin()->shape.size());
    } else
      return;
    auto first_sub = args.begin();
    if (!std::all_of(args.begin(), args.end(), [first_sub](const auto &lhs) {
          return is_same_spec(lhs, *first_sub);
        })) [[unlikely]]
      throw std::invalid_argument(
          "Shapes or types of subtensor does not match");
    dtype = first_sub->dtype;
    shape[0] = args.size();
    copy(first_sub->shape.begin(), first_sub->shape.end(), shape.begin() + 1);
    auto data_rawp = new uint8_t[numel() * size_of(dtype)];
    data_ptr.reset(data_rawp);
    auto step = first_sub->numel() * size_of(dtype);
    for (const auto &t : args) {
      auto start = t.data_ptr.get();
      std::copy(start, start + step, data_rawp);
      data_rawp += step;
    }
  }
  std::shared_ptr<TensorImpl> createImpl();
};

} // namespace INNC
