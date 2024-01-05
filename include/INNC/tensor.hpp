#pragma once

#include "storage.hpp"
#include "types.hpp"
#include <memory>

namespace INNC {
class Backward;
class Tensor;

// All member functions defined here never set up grad_fn. Also see `Tensor`.
class TensorFrame {
public:
  UntypedStorage data_;
  std::shared_ptr<TensorFrame> grad;
  const types dtype;
  const SizeVec sizes;
  const SizeVec strides;
  const size_t offset;

  size_t cnt_from_index(const SizeVec &index) const noexcept;
  std::string to_string_helper(std::ptrdiff_t offset = 0, size_t l = 0) const;

  bool requires_grad; // TODO 3 stricter encapsulation
  bool retain_grad;
  size_t _version;
  std::unique_ptr<Backward> grad_fn;

  TensorFrame() = delete;
  TensorFrame(const TensorFrame &) = delete;
  TensorFrame(TensorFrame &&t);
  TensorFrame &operator=(const TensorFrame &t) = delete;
  TensorFrame(types dtype, const SizeVec &sizes, const SizeVec &strides,
              const size_t offset, bool prealloc = true);
  ~TensorFrame();
  static TensorFrame make_stub();
  std::string to_string() const;
  static std::unique_ptr<TensorFrame>
  make_tf(types dtype, is_same_wo_cvref<SizeVec> auto &&sizes);
  static std::unique_ptr<TensorFrame> zeros(const SizeVec &sizes, types dtype);
  static std::unique_ptr<TensorFrame> ones(const SizeVec &sizes, types dtype);
  static std::unique_ptr<TensorFrame> zeros_like(const TensorFrame &t);
  static std::unique_ptr<TensorFrame> ones_like(const TensorFrame &t);
  static std::unique_ptr<TensorFrame>
  from_blob(void *data, const SizeVec &sizes, types dtype);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  std::unique_ptr<TensorFrame> type(types t);
  std::unique_ptr<TensorFrame> operator[](const std::string &slice);
  static std::unique_ptr<TensorFrame>
  transpose_without_grad(const std::shared_ptr<TensorFrame> &input, size_t dim0,
                         size_t dim1);
  static std::unique_ptr<TensorFrame>
  transpose(const std::shared_ptr<TensorFrame> &input, size_t dim0,
            size_t dim1);
  static std::unique_ptr<TensorFrame>
  reshape_without_grad(const TensorFrame &input, const SizeVec &sizes);
  static std::unique_ptr<TensorFrame>
  reshape(const std::shared_ptr<TensorFrame> &input,
          const std::initializer_list<int> &sizes);
  static std::unique_ptr<TensorFrame>
  reshape(const std::shared_ptr<TensorFrame> &input, const SizeVec &sizes);
  std::unique_ptr<TensorFrame> sum() const;
  void zero_grad() const noexcept;
  TensorFrame &operator+=(const TensorFrame &rhs);
  void try_accumulate_grad(TensorFrame *tf_w, TensorFrame *tf_o = nullptr);
  friend std::unique_ptr<TensorFrame> no_grad_add(const TensorFrame &lhs,
                                                  const TensorFrame &rhs);
  friend void check_same_size(const TensorFrame &lhs, const TensorFrame &rhs);
  friend class Backward;
  void backward();
};

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
 * std::cout << a.to_string << std::endl;
 * a = INNC::Tensor::ones({4, 3, 2}, INNC::i32);
 * std::cout << a.to_string << std::endl;
 * \endcode
 *
 */
class Tensor {
private:
  std::shared_ptr<TensorFrame> fptr;
  Tensor(std::unique_ptr<TensorFrame> &tf);
  Tensor(std::unique_ptr<TensorFrame> &&tf);
  Tensor(std::shared_ptr<TensorFrame> &tf);

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
  Tensor(Tensor &&t);
  Tensor &operator=(const Tensor &t);
  Tensor &operator=(Tensor &&t);
  Tensor(const SizeVec &sizes, types dtype);
  ~Tensor();
  void backward();
  std::string to_string() const;
  const SizeVec size() const;
  const SizeVec stride() const;
  static Tensor zeros(const SizeVec &size, types t);
  static Tensor ones(const SizeVec &size, types t);
  static Tensor zeros_like(const Tensor &t);
  static Tensor ones_like(const Tensor &t);
  static Tensor from_blob(void *data, const SizeVec &sizes, types dtype);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  Tensor type(types t);
  Tensor operator[](const std::string &slice);
  static Tensor transpose(const Tensor &input, size_t dim0, size_t dim1);
  static Tensor reshape(const Tensor &input,
                        const std::initializer_list<int> &sizes);
  Tensor reshape(const std::initializer_list<int> &sizes);
  Tensor reshape_as(const Tensor &input);
  Tensor &operator+=(const Tensor &rhs);
  bool requires_grad() const noexcept;
  void requires_grad(bool b);
  bool retain_grad() const noexcept;
  void retain_grad(bool b) noexcept;
  Tensor grad() const noexcept;
  Tensor sum() const;
  void zero_grad() const noexcept;
  bool is_contiguous() const noexcept; // TODO 2
  Tensor contiguous() const;           // TODO 2
  friend Tensor operator+(const Tensor &lhs, const Tensor &rhs);
  friend Tensor operator*(const Tensor &lhs, const Tensor &rhs);
  friend class Backward;
};

} // namespace INNC
