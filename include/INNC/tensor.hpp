#pragma once

#include "storage.hpp"
#include "types.hpp"
#include <memory>

namespace INNC {
class Backward;
class Tensor;

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

  bool requires_grad; // TODO stricter encapsulation
  bool retain_grad;
  size_t _version;
  std::unique_ptr<Backward> grad_fn;

  TensorFrame() = delete;
  TensorFrame(const TensorFrame &) = delete;
  TensorFrame(TensorFrame &&t);
  TensorFrame &operator=(const TensorFrame &t) = delete;
  TensorFrame(types dtype, const SizeVec &sizes, const SizeVec &strides,
              const size_t offset);
  ~TensorFrame();
  std::string to_string() const;
  static std::unique_ptr<TensorFrame> make_tf(types dtype,
                                              const SizeVec &sizes);
  static std::unique_ptr<TensorFrame> zeros(const std::vector<size_t> &sizes,
                                            types dtype);
  static std::unique_ptr<TensorFrame> ones(const std::vector<size_t> &sizes,
                                           types dtype);
  static std::unique_ptr<TensorFrame> zeros_like(const TensorFrame &t);
  static std::unique_ptr<TensorFrame> ones_like(const TensorFrame &t);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  std::unique_ptr<TensorFrame> type(types t);
  std::unique_ptr<TensorFrame> operator[](std::string slice);
  TensorFrame &operator+=(const TensorFrame &rhs);
  void try_accumulate_grad(TensorFrame &tf);
  friend std::unique_ptr<TensorFrame> operator+(const TensorFrame &lhs,
                                                const TensorFrame &rhs);
  friend void check_same_size(const TensorFrame &lhs, const TensorFrame &rhs);
  friend class Backward;
  void backward();
};

class Tensor {
private:
  std::shared_ptr<TensorFrame> fptr;
  Tensor(std::unique_ptr<TensorFrame> &tf);
  Tensor(std::unique_ptr<TensorFrame> &&tf);
  Tensor(std::shared_ptr<TensorFrame> &tf);

public:
  Tensor();
  Tensor(const Tensor &) = delete;
  Tensor(Tensor &&t);
  Tensor &operator=(const Tensor &t);
  Tensor &operator=(Tensor &&t);
  Tensor(const SizeVec &sizes, types dtype);
  ~Tensor();
  void backward();
  std::string to_string() const;
  const SizeVec size() const;
  const SizeVec stride() const;
  static Tensor zeros(const std::vector<size_t> &size, types t);
  static Tensor ones(const std::vector<size_t> &size, types t);
  static Tensor zeros_like(const Tensor &t);
  static Tensor ones_like(const Tensor &t);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  Tensor type(types t);
  Tensor operator[](std::string slice); // TODO WARN
  Tensor &operator+=(const Tensor &rhs);
  bool requires_grad() const noexcept;
  void requires_grad(bool b) noexcept;
  bool retain_grad() const noexcept;
  void retain_grad(bool b) noexcept;
  Tensor grad() const noexcept;
  friend Tensor operator+(const Tensor &lhs, const Tensor &rhs);
  friend class Backward;
};

} // namespace INNC
