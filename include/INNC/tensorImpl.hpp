#pragma once

#include "INNC/layouts.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"

namespace INNC {

class Backward;

// All member functions defined here never set up grad_fn. Also see `Tensor`.
class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
  TensorImpl(const TensorImpl &) = delete;
  TensorImpl(TensorImpl &&t) = delete;
  TensorImpl &operator=(const TensorImpl &t) = delete;
  struct Private {};

public:
  TensorImpl(types dtype, layouts dlayout, const std::shared_ptr<Layout> &view,
             const std::shared_ptr<UntypedStorage> &data_, Private);
  const std::shared_ptr<UntypedStorage> data_;
  std::shared_ptr<TensorImpl> grad;
  const std::shared_ptr<Layout> view;
  const types dtype;
  const layouts dlayout;
  bool requires_grad; // TODO 3 stricter encapsulation
  bool retain_grad;
  size_t _version;
  std::unique_ptr<Backward> grad_fn;

  static std::shared_ptr<TensorImpl> create(types dtype,
                                            const std::shared_ptr<Layout> &view,
                                            bool prealloc = true,
                                            layouts dlayout = layouts::strided);
  static std::shared_ptr<TensorImpl>
  create(types dtype, const std::shared_ptr<Layout> &view,
         const std::shared_ptr<UntypedStorage> &data_,
         layouts dlayout = layouts::strided);
  static std::shared_ptr<TensorImpl> create(types dtype, StridedLayout &&view,
                                            bool prealloc = true,
                                            layouts dlayout = layouts::strided);
  static std::shared_ptr<TensorImpl> create(std::int8_t a);
  static std::shared_ptr<TensorImpl> create(std::int16_t a);
  static std::shared_ptr<TensorImpl> create(std::int32_t a);
  static std::shared_ptr<TensorImpl> create(std::int64_t a);
  static std::shared_ptr<TensorImpl> create(float a);
  static std::shared_ptr<TensorImpl> create(double a);
  ~TensorImpl();
  static TensorImpl make_stub();
  size_t cnt_from_index(const SizeVec &index) const;
  size_t cnt_from_aug_index(const SizeVec &index) const;
  size_t dim() const noexcept;
  std::string to_string() const;
  static std::shared_ptr<TensorImpl> zeros(const SizeVec &sizes, types dtype);
  static std::shared_ptr<TensorImpl> ones(const SizeVec &sizes, types dtype);
  static std::shared_ptr<TensorImpl> zeros_like(const TensorImpl &t);
  static std::shared_ptr<TensorImpl> ones_like(const TensorImpl &t);
  static std::shared_ptr<TensorImpl> full(const SizeVec &sv, std::int64_t num,
                                          types dtype);
  static std::shared_ptr<TensorImpl> full(const SizeVec &sv, double num,
                                          types dtype);
  static std::shared_ptr<TensorImpl> eye(size_t n, types dtype = types::i8);
  static std::shared_ptr<TensorImpl> eye(size_t n, size_t m,
                                         types dtype = types::i8);
  static std::shared_ptr<TensorImpl> from_blob(void *data, const SizeVec &sizes,
                                               types dtype);
  static std::shared_ptr<TensorImpl> randn(const SizeVec &sizes, types dtype);
  static std::shared_ptr<TensorImpl> randn_like(const TensorImpl &t);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  std::shared_ptr<TensorImpl> type(types t);
  SignedVec stride() const;
  SizeVec size() const;
  size_t size(int d) const;
  std::shared_ptr<TensorImpl> operator[](const std::string &slice);
  static std::shared_ptr<TensorImpl>
  transpose(const std::shared_ptr<TensorImpl> &input, size_t dim0, size_t dim1);
  static std::shared_ptr<TensorImpl>
  reshape(const std::shared_ptr<TensorImpl> &input, const SizeVec &sizes);
  std::shared_ptr<TensorImpl> reshape(const SizeVec &sizes);
  static std::shared_ptr<TensorImpl>
  reshape(const std::shared_ptr<TensorImpl> &input, const SignedVec &sizes);
  std::shared_ptr<TensorImpl> reshape(const SignedVec &sizes);
  std::shared_ptr<TensorImpl> reshape_as(const TensorImpl &sizes);
  static std::shared_ptr<TensorImpl>
  cat(const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_ts,
      const size_t dim);
  std::shared_ptr<TensorImpl> sum();
  std::shared_ptr<TensorImpl> mean();
  std::shared_ptr<TensorImpl> abs();
  std::shared_ptr<TensorImpl> max();
  std::shared_ptr<TensorImpl> min();
  void zero_grad() const noexcept;
  std::shared_ptr<TensorImpl> operator-();
  std::shared_ptr<TensorImpl> operator+();
  friend std::shared_ptr<TensorImpl> operator+(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator-(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator*(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator/(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator>(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator<(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator>=(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator<=(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator==(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator!=(TensorImpl &l, TensorImpl &r);
  TensorImpl &operator+=(const TensorImpl &rhs);
  friend std::unique_ptr<TensorImpl> no_grad_add(const TensorImpl &lhs,
                                                 const TensorImpl &rhs);
  friend void check_same_size(const TensorImpl &lhs, const TensorImpl &rhs);
  friend class Backward;
  void backward();
  bool is_contiguous() const noexcept;
  std::shared_ptr<TensorImpl> contiguous();
  std::shared_ptr<TensorImpl> clone();
  std::shared_ptr<TensorImpl> detach();
  bool all() const;
};
} // namespace INNC
