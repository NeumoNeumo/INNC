#pragma once

#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include "INNC/view.hpp"

namespace INNC {

class Backward;

// All member functions defined here never set up grad_fn. Also see `Tensor`.
class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
  TensorImpl(const TensorImpl &) = delete;
  TensorImpl(TensorImpl &&t) = delete;
  TensorImpl &operator=(const TensorImpl &t) = delete;
  struct Private {};

public:
  TensorImpl(types dtype, layouts dlayout, const std::shared_ptr<View> &view,
             const std::shared_ptr<UntypedStorage> &data_, Private);
  const std::shared_ptr<UntypedStorage> data_;
  std::shared_ptr<TensorImpl> grad;
  const std::shared_ptr<View> view;
  const types dtype;
  const layouts dlayout;
  bool requires_grad; // TODO 3 stricter encapsulation
  bool retain_grad;
  size_t _version;
  std::unique_ptr<Backward> grad_fn;

  static std::shared_ptr<TensorImpl> create(types dtype,
                                            const std::shared_ptr<View> &view,
                                            bool prealloc = true,
                                            layouts dlayout = layouts::strided);
  static std::shared_ptr<TensorImpl>
  create(types dtype, const std::shared_ptr<View> &view,
         const std::shared_ptr<UntypedStorage> &data_,
         layouts dlayout = layouts::strided);
  static std::shared_ptr<TensorImpl> create(types dtype, StridedView &&view,
                                            bool prealloc = true,
                                            layouts dlayout = layouts::strided);

  ~TensorImpl();
  static TensorImpl make_stub();
  size_t cnt_from_index(const SizeVec &index) const;
  size_t dim() const noexcept;
  std::string to_string() const;
  static std::shared_ptr<TensorImpl> zeros(const SizeVec &sizes, types dtype);
  static std::shared_ptr<TensorImpl> ones(const SizeVec &sizes, types dtype);
  static std::shared_ptr<TensorImpl> zeros_like(const TensorImpl &t);
  static std::shared_ptr<TensorImpl> ones_like(const TensorImpl &t);
  static std::shared_ptr<TensorImpl> from_blob(void *data, const SizeVec &sizes,
                                               types dtype);
  size_t numel() const noexcept;
  void release() noexcept;
  INNC::types type() const;
  std::shared_ptr<TensorImpl> type(types t);
  DiffVec stride() const;
  std::shared_ptr<TensorImpl> operator[](const std::string &slice);
  static std::shared_ptr<TensorImpl>
  transpose(const std::shared_ptr<TensorImpl> &input, size_t dim0, size_t dim1);
  std::shared_ptr<TensorImpl> sum();
  void zero_grad() const noexcept;
  friend std::shared_ptr<TensorImpl> operator+(TensorImpl &l, TensorImpl &r);
  friend std::shared_ptr<TensorImpl> operator*(TensorImpl &l, TensorImpl &r);
  TensorImpl &operator+=(const TensorImpl &rhs);
  void try_accumulate_grad(TensorImpl *tf_w, TensorImpl *tf_o = nullptr);
  friend std::unique_ptr<TensorImpl> no_grad_add(const TensorImpl &lhs,
                                                 const TensorImpl &rhs);
  friend void check_same_size(const TensorImpl &lhs, const TensorImpl &rhs);
  friend class Backward;
  void backward();
};
} // namespace INNC
