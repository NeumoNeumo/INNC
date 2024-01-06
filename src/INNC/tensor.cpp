#include "INNC/dispatcher.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/storage.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/traits.hpp"
#include "INNC/utils/utils.hpp"
#include "INNC/view.hpp"
#include <cstring>
#include <queue>

namespace INNC {
Tensor::Tensor() = default;
Tensor::Tensor(Tensor &&t) = default;
Tensor::Tensor(std::unique_ptr<TensorImpl> &tf) : fptr(std::move(tf)){};
Tensor::Tensor(std::unique_ptr<TensorImpl> &&tf) : fptr(std::move(tf)){};
Tensor::Tensor(std::shared_ptr<TensorImpl> &tf) : fptr(tf){};
Tensor &Tensor::operator=(const Tensor &t) = default;
Tensor &Tensor::operator=(Tensor &&t) = default;
Tensor::Tensor(const SizeVec &sizes, types dtype)
    : fptr(std::make_unique<TensorImpl>(dtype, StridedView{sizes})){};
Tensor::~Tensor() = default;

void Tensor::backward() { fptr->backward(); }

std::string Tensor::to_string() const {
  if (fptr == nullptr)
    return "[]";
  return fptr->to_string();
}

const SizeVec Tensor::size() const { return fptr->view->sizes; }

const DiffVec Tensor::stride() const { return fptr->view->strides; }

Tensor Tensor::zeros(const SizeVec &size, types t) {
  return Tensor(TensorImpl::zeros(size, t));
}

Tensor Tensor::ones(const SizeVec &size, types t) {
  return Tensor(TensorImpl::ones(size, t));
}

Tensor Tensor::zeros_like(const Tensor &t) {
  return Tensor(TensorImpl::zeros_like(*t.fptr));
}

Tensor Tensor::ones_like(const Tensor &t) {
  return Tensor(TensorImpl::ones_like(*t.fptr));
}

Tensor Tensor::from_blob(void *data, const SizeVec &sizes, types dtype) {
  return Tensor(TensorImpl::from_blob(data, sizes, dtype));
}

size_t Tensor::numel() const noexcept { return fptr->numel(); }

void Tensor::release() noexcept { fptr->release(); }

INNC::types Tensor::type() const { return fptr->type(); }

Tensor Tensor::type(types t) { return Tensor(fptr->type(t)); }

Tensor &Tensor::operator+=(const Tensor &rhs) {
  *fptr += *rhs.fptr;
  return *this;
}

template <typename L, typename R>
void tensor_add(TensorImpl *dst, TensorImpl *l, TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<L *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) =
        *(l_ptr + l->cnt_from_index(sv)) + *(r_ptr + r->cnt_from_index(sv));
  });
}

template <typename L, typename R>
void tensor_mul(TensorImpl *dst, TensorImpl *l, TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<L *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) =
        *(l_ptr + l->cnt_from_index(sv)) * *(r_ptr + r->cnt_from_index(sv));
  });
}

generate_binary_op_helper(tensor_add);
generate_binary_op_helper(tensor_mul);

Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
  auto tf =
      apply_binary_operator<tensor_add_helper, AddBack>(lhs.fptr, rhs.fptr);
  return Tensor(tf);
}

Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  auto tf =
      apply_binary_operator<tensor_mul_helper, MulBack>(lhs.fptr, rhs.fptr);
  return Tensor(tf);
}

TensorImpl &TensorImpl::operator+=(const TensorImpl &rhs) {
  run_expect(
      !requires_grad,
      "This inplace operation cannot perform on a tensor that requires grad.");
  tensor_add_helper::dispatch(dtype, rhs.dtype)(this, this,
                                                const_cast<TensorImpl *>(&rhs));
  return *this;
}

bool Tensor::requires_grad() const noexcept { return fptr->requires_grad; }

void Tensor::requires_grad(bool b) {
  if (b)
    run_expect(fptr->type() == f32 || fptr->type() == f64,
               "Only matrices of floating point number can require grad");
  fptr->requires_grad = b;
}

bool Tensor::retain_grad() const noexcept { return fptr->retain_grad; }
void Tensor::retain_grad(bool b) noexcept { fptr->retain_grad = b; }

Tensor Tensor::grad() const noexcept {
  if (fptr->grad.get() == nullptr)
    return Tensor();
  return Tensor(fptr->grad);
}

Tensor Tensor::sum() const {
  auto tf = fptr->sum();
  if (fptr->requires_grad) {
    tf->requires_grad = true;
    tf->grad_fn.reset(new SumBack(tf.get(), {fptr}));
  }
  return Tensor(tf);
}

void Tensor::zero_grad() const noexcept { fptr->zero_grad(); }

Tensor Tensor::operator[](const std::string &slice) {
  return Tensor((*this->fptr)[slice]);
}

SizeVec cnt_to_index(size_t x, const SizeVec &sizes) {
  SizeVec result;
  result.resize(sizes.size());
  for (size_t i = 0; i < sizes.size(); i++) {
    result[sizes.size() - i - 1] = x % sizes[sizes.size() - i - 1];
    x /= sizes[sizes.size() - i - 1];
  }
  return result;
}

Tensor Tensor::transpose(const Tensor &input, size_t dim0, size_t dim1) {
  auto tf = TensorImpl::transpose(input.fptr, dim0, dim1);
  return Tensor(tf);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) {
  auto tf = TensorImpl::transpose(fptr, dim0, dim1);
  return Tensor(tf);
}

} // namespace INNC
