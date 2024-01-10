#include "INNC/dispatcher.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/storage.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/traits.hpp"
#include "INNC/view.hpp"
#include <cstring>
#include <queue>

namespace INNC {
Tensor::Tensor() = default;
Tensor::Tensor(Tensor &&t) = default;
Tensor::Tensor(std::unique_ptr<TensorImpl> &&tf) : fptr(std::move(tf)){};
Tensor::Tensor(std::shared_ptr<TensorImpl> tf) : fptr(tf){};
Tensor &Tensor::operator=(const Tensor &t) = default;
Tensor &Tensor::operator=(Tensor &&t) = default;
Tensor::Tensor(const SizeVec &sizes, types dtype)
    : fptr(TensorImpl::create(dtype, StridedView{sizes})){};
Tensor::~Tensor() = default;

void Tensor::backward() { fptr->backward(); }

std::string Tensor::to_string() const {
  if (fptr == nullptr)
    return "[]";
  return fptr->to_string();
}

const SizeVec Tensor::size() const { return fptr->view->sizes; }

const DiffVec Tensor::stride() const { return fptr->stride(); }

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

Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr + *rhs.fptr);
}

Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr * *rhs.fptr);
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
  return Tensor(tf);
}

void Tensor::zero_grad() const noexcept { fptr->zero_grad(); }

Tensor Tensor::operator[](const std::string &slice) {
  return Tensor((*this->fptr)[slice]);
}

Tensor Tensor::transpose(const Tensor &input, size_t dim0, size_t dim1) {
  auto tf = TensorImpl::transpose(input.fptr, dim0, dim1);
  return Tensor(tf);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) {
  auto tf = TensorImpl::transpose(fptr, dim0, dim1);
  return Tensor(tf);
}

SizeVec initializer_list_to_SizeVec(const std::initializer_list<int> &sizes,
                                    size_t numel = 0) {
  std::vector<int> sizes_ = sizes;
  size_t minus_one_exist = false;
  size_t minus_one_index;
  for (size_t i = 0; i < sizes_.size(); i++) {
    if (sizes_[i] == -1) {
      run_expect(!minus_one_exist,
                 "Cannot have more than two minus ones in size.");
      minus_one_exist = true;
      minus_one_index = i;
    }
  }
  if (minus_one_exist) {
    size_t s = 1;
    for (size_t i = 0; i < sizes_.size(); i++)
      if (i != minus_one_index)
        s *= sizes_[i];
    run_expect(numel % s == 0, "The input of sizes and numel is illegal.");
    sizes_[minus_one_index] = numel / s;
  }
  SizeVec sizes__;
  for (size_t i = 0; i < sizes_.size(); i++) {
    sizes__.push_back(sizes_[i]);
  }
  return sizes__;
}

Tensor Tensor::reshape(const Tensor &input,
                       const std::initializer_list<int> &sizes) {
  auto tf = TensorImpl::reshape(
      input.fptr, initializer_list_to_SizeVec(sizes, input.fptr->numel()));
  return tf;
}
Tensor Tensor::reshape(const std::initializer_list<int> &sizes) {
  auto tf =
      TensorImpl::reshape(fptr, initializer_list_to_SizeVec(sizes, numel()));
  return tf;
}
Tensor Tensor::reshape_as(const Tensor &input) {
  auto tf = TensorImpl::reshape(fptr, input.size());
  return tf;
}

} // namespace INNC
