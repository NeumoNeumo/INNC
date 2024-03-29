#include "INNC/dispatcher.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/layouts.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/traits.hpp"

namespace INNC {
Tensor::Tensor() = default;
Tensor::Tensor(Tensor &&t) = default;
Tensor::Tensor(std::unique_ptr<TensorImpl> &&tf) : fptr(std::move(tf)){};
Tensor::Tensor(std::shared_ptr<TensorImpl> tf) : fptr(tf){};
Tensor::Tensor(std::int8_t a) : Tensor(TensorImpl::create(a)) {}
Tensor::Tensor(std::int16_t a) : Tensor(TensorImpl::create(a)) {}
Tensor::Tensor(std::int32_t a) : Tensor(TensorImpl::create(a)) {}
Tensor::Tensor(std::int64_t a) : Tensor(TensorImpl::create(a)) {}
Tensor::Tensor(float a) : Tensor(TensorImpl::create(a)) {}
Tensor::Tensor(double a) : Tensor(TensorImpl::create(a)) {}
Tensor::Tensor(TensorInit init) : fptr{init.createImpl()} {}

Tensor &Tensor::operator=(const Tensor &t) = default;
Tensor &Tensor::operator=(Tensor &&t) = default;
Tensor &Tensor::operator=(TensorInit init) {
  fptr = init.createImpl();
  return *this;
}
Tensor::Tensor(const SizeVec &sizes, types dtype)
    : fptr(TensorImpl::create(dtype, StridedLayout{sizes})){};
Tensor::~Tensor() = default;

void Tensor::backward() { fptr->backward(); }

std::string Tensor::to_string() const {
  if (fptr == nullptr)
    return "[]";
  return fptr->to_string();
}

const SizeVec Tensor::size() const { return fptr->size(); }

const SignedVec Tensor::stride() const { return fptr->stride(); }

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

Tensor Tensor::eye(size_t n, types dtype) {
  return Tensor(TensorImpl::eye(n, dtype));
}
Tensor Tensor::eye(size_t n, size_t m, types dtype) {
  return Tensor(TensorImpl::eye(n, m, dtype));
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

Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr - *rhs.fptr);
}

Tensor Tensor::operator+() { return Tensor(+*fptr); }

Tensor Tensor::operator-() { return Tensor(-*fptr); }

Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr * *rhs.fptr);
}

Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr / *rhs.fptr);
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

Tensor Tensor::sum() const { return Tensor(fptr->sum()); }

Tensor Tensor::mean() const { return Tensor(fptr->mean()); }

Tensor Tensor::abs() const { return Tensor(fptr->abs()); }

Tensor Tensor::max() const { return Tensor(fptr->max()); }

Tensor Tensor::min() const { return Tensor(fptr->min()); }

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

Tensor Tensor::reshape(const Tensor &input, const SignedVec &sizes) {
  return TensorImpl::reshape(input.fptr, sizes);
}

Tensor Tensor::reshape(const SignedVec &sizes) {
  return TensorImpl::reshape(fptr, sizes);
}

Tensor Tensor::reshape_as(const Tensor &input) {
  return fptr->reshape_as(*input.fptr);
}

bool Tensor::is_contiguous() const noexcept { return fptr->is_contiguous(); }
Tensor Tensor::contiguous() const { return Tensor(fptr->contiguous()); }

Tensor Tensor::clone() const { return Tensor(fptr->clone()); }

Tensor Tensor::detach() const { return Tensor(fptr->detach()); }

bool Tensor::all() const { return fptr->all(); }

Tensor operator<(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr < *rhs.fptr);
}

Tensor operator>(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr > *rhs.fptr);
}

Tensor operator<=(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr <= *rhs.fptr);
}

Tensor operator>=(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr >= *rhs.fptr);
}

Tensor operator==(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr == *rhs.fptr);
}

Tensor operator!=(const Tensor &lhs, const Tensor &rhs) {
  return Tensor(*lhs.fptr != *rhs.fptr);
}

Tensor Tensor::randn(const SizeVec &sizes, types dtype) {
  return Tensor(TensorImpl::randn(sizes, dtype));
}

Tensor Tensor::randn_like(const Tensor &t) {
  return Tensor(TensorImpl::randn_like(*t.fptr));
}

Tensor Tensor::cat(const std::vector<Tensor> &input_tensors, const size_t dim) {
  std::vector<std::shared_ptr<INNC::TensorImpl>> input_tfs_;
  for (const auto &t : input_tensors)
    input_tfs_.emplace_back(t.fptr);
  return Tensor(TensorImpl::cat(input_tfs_, dim));
}

Tensor Tensor::full(const SizeVec &sv, std::int64_t num, types dtype) {
  return Tensor(TensorImpl::full(sv, num, dtype));
}

Tensor Tensor::full(const SizeVec &sv, double num, types dtype) {
  return Tensor(TensorImpl::full(sv, num, dtype));
}

std::shared_ptr<TensorImpl> TensorInit::createImpl() {
  if (shape.empty())
    return nullptr;
  auto ret = TensorImpl::create(dtype, StridedLayout{shape}, false);
  ret->data_->reset_blob(std::move(data_ptr));
  return ret;
}

} // namespace INNC
