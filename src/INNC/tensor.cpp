#include "INNC/tensor.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>

namespace INNC {

#define generate_binary_op_list(op)                                            \
  void (*op##_spec_list[types::Count][types::Count])(void *dst, size_t size,   \
                                                     void *l, void *r) = {     \
      {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,            \
       op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,           \
       op<std::int8_t, float>, op<std::int8_t, double>},                       \
      {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,          \
       op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,         \
       op<std::int16_t, float>, op<std::int16_t, double>},                     \
      {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,          \
       op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,         \
       op<std::int32_t, float>, op<std::int32_t, double>},                     \
      {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,          \
       op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,         \
       op<std::int64_t, float>, op<std::int64_t, double>},                     \
      {op<float, std::int8_t>, op<float, std::int16_t>,                        \
       op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>,     \
       op<float, double>},                                                     \
      {op<double, std::int8_t>, op<double, std::int16_t>,                      \
       op<double, std::int32_t>, op<double, std::int64_t>, op<double, float>,  \
       op<double, double>}}

size_t TensorFrame::cnt_from_index(const SizeVec &index) const noexcept {
  assertm(index.size() == this->sizes.size(), "index mismatches sizes");
  size_t pos = this->offset;
  for (size_t i = 0; i < index.size(); ++i) {
    pos += this->strides[i] * index[i];
  }
  return pos;
}

std::string TensorFrame::to_string_helper(std::ptrdiff_t offset,
                                          size_t l) const {
  if (l == this->sizes.size()) {
    void *ptr = this->data_.get() + offset;
    return INNC::innc_type_to_string(ptr, this->dtype);
  }
  bool begin = true;
  std::string ret = "[";
  for (size_t i = 0; i < this->sizes[l]; ++i) {
    std::ptrdiff_t sub_offset =
        offset + i * this->strides[l] * INNC::size_of(this->dtype);
    if (begin) {
      ret += to_string_helper(sub_offset, l + 1);
      begin = false;
      continue;
    }
    ret += ", " + to_string_helper(sub_offset, l + 1);
  }
  return ret + "]";
}

std::string TensorFrame::to_string() const {
  if (this->sizes.size() == 0)
    return "[]";
  return to_string_helper();
}

TensorFrame::~TensorFrame() = default;

TensorFrame::TensorFrame(TensorFrame &&t) = default;

// TODO rvalue optimization
std::unique_ptr<TensorFrame> TensorFrame::make_tf(types dtype,
                                                  const SizeVec &sizes) {
  run_expect(sizes.size() != 0, "`size` must have at least one dimension.");
  SizeVec strides;
  strides.resize(sizes.size());
  strides[sizes.size() - 1] = 1;
  for (int i = sizes.size() - 1; i >= 1; --i) {
    strides[i - 1] = strides[i] * sizes[i];
  }
  return std::make_unique<TensorFrame>(dtype, sizes, strides, 0);
}

TensorFrame::TensorFrame(types dtype, const SizeVec &sizes,
                         const SizeVec &strides, const size_t offset)
    : dtype(dtype), sizes(sizes), strides(strides), offset(offset) {
  size_t space = size_of(dtype);
  for (auto s : sizes) {
    space *= s;
  }
  run_expect(space != 0, "All dimensions of `size` must be non-zero.");
  this->data_.create(space);
  this->requires_grad = false;
  this->retain_grad = false;
  this->_version = 0;
  this->grad_fn.reset();
}

template <typename L, typename R>
void tensor_add(void *dst, size_t elem_num, void *l, void *r) {
  using dst_type = std::common_type_t<L, R>;
  for (size_t i = 0; i < elem_num; ++i) {
    *(static_cast<dst_type *>(dst) + i) =
        *(static_cast<L *>(l) + i) + *(static_cast<R *>(r) + i);
  }
}

template <typename L, typename R>
void tensor_sub(void *dst, size_t elem_num, void *l, void *r) {
  using dst_type = std::common_type_t<L, R>;
  for (size_t i = 0; i < elem_num; ++i) {
    *(static_cast<dst_type *>(dst) + i) =
        *(static_cast<L *>(l) + i) - *(static_cast<R *>(r) + i);
  }
}

template <typename L, typename R>
void tensor_mul(void *dst, size_t elem_num, void *l, void *r) {
  using dst_type = std::common_type_t<L, R>;
  for (size_t i = 0; i < elem_num; ++i) {
    *(static_cast<dst_type *>(dst) + i) =
        *(static_cast<L *>(l) + i) * *(static_cast<R *>(r) + i);
  }
}

template <typename L, typename R>
void tensor_div(void *dst, size_t elem_num, void *l, void *r) {
  using dst_type = std::common_type_t<L, R>;
  for (size_t i = 0; i < elem_num; ++i) {
    *(static_cast<dst_type *>(dst) + i) =
        *(static_cast<L *>(l) + i) / *(static_cast<R *>(r) + i);
  }
}

template <typename TensorType, typename NumberType>
void tensor_fill(void *tdata, size_t elem_num, void *ndata, void *_discard) {
  TensorType num = *static_cast<NumberType *>(ndata);
  for (size_t i = 0; i < elem_num; ++i)
    *(static_cast<TensorType *>(tdata) + i) = num;
}

template <typename FromType, typename ToType>
void tensor_to_type(void *todata, size_t elem_num, void *fromdata,
                    void *_discard) {
  for (size_t i = 0; i < elem_num; ++i)
    *(static_cast<ToType *>(todata) + i) =
        (ToType) * (static_cast<FromType *>(fromdata) + i);
}

generate_binary_op_list(tensor_add);
generate_binary_op_list(tensor_sub);
generate_binary_op_list(tensor_mul);
generate_binary_op_list(tensor_div);
generate_binary_op_list(tensor_fill);
generate_binary_op_list(tensor_to_type);

std::unique_ptr<TensorFrame> TensorFrame::ones(const SizeVec &sizes,
                                               types dtype) {
  auto ret = make_tf(dtype, sizes);
  uint8_t i8_one = 1;
  tensor_fill_spec_list[dtype][i8](ret->data_.get(), ret->numel(), &i8_one,
                                   nullptr);
  return ret;
}

std::unique_ptr<TensorFrame>
TensorFrame::zeros(const SizeVec &sizes, types dtype) {
  auto ret = make_tf(dtype, sizes);
  auto start = ret->data_.get();
  std::fill(start, start + ret->data_.get_size(), 0);
  return ret;
}

std::unique_ptr<TensorFrame> TensorFrame::zeros_like(const TensorFrame &t) {
  return TensorFrame::zeros(t.sizes, t.dtype);
}

std::unique_ptr<TensorFrame> TensorFrame::ones_like(const TensorFrame &t) {
  return TensorFrame::ones(t.sizes, t.dtype);
}

void check_same_size(const TensorFrame &lhs, const TensorFrame &rhs) {
  bool dim_eq = true;
  if (lhs.sizes.size() == rhs.sizes.size()) {
    for (size_t i = 0; i < lhs.sizes.size(); ++i)
      if (lhs.sizes[i] != rhs.sizes[i]) {
        dim_eq = false;
        break;
      }
  } else
    dim_eq = false;
  if (!dim_eq)
    std::cerr << "Added tensors' dimension must match. Trying to add " +
                     lhs.sizes.to_string() + " with " + rhs.sizes.to_string()
              << std::endl
              << "Broadcasting has not been supported yet.";
}

TensorFrame &TensorFrame::operator+=(const TensorFrame &rhs) {
  run_expect(
      !requires_grad,
      "This inplace operation cannot perform on a tensor that requires grad.");
  tensor_add_spec_list[dtype][rhs.dtype](data_.get(), numel(), data_.get(),
                                         rhs.data_.get());
  return *this;
}

std::unique_ptr<TensorFrame> operator+(TensorFrame &lhs, TensorFrame &rhs) {
  check_same_size(lhs, rhs);
  if (lhs.grad_fn.get() != nullptr)
    ++lhs.grad_fn->n_outway;
  if (rhs.grad_fn.get() != nullptr)
    ++rhs.grad_fn->n_outway;
  auto ret =
      TensorFrame::make_tf(INNC::common_type(lhs.dtype, rhs.dtype), lhs.sizes);
  auto ret_mat_start = ret->data_.get();
  tensor_add_spec_list[lhs.dtype][rhs.dtype](ret_mat_start, lhs.numel(),
                                             lhs.data_.get(), rhs.data_.get());
  if (!lhs.requires_grad && !rhs.requires_grad)
    return ret;
  ret->requires_grad = true;
  ret->grad_fn.reset(new AddBack(ret.get(), {&lhs, &rhs}));
  return ret;
}

size_t TensorFrame::numel() const noexcept {
  size_t ret = 1;
  for (auto i : this->sizes)
    ret *= i;
  return ret;
}

void TensorFrame::release() noexcept { data_.reset(); }

inline INNC::types TensorFrame::type() const { return this->dtype; }

std::unique_ptr<TensorFrame> TensorFrame::type(types t) {
  auto tf = make_tf(t, sizes);
  tensor_to_type_spec_list[dtype][t](tf->data_.get(), numel(), data_.get(),
                                     nullptr);
  return tf;
}

void TensorFrame::try_accumulate_grad(TensorFrame &tf) {
  if (!requires_grad)
    return;
  if (grad_fn.get() != nullptr)
    ++grad_fn->accumulated_n_outway;
  if (grad.get() == nullptr)
    grad = TensorFrame::zeros_like(*this);
  *grad += tf;
}

void TensorFrame::backward() {
  run_expect(requires_grad,
             "Cannot backward from a tensor that does not require grad.");
  run_expect(grad_fn != nullptr,
             "Cannot backward from a tensor that has no grad_func");

  std::queue<TensorFrame *> q; // TODO multiprocessing
  q.push(this);
  grad_fn->visited = false;
  while (!q.empty()) { // init
    auto t = q.front();
    q.pop();
    for (auto it : t->grad_fn->input_tfs) {
      if (it->grad_fn.get() != nullptr && it->grad_fn->visited) {
        it->grad_fn->visited = false;
        it->grad_fn->accumulated_n_outway = 0;
        q.push(it);
      }
    }
  }
  q.push(this);
  grad_fn->visited = true;
  grad = ones_like(*this);
  while (!q.empty()) {
    auto t = q.front();
    q.pop();
    t->grad_fn->step_back();
    if (!t->retain_grad)
      t->grad.reset();
    for (auto it : t->grad_fn->input_tfs) {
      if (it->grad_fn.get() != nullptr && !it->grad_fn->visited &&
          it->grad_fn->is_ready()) {
        it->grad_fn->visited = true;
        q.push(it);
      }
    }
  }
}

Tensor::Tensor() = default;
Tensor::Tensor(Tensor &&t) = default;
Tensor::Tensor(std::unique_ptr<TensorFrame> &tf) : fptr(std::move(tf)){};
Tensor::Tensor(std::unique_ptr<TensorFrame> &&tf) : fptr(std::move(tf)){};
Tensor::Tensor(std::shared_ptr<TensorFrame> &tf) : fptr(tf){};
Tensor &Tensor::operator=(const Tensor &t) = default;
Tensor &Tensor::operator=(Tensor &&t) = default;
Tensor::Tensor(const SizeVec &sizes, types dtype)
    : fptr(TensorFrame::make_tf(dtype, sizes)) {}
Tensor::~Tensor() = default;

void Tensor::backward() { fptr->backward(); }

std::string Tensor::to_string() const {
  if (fptr == nullptr)
    return "Tensor not exist.";
  return fptr->to_string();
}

const SizeVec Tensor::size() const { return fptr->sizes; }

const SizeVec Tensor::stride() const { return fptr->strides; }

Tensor Tensor::zeros(const SizeVec &size, types t) {
  return Tensor(TensorFrame::zeros(size, t));
}

Tensor Tensor::ones(const SizeVec &size, types t) {
  return Tensor(TensorFrame::ones(size, t));
}

Tensor Tensor::zeros_like(const Tensor &t) {
  return Tensor(TensorFrame::zeros_like(*t.fptr));
}

Tensor Tensor::ones_like(const Tensor &t) {
  return Tensor(TensorFrame::ones_like(*t.fptr));
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
  auto tf = *lhs.fptr + *rhs.fptr;
  return Tensor(tf);
}

bool Tensor::requires_grad() const noexcept { return fptr->requires_grad; }

void Tensor::requires_grad(bool b) noexcept { fptr->requires_grad = b; }

bool Tensor::retain_grad() const noexcept { return fptr->retain_grad; }
void Tensor::retain_grad(bool b) noexcept { fptr->retain_grad = b; }

Tensor Tensor::grad() const noexcept {
  if (fptr->grad.get() == nullptr)
    return Tensor();
  return Tensor(fptr->grad);
}

} // namespace INNC
