#include "INNC/tensor.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include <cstddef>
#include <cstdlib>
#include <iostream>

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

size_t Tensor::cnt_from_index(const SizeVec &index) const noexcept {
  assertm(index.size() == this->sizes.size(), "index mismatches sizes");
  size_t pos = this->offset;
  for (size_t i = 0; i < index.size(); ++i) {
    pos += this->strides[i] * index[i];
  }
  return pos;
}

const SizeVec Tensor::size() const { return this->sizes; }
const SizeVec Tensor::stride() const { return this->strides; }

std::string Tensor::to_string_helper(std::ptrdiff_t offset, size_t l) const {
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

std::string Tensor::to_string() const {
  if (this->sizes.size() == 0)
    return "[]";
  return to_string_helper();
}

Tensor Tensor::zeros(const std::vector<size_t> &sizes, types dtype) {
  Tensor ret(sizes, dtype);
  auto start = ret.data_.get();
  std::fill(start, start + ret.data_.get_size(), 0);
  return ret;
}

Tensor::~Tensor() = default;

Tensor::Tensor(const SizeVec &sizes, types dtype) : dtype(dtype), sizes(sizes) {
  run_expect(sizes.size() != 0, "`size` must have at least one dimension.");
  size_t space = size_of(dtype);
  for (auto s : sizes) {
    space *= s;
  }
  run_expect(space != 0, "All dimensions of `size` must be non-zero.");
  this->data_.create(space);
  this->strides.resize(sizes.size());
  this->strides[sizes.size() - 1] = 1;
  for (int i = sizes.size() - 1; i >= 1; --i) {
    this->strides[i - 1] = this->strides[i] * sizes[i];
  }
  this->offset = 0;
  this->is_leaf = true;
  this->requires_grad = false;
  this->retain_grad = false;
  this->_version = 0;
  this->grad_fn = nullptr;
}

template <typename L, typename R>
void tensor_add(void *dst, size_t elem_num, void *l, void *r) {
  using dst_type = std::common_type_t<L, R>;
  for (size_t i = 0; i < elem_num; ++i) {
    *(static_cast<dst_type *>(dst) + i) =
        *(static_cast<L *>(l) + i) + *(static_cast<R *>(r) + i);
  }
}

generate_binary_op_list(tensor_add);

Tensor Tensor::operator+(const Tensor &rhs) {
  bool dim_eq = true;
  size_t elem_num = 1;
  if (this->sizes.size() == rhs.sizes.size()) {
    for (size_t i = 0; i < this->sizes.size(); ++i) {
      elem_num *= sizes[i];
      if (sizes[i] != rhs.sizes[i])
        dim_eq = false;
    }
  } else
    dim_eq = false;
  if (!dim_eq)
    std::cerr << "Added tensors' dimension must match. Trying to add " +
                     sizes.to_string() + " with " + rhs.sizes.to_string()
              << std::endl
              << "Broadcasting has not been supported yet.";
  Tensor ret(this->size(), INNC::common_type(this->dtype, rhs.dtype));
  auto ret_mat_start = ret.data_.get();
  tensor_add_spec_list[this->dtype][rhs.dtype](ret_mat_start, elem_num,
                                               data_.get(), rhs.data_.get());
  return ret;
}

size_t Tensor::elem_num() const noexcept {
  size_t ret = 1;
  for (auto i : this->sizes)
    ret *= i;
  return ret;
}

template <typename TensorType, typename NumberType>
void tensor_fill(void *tdata, size_t elem_num, void *ndata, void *_discard) {
  TensorType num = *static_cast<NumberType *>(ndata);
  for (size_t i = 0; i < elem_num; ++i)
    *(static_cast<TensorType *>(tdata) + i) = num;
}

generate_binary_op_list(tensor_fill);

Tensor Tensor::ones(const std::vector<size_t> &size, types t) {
  Tensor ret(size, t);
  uint8_t i8_one = 1;
  tensor_fill_spec_list[t][i8](ret.data_.get(), ret.elem_num(), &i8_one,
                               nullptr);
  return ret;
}

} // namespace INNC
