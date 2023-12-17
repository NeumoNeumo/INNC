#pragma once

#include "storage.hpp"
#include "types.hpp"

namespace INNC {

class Backward;

class Tensor {
private:
  UntypedStorage data_;
  UntypedStorage grad;
  types dtype;
  SizeVec sizes;
  SizeVec strides;
  size_t offset;

  size_t cnt_from_index(const SizeVec &index) const noexcept;
  std::string to_string_helper(std::ptrdiff_t offset = 0, size_t l = 0) const;

public:
  bool is_leaf;
  bool requires_grad;
  bool retain_grad;
  size_t _version;
  Backward *grad_fn;

  Tensor(const SizeVec &sizes, types dtype);
  ~Tensor();
  std::string to_string() const;
  const SizeVec size() const;
  const SizeVec stride() const;
  static Tensor zeros(const std::vector<size_t> &size, types t);
  static Tensor ones(const std::vector<size_t> &size, types t);
  size_t elem_num() const noexcept;
  Tensor type(types t);
  Tensor operator[](std::string slice);
  Tensor operator+(const Tensor &rhs);
};

} // namespace INNC
