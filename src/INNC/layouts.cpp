#include "INNC/layouts.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/storage.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include <iostream>

namespace INNC {

std::string to_string(layouts l) { return layout_to_string_aux_arr[l]; }

Layout::Layout() : sizes{} {}

Layout::Layout(const SizeVec &sizes) : sizes(sizes) {}

Layout::Layout(SizeVec &&sizes) : sizes(std::move(sizes)) {}

Layout::~Layout() = default;

size_t Layout::dim() const noexcept { return sizes.size(); }

size_t Layout::numel() const noexcept {
  size_t num = 1;
  for (auto s : sizes)
    num *= s;
  return num;
}

StridedLayout::StridedLayout() = default;

StridedLayout::StridedLayout(const SizeVec &sizes) : Layout(sizes), offset(0) {
  auto sn = sizes.size();
  if (sn == 0)
    return;
  strides.resize(sn);
  strides[sn - 1] = 1;
  for (size_t idx = sn - 1; idx > 0; --idx) {
    strides[idx - 1] = strides[idx] * sizes[idx];
  }
}

StridedLayout::StridedLayout(const SizeVec &sizes, const DiffVec &strides)
    : Layout(sizes), strides(strides), offset(0) {}

StridedLayout::StridedLayout(const SizeVec &sizes, const DiffVec &strides,
                             const size_t offset)
    : Layout(sizes), strides(strides), offset(offset) {}

StridedLayout::StridedLayout(const StridedLayout &sv) = default;

StridedLayout::StridedLayout(StridedLayout &&sv) = default;

size_t StridedLayout::cnt_from_index(const SizeVec &index) const {
  run_expect(index.size() == dim(), "index mismatches sizes");
  size_t pos = offset;
  for (size_t i = 0; i < index.size(); ++i) {
    pos += strides[i] * index[i];
  }
  return pos;
}

std::string StridedLayout::to_string_from_helper(const UntypedStorage &data_,
                                                 types dtype,
                                                 std::ptrdiff_t offset,
                                                 size_t depth) const {
  if (depth == dim()) {
    void *ptr = data_.get_blob() + offset;
    return INNC::innc_type_to_string(ptr, dtype);
  }
  bool begin = true;
  std::string ret = "[";
  for (size_t i = 0; i < sizes[depth]; ++i) {
    std::ptrdiff_t sub_offset =
        offset + i * strides[depth] * INNC::size_of(dtype);
    if (begin) {
      ret += to_string_from_helper(data_, dtype, sub_offset, depth + 1);
      begin = false;
      continue;
    }
    ret += ", " + to_string_from_helper(data_, dtype, sub_offset, depth + 1);
  }
  return ret + "]";
}

std::string StridedLayout::to_string_from(const UntypedStorage &data_,
                                          types dtype) const {
  if (dim() == 0)
    return INNC::innc_type_to_string(
        data_.get_blob() + offset * INNC::size_of(dtype), dtype);
  return to_string_from_helper(data_, dtype, offset * INNC::size_of(dtype));
}

bool StridedLayout::is_contiguous() {
  long long step = 1;
  for (long int d = dim() - 1; d >= 0; --d) {
    if (step != strides[d])
      return false;
    step *= sizes[d];
  }
  return true;
}

std::shared_ptr<TensorImpl> StridedLayout::contiguous_from(TensorImpl &t) {
  return t.clone();
}

} // namespace INNC
