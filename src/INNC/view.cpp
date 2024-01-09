#include "INNC/view.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include <iostream>

namespace INNC {

std::string to_string(layouts l) { return layout_to_string_aux_arr[l]; }

View::View() : sizes{} {}

View::View(const SizeVec &sizes) : sizes(sizes) {}

View::View(SizeVec &&sizes) : sizes(std::move(sizes)) {}

View::~View() = default;

size_t View::dim() const noexcept { return sizes.size(); }

size_t View::numel() const noexcept {
  size_t num = 1;
  for (auto s : sizes)
    num *= s;
  return num;
}

StridedView::StridedView() = default;

StridedView::StridedView(const SizeVec &sizes) : View(sizes), offset(0) {
  auto sn = sizes.size();
  if (sn == 0)
    return;
  strides.resize(sn);
  strides[sn - 1] = 1;
  for (size_t idx = sn - 1; idx > 0; --idx) {
    strides[idx - 1] = strides[idx] * sizes[idx];
  }
}

StridedView::StridedView(const SizeVec &sizes, const DiffVec &strides)
    : View(sizes), strides(strides), offset(0) {}

StridedView::StridedView(const SizeVec &sizes, const DiffVec &strides,
                         const size_t offset)
    : View(sizes), strides(strides), offset(offset) {}

StridedView::StridedView(const StridedView &sv) = default;

StridedView::StridedView(StridedView &&sv) = default;

size_t StridedView::cnt_from_index(const SizeVec &index) const {
  assertm(index.size() == dim(), "index mismatches sizes");
  size_t pos = offset;
  for (size_t i = 0; i < index.size(); ++i) {
    pos += strides[i] * index[i];
  }
  return pos;
}

std::string StridedView::to_string_from_helper(const UntypedStorage &data_,
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

std::string StridedView::to_string_from(const UntypedStorage &data_,
                                        types dtype) const {
  if (dim() == 0)
    return INNC::innc_type_to_string(
        data_.get_blob() + offset * INNC::size_of(dtype), dtype);
  return to_string_from_helper(data_, dtype, offset * INNC::size_of(dtype));
}

} // namespace INNC
