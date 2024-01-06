#include "INNC/view.hpp"
#include <iostream>

namespace INNC {
StridedView::StridedView() = default;

StridedView::StridedView(const SizeVec &sizes) : sizes(sizes), offset(0) {
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
    : sizes(sizes), strides(strides), offset(0) {}

StridedView::StridedView(const SizeVec &sizes, const DiffVec &strides,
                         const size_t offset)
    : sizes(sizes), strides(strides), offset(offset) {}

StridedView::StridedView(const StridedView &sv) = default;

StridedView::StridedView(StridedView &&sv) = default;

size_t StridedView::dim() const noexcept { return sizes.size(); }

size_t StridedView::numel() const noexcept {
  size_t num = 1;
  for (auto s : sizes)
    num *= s;
  return num;
}

} // namespace INNC
