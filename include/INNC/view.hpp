#pragma once
#include "INNC/storage.hpp"
#include "INNC/types.hpp"

namespace INNC {
class StridedView {
public:
  SizeVec sizes;
  DiffVec strides;
  size_t offset;
  StridedView();
  StridedView(const SizeVec &sizes);
  StridedView(const SizeVec &sizes, const DiffVec &strides);
  StridedView(const SizeVec &sizes, const DiffVec &strides,
              const size_t offset);
  StridedView(const StridedView &sv);
  StridedView(StridedView &&sv);
  size_t dim() const noexcept;
  size_t numel() const noexcept;
};
} // namespace INNC
