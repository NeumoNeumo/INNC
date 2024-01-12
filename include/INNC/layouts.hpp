#pragma once
#include "INNC/storage.hpp"
#include "INNC/types.hpp"

namespace INNC {
class TensorImpl;

// BEGIN hard-coded variables
enum layouts { strided, sparse };
const char *const layout_to_string_aux_arr[] = {"strided", "sparse"};
// END hard-coded variables

std::string to_string(layouts l);

class Layout {
public:
  SizeVec sizes;
  Layout();
  Layout(const SizeVec &sizes);
  Layout(SizeVec &&sizes);
  virtual ~Layout();
  virtual size_t dim() const noexcept;
  virtual size_t numel() const noexcept;
  virtual std::string to_string_from(const UntypedStorage &data_,
                                     types dtype) const = 0;
  virtual bool is_contiguous() = 0;
  virtual std::shared_ptr<TensorImpl> contiguous_from(TensorImpl &t) = 0;
};

class StridedLayout : public Layout {
  std::string to_string_from_helper(const UntypedStorage &data_, types dtype,
                                    std::ptrdiff_t offset = 0,
                                    size_t depth = 0) const;

public:
  SignedVec strides;
  size_t offset;
  StridedLayout();
  StridedLayout(const SizeVec &sizes);
  StridedLayout(const SizeVec &sizes, const SignedVec &strides);
  StridedLayout(const SizeVec &sizes, const SignedVec &strides,
                const size_t offset);
  StridedLayout(const StridedLayout &sv);
  StridedLayout(StridedLayout &&sv);
  size_t cnt_from_index(const SizeVec &index) const;
  std::string to_string_from(const UntypedStorage &data_,
                             types dtype) const override;
  bool is_contiguous() override;
  std::shared_ptr<TensorImpl> contiguous_from(TensorImpl &t) override;
};
} // namespace INNC
