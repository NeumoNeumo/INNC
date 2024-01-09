#pragma once
#include "INNC/storage.hpp"
#include "INNC/types.hpp"

namespace INNC {

// BEGIN hard-coded variables
enum layouts { strided, sparse };
const char *const layout_to_string_aux_arr[] = {"strided", "sparse"};
// END hard-coded variables

std::string to_string(layouts l);

class View {
public:
  SizeVec sizes;
  View();
  View(const SizeVec &sizes);
  View(SizeVec &&sizes);
  virtual ~View();
  virtual size_t dim() const noexcept;
  virtual size_t numel() const noexcept;
  virtual std::string to_string_from(const UntypedStorage &data_,
                                     types dtype) const = 0;
};

class StridedView : public View {
  std::string to_string_from_helper(const UntypedStorage &data_, types dtype,
                                    std::ptrdiff_t offset = 0,
                                    size_t depth = 0) const;

public:
  DiffVec strides;
  size_t offset;
  StridedView();
  StridedView(const SizeVec &sizes);
  StridedView(const SizeVec &sizes, const DiffVec &strides);
  StridedView(const SizeVec &sizes, const DiffVec &strides,
              const size_t offset);
  StridedView(const StridedView &sv);
  StridedView(StridedView &&sv);
  size_t cnt_from_index(const SizeVec &index) const;
  std::string to_string_from(const UntypedStorage &data_,
                             types dtype) const override;
};
} // namespace INNC
