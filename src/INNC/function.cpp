#include "INNC/function.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/tensor.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include <memory>

namespace INNC {

size_t Backward::global_back_version = 0;

Backward::Backward(
    TensorImpl *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_tfs)
    : input_tfs(input_tfs), this_tf(this_tf), n_outway(0),
      accumulated_n_outway(0), back_version(0) {}

Backward::~Backward() = default;

TensorImpl &Backward::get_out_grad() { return *this_tf->grad; }

void AddBack::step_back() {
  input_tfs[0]->try_accumulate_grad(&get_out_grad());
  input_tfs[1]->try_accumulate_grad(&get_out_grad());
}

void SubBack::step_back() {
  input_tfs[0]->try_accumulate_grad(&get_out_grad());
  input_tfs[1]->try_accumulate_grad(&get_out_grad(),
                                    TensorImpl::create(-1).get());
}

void MulBack::step_back() {
  input_tfs[0]->try_accumulate_grad(input_tfs[1].get(), &get_out_grad());
  input_tfs[1]->try_accumulate_grad(input_tfs[0].get(), &get_out_grad());
}

void DivBack::step_back() {
  // TODO 1
}

void SumBack::step_back() {
  input_tfs[0]->try_accumulate_grad(TensorImpl::ones_like(*input_tfs[0]).get());
}

void NoBack::step_back() { input_tfs[0]->try_accumulate_grad(nullptr); }

void CloneBack::step_back() {
  input_tfs[0]->try_accumulate_grad(&get_out_grad());
}

CatBack::CatBack(
    TensorImpl *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_tfs,
    const size_t dim)
    : Backward(this_tf, input_tfs), dim(dim) {}

void CatBack::step_back() {
  size_t s = 0;
  for (size_t i = 0; i < input_tfs.size(); i++) {
    auto ret = INNC::TensorImpl::create(
        this_tf->dtype,
        std::make_unique<StridedLayout>(
            input_tfs[i]->view->sizes,
            dynamic_cast<StridedLayout *>(this_tf->view.get())->strides,
            s * dynamic_cast<StridedLayout *>(this_tf->view.get())
                    ->strides[dim]),
        this_tf->grad->data_);
    input_tfs[i]->try_accumulate_grad(ret.get());
    s += input_tfs[i]->view->sizes[dim];
  }
}

} // namespace INNC
