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

void MulBack::step_back() {
  input_tfs[0]->try_accumulate_grad(input_tfs[1].get(), &get_out_grad());
  input_tfs[1]->try_accumulate_grad(input_tfs[0].get(), &get_out_grad());
}

void SumBack::step_back() {
  input_tfs[0]->try_accumulate_grad(TensorImpl::ones_like(*input_tfs[0]).get());
}

void TransposeBack::step_back() {}

} // namespace INNC
