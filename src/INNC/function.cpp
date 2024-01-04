#include "INNC/function.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/tensor.hpp"
#include "INNC/types.hpp"
#include <memory>

namespace INNC {

size_t Backward::global_back_version = 0;

Backward::Backward(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : input_tfs(input_tfs), this_tf(this_tf), n_outway(0),
      accumulated_n_outway(0), back_version(0) {}

Backward::~Backward() = default;

TensorFrame &Backward::get_out_grad() { return *this_tf->grad.get(); }

AddBack::AddBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void AddBack::step_back() {
  input_tfs[0]->try_accumulate_grad(&get_out_grad());
  input_tfs[1]->try_accumulate_grad(&get_out_grad());
}

MulBack::MulBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void MulBack::step_back() {
  input_tfs[0]->try_accumulate_grad(input_tfs[1].get(), &get_out_grad());
  input_tfs[1]->try_accumulate_grad(input_tfs[0].get(), &get_out_grad());
}

SumBack::SumBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void SumBack::step_back() {
  input_tfs[0]->try_accumulate_grad(
      TensorFrame::ones_like(*input_tfs[0].get()).get());
}

} // namespace INNC
