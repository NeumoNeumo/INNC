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
  input_tfs[0]->try_accumulate_grad(*this_tf->grad.get());
  input_tfs[1]->try_accumulate_grad(*this_tf->grad.get());
}

MulBack::MulBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void MulBack::step_back() {
  input_tfs[0]->try_accumulate_grad(*input_tfs[1].get());
  input_tfs[1]->try_accumulate_grad(*input_tfs[0].get());
}

SumBack::SumBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void SumBack::step_back() {
  input_tfs[0]->try_accumulate_grad(
      *TensorFrame::ones_like(*input_tfs[0].get()).get());
}

ReshapeBack::ReshapeBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void ReshapeBack::step_back() {
  input_tfs[0]->try_accumulate_grad(*TensorFrame::reshape_without_grad(
      *this_tf->grad.get(), input_tfs[0]->sizes));
}

TransposeBack::TransposeBack(
    TensorFrame *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorFrame>> &input_tfs)
    : Backward(this_tf, input_tfs){};

void TransposeBack::step_back() {
  SizeVec dim01;
  for (size_t i = 0; i < this_tf->sizes.size(); i++) {
    if (input_tfs[0].get()->sizes[i] != this_tf->sizes[i])
      dim01.push_back(i);
  }
  input_tfs[0]->try_accumulate_grad(
      *TensorFrame::transpose_without_grad(this_tf->grad, dim01[0], dim01[1]));
}

} // namespace INNC
