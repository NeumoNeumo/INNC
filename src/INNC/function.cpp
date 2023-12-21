#include "INNC/function.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/tensor.hpp"
#include "INNC/types.hpp"
#include <memory>

namespace INNC {

Backward::Backward(TensorFrame *this_tf,
                   const std::vector<INNC::TensorFrame *> &input_tfs)
    : input_tfs(input_tfs), this_tf(this_tf), n_outway(1),
      accumulated_n_outway(0), visited(false) {}

Backward::~Backward() = default;

TensorFrame &Backward::get_out_grad() { return *this_tf->grad.get(); }

AddBack::AddBack(TensorFrame *this_tf,
                 const std::vector<INNC::TensorFrame *> &input_tfs)
    : Backward(this_tf, input_tfs){};

void AddBack::step_back() {
  input_tfs[0]->try_accumulate_grad(*TensorFrame::ones_like(*input_tfs[0]));
  input_tfs[1]->try_accumulate_grad(*TensorFrame::ones_like(*input_tfs[1]));
}

} // namespace INNC
