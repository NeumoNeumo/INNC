#include "INNC/function.hpp"
#include "INNC/dispatcher.hpp"
#include "INNC/ops.hpp"
#include "INNC/tensor.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/utils.hpp"
#include <memory>

namespace INNC {

size_t Backward::global_back_version = 0;

Backward::Backward(
    TensorImpl *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_tfs)
    : input_tfs(input_tfs), this_tf(this_tf), n_outway(0),
      accumulated_n_outway(0), back_version(0) {}

Backward::~Backward() = default;

void tensor_mul_add_f(TensorImpl &dst, TensorImpl &tf1, TensorImpl &tf2) {
  apply_no_grad_binary_op<native::tensor_mul_acc_f_helper>(dst, tf1, tf2);
}

void Backward::try_accumulate_grad(TensorImpl *tf_prev, TensorImpl *tf_w,
                                   TensorImpl *tf_o) {
  if (!tf_prev->requires_grad)
    return;
  try_accumulate_update(tf_prev);
  if (tf_o == nullptr) {
    if (tf_w == nullptr)
      return;
    *tf_prev->grad += *tf_w;
  } else if (is_float(tf_w->type()))
    tensor_mul_add_f(*tf_prev->grad.get(), *tf_w, *tf_o);
  else
    tensor_mul_add_f(*tf_prev->grad.get(), *tf_o, *tf_w);
}

void Backward::try_accumulate_update(TensorImpl *tf_prev, bool zero_init) {
  if (!tf_prev->requires_grad)
    return;
  if (tf_prev->grad_fn.get() != nullptr)
    ++tf_prev->grad_fn->accumulated_n_outway;
  if (tf_prev->grad.get() == nullptr) {
    tf_prev->grad = TensorImpl::create(tf_prev->dtype, tf_prev->view);
    if (zero_init)
      tf_prev->grad->data_->zero_();
  } else if (!tf_prev->grad->data_->is_alloc()) {
    tf_prev->grad->data_->alloc();
    if (zero_init)
      tf_prev->grad->data_->zero_();
  }
}

TensorImpl &Backward::get_out_grad() { return *this_tf->grad; }

void AddBack::step_back() {
  try_accumulate_grad(input_tfs[0].get(), &get_out_grad());
  try_accumulate_grad(input_tfs[1].get(), &get_out_grad());
}

void SubBack::step_back() {
  try_accumulate_grad(input_tfs[0].get(), &get_out_grad());
  try_accumulate_grad(input_tfs[1].get(), &get_out_grad(),
                      TensorImpl::create(-1).get());
}

void MulBack::step_back() {
  try_accumulate_grad(input_tfs[0].get(), input_tfs[1].get(), &get_out_grad());
  try_accumulate_grad(input_tfs[1].get(), input_tfs[0].get(), &get_out_grad());
}

void tensor_div_back_numerator_f(TensorImpl &dst, TensorImpl &tf1,
                                 TensorImpl &tf2) {
  apply_no_grad_binary_op<native::tensor_div_back_numerator_helper>(dst, tf1,
                                                                    tf2);
}
void DivBack::step_back() {
  try_accumulate_update(input_tfs[0].get());
  tensor_div_back_numerator_f(*input_tfs[0]->grad, get_out_grad(),
                              *input_tfs[1]);
  try_accumulate_update(input_tfs[1].get());
  native::tensor_div_back_denominator_helper::dispatch(input_tfs[1]->type(),
                                                       this_tf->type())(
      input_tfs[1]->grad.get(), &get_out_grad(), this_tf, input_tfs[1].get());
}

void SumBack::step_back() {
  try_accumulate_grad(input_tfs[0].get(),
                      TensorImpl::ones_like(*input_tfs[0]).get());
}

void NoBack::step_back() { try_accumulate_grad(input_tfs[0].get(), nullptr); }

void CloneBack::step_back() {
  try_accumulate_grad(input_tfs[0].get(), &get_out_grad());
}

KnownGradBack::KnownGradBack(
    TensorImpl *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_tfs,
    std::shared_ptr<INNC::TensorImpl> grad)
    : Backward(this_tf, input_tfs), grad(grad) {}

void KnownGradBack::step_back() {
  try_accumulate_grad(input_tfs[0].get(), grad.get(), &get_out_grad());
}

SingletonBack::SingletonBack(
    TensorImpl *this_tf,
    const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_tfs,
    const SizeVec &sv)
    : Backward(this_tf, input_tfs), sv(sv) {}

void SingletonBack::step_back() { try_accumulate_update(input_tfs[0].get()); }

} // namespace INNC
