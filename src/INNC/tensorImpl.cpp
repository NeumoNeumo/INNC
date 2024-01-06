#include "INNC/tensorImpl.hpp"
#include "INNC/dispatcher.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/compile_opt.hpp"
#include "INNC/utils/traits.hpp"
#include "INNC/utils/utils.hpp"
#include "INNC/view.hpp"
#include <cstring>
#include <queue>

namespace INNC {
size_t TensorImpl::dim() const noexcept { return view->dim(); }

size_t TensorImpl::cnt_from_index(const SizeVec &index) const noexcept {
  assertm(index.size() == dim(), "index mismatches sizes");
  size_t pos = view->offset;
  for (size_t i = 0; i < index.size(); ++i) {
    pos += view->strides[i] * index[i];
  }
  return pos;
}

std::string to_string_helper(const TensorImpl *tf, std::ptrdiff_t offset = 0,
                             size_t depth = 0) {
  if (depth == tf->dim()) {
    void *ptr = tf->data_->get_blob() + offset;
    return INNC::innc_type_to_string(ptr, tf->dtype);
  }
  bool begin = true;
  std::string ret = "[";
  for (size_t i = 0; i < tf->view->sizes[depth]; ++i) {
    std::ptrdiff_t sub_offset =
        offset + i * tf->view->strides[depth] * INNC::size_of(tf->dtype);
    if (begin) {
      ret += to_string_helper(tf, sub_offset, depth + 1);
      begin = false;
      continue;
    }
    ret += ", " + to_string_helper(tf, sub_offset, depth + 1);
  }
  return ret + "]";
}

std::string TensorImpl::to_string() const {
  if (__UNLIKELY(dim() == 0))
    return INNC::innc_type_to_string(
        data_->get_blob() + view->offset * INNC::size_of(dtype), dtype);
  return to_string_helper(this, view->offset * INNC::size_of(dtype));
}

TensorImpl::~TensorImpl() = default;

TensorImpl::TensorImpl(TensorImpl &&t) = default;

TensorImpl::TensorImpl(types dtype, const std::shared_ptr<StridedView> &view,
                       const std::shared_ptr<UntypedStorage> &data_)
    : data_(data_), view(view), dtype(dtype) {
  requires_grad = false;
  retain_grad = false;
  _version = 0;
}

TensorImpl::TensorImpl(types dtype, const std::shared_ptr<StridedView> &view,
                       bool prealloc)
    : TensorImpl(dtype, view,
                 std::make_unique<UntypedStorage>(
                     size_of(dtype) * view->numel(), prealloc)) {
  if (prealloc)
    data_->create();
}

TensorImpl::TensorImpl(types dtype, StridedView &&view, bool prealloc)
    : TensorImpl{dtype, std::make_unique<StridedView>(std::move(view)),
                 prealloc} {}

template <typename TensorType, typename NumberType>
void tensor_fill(TensorImpl *tdata, TensorImpl *ndata) {
  TensorType num = *reinterpret_cast<NumberType *>(ndata->data_->get_blob());
  auto t_ptr = reinterpret_cast<TensorType *>(tdata->data_->get_blob());
  for_each_sizevec(tdata->view->sizes, [=](const SizeVec &sv) {
    *(t_ptr + tdata->cnt_from_index(sv)) = num;
  });
}

template <typename ToType, typename FromType>
void tensor_to_type(TensorImpl *todata, TensorImpl *fromdata) {
  ToType *to_ptr = reinterpret_cast<ToType *>(todata->data_->get_blob());
  FromType *from_ptr =
      reinterpret_cast<FromType *>(fromdata->data_->get_blob());
  for_each_sizevec(todata->view->sizes, [=](const SizeVec &sv) {
    *(to_ptr + todata->cnt_from_index(sv)) =
        *(from_ptr + fromdata->cnt_from_index(sv));
  });
}

template <typename ToType, typename FromType>
void tensor_sum(TensorImpl *todata, TensorImpl *fromdata) {
  SizeVec sv;
  auto &data_sizes = fromdata->view->sizes;
  auto last_idx = data_sizes.size() - 1;
  ToType *to_ptr = reinterpret_cast<ToType *>(todata->data_->get_blob());
  FromType *from_ptr =
      reinterpret_cast<FromType *>(fromdata->data_->get_blob());
  sv.resize(last_idx + 1);
  for (auto &it : sv)
    it = 0;
  while (true) {
    size_t ptr = last_idx;
    while (sv[ptr] == data_sizes[ptr]) {
      if (ptr == 0)
        return;
      sv[ptr] = 0;
      ++sv[--ptr];
    }
    *to_ptr += *(from_ptr + fromdata->cnt_from_index(sv));
    ++sv[last_idx];
  }
}

template <typename D, typename L, typename R>
void tensor_mul_acc_f(TensorImpl *dst, TensorImpl *l, TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<D *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) +=
        *(l_ptr + l->cnt_from_index(sv)) * *(r_ptr + r->cnt_from_index(sv));
  });
}

generate_unary_op_helper(tensor_fill);
generate_unary_op_helper(tensor_to_type);
generate_unary_op_helper(tensor_sum);
generate_ffi_op_helper(tensor_mul_acc_f);

std::unique_ptr<TensorImpl> TensorImpl::ones(const SizeVec &sizes,
                                             types dtype) {
  TensorImpl one_{i8, SizeVec{}};
  *reinterpret_cast<char *>(one_.data_->get_blob()) = 1;
  auto ret = std::make_unique<TensorImpl>(dtype, StridedView{sizes});
  tensor_fill_helper::dispatch(dtype, i8)(ret.get(), &one_);
  return ret;
}

std::unique_ptr<TensorImpl> TensorImpl::zeros(const SizeVec &sizes,
                                              types dtype) {
  auto ret = std::make_unique<TensorImpl>(dtype, StridedView{sizes});
  ret->data_->zero_();
  return ret;
}

std::unique_ptr<TensorImpl> TensorImpl::zeros_like(const TensorImpl &t) {
  return TensorImpl::zeros(t.view->sizes, t.dtype);
}

std::unique_ptr<TensorImpl> TensorImpl::ones_like(const TensorImpl &t) {
  return TensorImpl::ones(t.view->sizes, t.dtype);
}

std::unique_ptr<TensorImpl>
TensorImpl::from_blob(void *data, const SizeVec &sizes, types dtype) {
  auto ret = std::make_unique<TensorImpl>(dtype, StridedView{sizes});
  std::memcpy(ret->data_->get_blob(), data,
              ret->numel() * INNC::size_of(dtype));
  return ret;
}

void check_same_size(const TensorImpl &lhs, const TensorImpl &rhs) {
  bool dim_eq = true;
  auto &lsizes = lhs.view->sizes;
  auto &rsizes = rhs.view->sizes;
  if (__LIKELY(lhs.dim() == rhs.dim())) {
    auto dim = lhs.dim();
    for (size_t i = 0; i < dim; ++i)
      if (lsizes[i] != rsizes[i]) {
        dim_eq = false;
        break;
      }
  } else
    dim_eq = false;
  if (!dim_eq)
    std::cerr << "Tensors' dimension must match. Trying to add " +
                     lsizes.to_string() + " with " + rsizes.to_string()
              << std::endl
              << "Broadcasting has not been supported yet.";
}

size_t TensorImpl::numel() const noexcept { return view->numel(); }

void TensorImpl::release() noexcept { data_->release(); }

inline INNC::types TensorImpl::type() const { return this->dtype; }

std::unique_ptr<TensorImpl> TensorImpl::type(types t) {
  auto tf = std::make_unique<TensorImpl>(t, StridedView{view->sizes});
  tensor_to_type_helper::dispatch(t, dtype)(tf.get(), this);
  return tf;
}

void tensor_mul_add_f(TensorImpl &dst, TensorImpl &tf1, TensorImpl &tf2) {
  apply_no_grad_binary_op<tensor_mul_acc_f_helper>(dst, tf1, tf2);
}

void TensorImpl::try_accumulate_grad(TensorImpl *tf_w, TensorImpl *tf_o) {
  if (!requires_grad)
    return;
  if (grad_fn.get() != nullptr)
    ++grad_fn->accumulated_n_outway;
  if (grad.get() == nullptr) {
    grad = std::make_unique<TensorImpl>(dtype, view);
    grad->data_->zero_();
  } else if (!grad->data_->is_alloc()) {
    grad->data_->create();
    grad->data_->zero_();
  }
  if (tf_o == nullptr)
    *grad += *tf_w;
  else if (is_float(tf_w->type()))
    tensor_mul_add_f(*grad.get(), *tf_w, *tf_o);
  else
    tensor_mul_add_f(*grad.get(), *tf_o, *tf_w);
}

void refresh_n_outway(TensorImpl *tf) {
  std::queue<TensorImpl *> q;
  tf->grad_fn->n_outway = 0;
  tf->grad_fn->back_version = ++Backward::global_back_version;
  q.push(tf);
  while (!q.empty()) {
    auto t = q.front();
    q.pop();
    for (const auto &it : t->grad_fn->input_tfs) {
      if (__LIKELY(it->grad_fn.get() != nullptr &&
                   it->grad_fn->back_version < Backward::global_back_version)) {
        it->grad_fn->back_version = Backward::global_back_version;
        it->grad_fn->n_outway = 0;
        q.push(it.get());
      }
    }
  }
  tf->grad_fn->back_version = ++Backward::global_back_version;
  q.push(tf);
  while (!q.empty()) {
    auto t = q.front();
    q.pop();
    for (const auto &it : t->grad_fn->input_tfs) {
      if (it->grad_fn.get() != nullptr) {
        ++it->grad_fn->n_outway;
        if (__LIKELY(it->grad_fn->back_version <
                     Backward::global_back_version)) {
          it->grad_fn->back_version = Backward::global_back_version;
          q.push(it.get());
        }
      }
    }
  }
}

void TensorImpl::backward() {
  run_expect(requires_grad,
             "Cannot backward from a tensor that does not require grad.");
  run_expect(grad_fn.get() != nullptr,
             "Cannot backward from a tensor that has no grad_func");
  run_expect(numel() == 1,
             "Only scalar number could be the start of a backward propagation");
  refresh_n_outway(this);
  std::queue<TensorImpl *> q; // TODO 3 multiprocessing
  q.push(this);
  grad = ones_like(*this);
  grad_fn->back_version = ++Backward::global_back_version;
  while (!q.empty()) {
    auto t = q.front();
    q.pop();
    t->grad_fn->step_back();
    if (!t->retain_grad)
      t->grad.reset();
    for (const auto &it : t->grad_fn->input_tfs) {
      if (it->grad_fn.get() != nullptr &&
          it->grad_fn->back_version < Backward::global_back_version &&
          it->grad_fn->is_ready()) {
        it->grad_fn->back_version = Backward::global_back_version;
        q.push(it.get());
      }
    }
  }
}

std::unique_ptr<TensorImpl> TensorImpl::sum() const {
  INNC::types dst_t;
  if (dtype <= i64)
    dst_t = i64;
  else
    dst_t = f64;
  auto tf = zeros(SizeVec{1}, dst_t);
  tensor_sum_helper::dispatch(dst_t, dtype)(tf.get(),
                                            const_cast<TensorImpl *>(this));
  return tf;
}

void TensorImpl::zero_grad() const noexcept {
  if (grad.get() == nullptr)
    return;
  grad->data_->zero_();
}

std::unique_ptr<TensorImpl> TensorImpl::operator[](const std::string &slice) {
  std::vector<std::string> each_dim = ssplit(slice, ',');
  SizeVec _sizes;
  DiffVec _strides;
  auto &sizes = view->sizes;
  auto &strides = view->strides;
  auto offset = view->offset;
  size_t _offset = offset;
  run_expect(each_dim.size() <= sizes.size(),
             "Dimension of slicing is larger than sizes.");
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    std::vector<std::string> split_slice;
    if (dim < each_dim.size()) {
      split_slice = ssplit(each_dim[dim], ':');
      for (auto &s : split_slice)
        trim_(s);
    } else
      split_slice = {"", "", ""};
    if (split_slice.size() == 1) { // like a[-2]
      long long idx = std::stoll(split_slice[0]);
      if (idx < 0)
        idx += sizes[dim];
      run_expect(
          idx >= 0 && idx < static_cast<long long>(sizes[dim]),
          sformat("index %d is out of bounds for dimension %lu with size %lu",
                  idx, dim, sizes[dim]));
      _offset += strides[dim] * idx;
    } else { // like a[-2:]
      if (sizes[dim] == 0) {
        _sizes.push_back(0);
        _strides.push_back(0);
        continue;
      }
      for (int i = split_slice.size(); i < 3; ++i)
        split_slice.push_back("");
      long long step, beg, end;
      if (split_slice[2].size() != 0) {
        step = stoll(split_slice[2]);
        run_expect(step != 0, "slice step cannot be zero");
      } else
        step = 1;
      if (split_slice[0].size() != 0) {
        beg = stoll(split_slice[0]);
        if (beg < 0)
          beg += sizes[dim];
        if (beg < 0) {
          if (step > 0)
            beg = 0;
          else {
            _sizes.push_back(0);
            _strides.push_back(0);
            continue;
          }
        }
        if (beg >= static_cast<long long>(sizes[dim])) {
          if (step > 0) {
            _sizes.push_back(0);
            _strides.push_back(0);
            continue;
          } else
            beg = sizes[dim] - 1;
        }
      } else
        beg = step > 0 ? 0 : sizes[dim] - 1;
      if (split_slice[1].size() != 0) {
        end = stoll(split_slice[1]);
        if (end < 0)
          end += sizes[dim];
        if (step > 0) {
          if (end <= 0) {
            _sizes.push_back(0);
            _strides.push_back(0);
            continue;
          }
          if (end > static_cast<long long>(sizes[dim]))
            end = sizes[dim];
        } else {
          if (end < 0)
            end = -1;
          if (end >= static_cast<long long>(sizes[dim]) - 1) {
            _sizes.push_back(0);
            _strides.push_back(0);
            continue;
          }
        }
      } else
        end = step > 0 ? sizes[dim] : -1;
      if ((step > 0 && beg >= end) || (step < 0 && beg <= end)) {
        _sizes.push_back(0);
        _strides.push_back(0);
        continue;
      }
      _sizes.push_back((std::abs(end - beg) - 1) / std::abs(step) + 1);
      _strides.push_back(step * strides[dim]);
      _offset += beg * strides[dim];
    } // like a[-2:]
  }
  auto ret = std::make_unique<TensorImpl>(
      dtype, std::make_unique<StridedView>(_sizes, _strides, _offset), data_);
  // share_grad_storage(*ret, *this); // TODO
  return ret;
}

void share_grad_storage(TensorImpl &to, TensorImpl &from) {
  if (from.grad == nullptr) {
    from.grad.reset(new TensorImpl(from.dtype, from.view, false));
  }
  to.grad.reset(new TensorImpl(to.dtype, to.view, from.grad->data_));
}

std::unique_ptr<TensorImpl>
TensorImpl::transpose(const std::shared_ptr<TensorImpl> &input, size_t dim0,
                      size_t dim1) {
  auto dim = input->dim();
  run_expect(dim0 >= 0 && dim1 >= 0 && dim0 < dim && dim1 < dim,
             sformat("Index out of range dimension %lu. Actual input of "
                     "transpose: (%lu, %lu)",
                     dim, dim0, dim1));
  run_expect(
      dim0 != dim1,
      sformat("dim0 and dim1 must be distinguished. But they are both %lu.",
              dim0));
  SizeVec _sizes = input->view->sizes;
  DiffVec _strides = input->view->strides;
  std::swap(_sizes[dim0], _sizes[dim1]);
  std::swap(_strides[dim0], _strides[dim1]);
  auto tf = std::make_unique<TensorImpl>(
      input->dtype,
      std::make_unique<StridedView>(_sizes, _strides, input->view->offset),
      input->data_);
  if (!input->requires_grad)
    return tf;
  tf->requires_grad = true;
  tf->grad_fn.reset(new TransposeBack(tf.get(), {input}));
  share_grad_storage(*tf, *input);
  return tf;
}
} // namespace INNC
