#include "INNC/tensorImpl.hpp"
#include "INNC/dispatcher.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/layouts.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/compile_opt.hpp"
#include "INNC/utils/traits.hpp"
#include "INNC/utils/utils.hpp"
#include <cstring>
#include <queue>

namespace INNC {

size_t TensorImpl::dim() const noexcept { return view->dim(); }

size_t TensorImpl::cnt_from_index(const SizeVec &index) const {
  if (dlayout == layouts::strided)
    return dynamic_cast<StridedLayout *>(view.get())->cnt_from_index(index);
  else
    throw std::logic_error(
        sformat("This layout %s has not been implemented", layouts::sparse));
}

DiffVec TensorImpl::stride() const {
  if (typeid(view) == typeid(std::shared_ptr<StridedLayout>)) {
    return dynamic_cast<StridedLayout *>(view.get())->strides;
  } else {
    throw std::runtime_error("Scalar do not have stride");
  }
}

std::string TensorImpl::to_string() const {
  return view->to_string_from(*data_, dtype);
}

TensorImpl::~TensorImpl() = default;

TensorImpl::TensorImpl(types dtype, layouts dlayout,
                       const std::shared_ptr<Layout> &view,
                       const std::shared_ptr<UntypedStorage> &data_, Private)
    : data_(data_), view(view), dtype(dtype), dlayout(dlayout) {
  requires_grad = false;
  retain_grad = false;
  _version = 0;
}

std::shared_ptr<TensorImpl>
TensorImpl::create(types dtype, const std::shared_ptr<Layout> &view,
                   const std::shared_ptr<UntypedStorage> &data_,
                   layouts dlayout) {
  return std::make_shared<TensorImpl>(dtype, dlayout, view, data_, Private{});
}

std::shared_ptr<TensorImpl>
TensorImpl::create(types dtype, const std::shared_ptr<Layout> &view,
                   bool prealloc, layouts dlayout) {
  auto ptr = create(dtype, view,
                    std::make_unique<UntypedStorage>(
                        size_of(dtype) * view->numel(), prealloc),
                    dlayout);
  if (prealloc)
    ptr->data_->alloc();
  return ptr;
}

std::shared_ptr<TensorImpl> TensorImpl::create(types dtype,
                                               StridedLayout &&view,
                                               bool prealloc, layouts dlayout) {
  return create(dtype, std::make_shared<StridedLayout>(std::move(view)),
                prealloc, dlayout);
}

std::shared_ptr<TensorImpl> TensorImpl::create(std::int8_t a) {
  auto ret = create(types::i8, SizeVec{});
  *reinterpret_cast<std::int8_t *>(ret->data_->get_blob()) = a;
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::create(std::int16_t a) {
  auto ret = create(types::i16, SizeVec{});
  *reinterpret_cast<std::int16_t *>(ret->data_->get_blob()) = a;
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::create(std::int32_t a) {
  auto ret = create(types::i32, SizeVec{});
  *reinterpret_cast<std::int32_t *>(ret->data_->get_blob()) = a;
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::create(std::int64_t a) {
  auto ret = create(types::i64, SizeVec{});
  *reinterpret_cast<std::int64_t *>(ret->data_->get_blob()) = a;
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::create(float a) {
  auto ret = create(types::f32, SizeVec{});
  *reinterpret_cast<float *>(ret->data_->get_blob()) = a;
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::create(double a) {
  auto ret = create(types::f64, SizeVec{});
  *reinterpret_cast<double *>(ret->data_->get_blob()) = a;
  return ret;
}

template <typename L, typename R>
void tensor_add(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<L *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) =
        *(l_ptr + l->cnt_from_index(sv)) + *(r_ptr + r->cnt_from_index(sv));
  });
}

template <typename L, typename R>
void tensor_mul(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<L *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) =
        *(l_ptr + l->cnt_from_index(sv)) * *(r_ptr + r->cnt_from_index(sv));
  });
}

template <typename TensorType, typename NumberType>
void tensor_fill(TensorImpl *tdata, const TensorImpl *ndata) {
  TensorType num = *reinterpret_cast<NumberType *>(ndata->data_->get_blob());
  auto t_ptr = reinterpret_cast<TensorType *>(tdata->data_->get_blob());
  if (__LIKELY(tdata->dlayout == layouts::strided)) {
    if (__LIKELY(tdata->view->dim() != 0)) {
      for_each_sizevec(tdata->view->sizes, [=](const SizeVec &sv) {
        *(t_ptr + tdata->cnt_from_index(sv)) = num;
      });
    } else {
      *(t_ptr + dynamic_cast<StridedLayout *>(tdata->view.get())->offset) = num;
    }
  } else {
    throw std::logic_error("Not implemented yet");
  }
}

template <typename ToType, typename FromType>
void tensor_to_type(TensorImpl *to, const TensorImpl *from) {
  ToType *to_ptr = reinterpret_cast<ToType *>(to->data_->get_blob());
  FromType *from_ptr = reinterpret_cast<FromType *>(from->data_->get_blob());
  for_each_sizevec(to->view->sizes, [=](const SizeVec &sv) {
    *(to_ptr + to->cnt_from_index(sv)) = *(from_ptr + from->cnt_from_index(sv));
  });
}

template <typename ToType, typename FromType>
void tensor_sum(TensorImpl *to, const TensorImpl *from) {
  if (from->view->numel() == 0)
    return;
  SizeVec sv;
  auto &data_sizes = from->view->sizes;
  auto last_idx = data_sizes.size() - 1;
  ToType *to_ptr = reinterpret_cast<ToType *>(to->data_->get_blob());
  FromType *from_ptr = reinterpret_cast<FromType *>(from->data_->get_blob());
  if (__LIKELY(from->view->dim() != 0)) {
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
      *to_ptr += *(from_ptr + from->cnt_from_index(sv));
      ++sv[last_idx];
    }
  } else {
    if (from->dlayout == layouts::strided) {
      *to_ptr =
          *(from_ptr + dynamic_cast<StridedLayout *>(from->view.get())->offset);
    } else {
      throw std::logic_error("Not implemented yet");
    }
  }
}

template <typename D, typename L, typename R>
void tensor_mul_acc_f(TensorImpl *dst, const TensorImpl *l,
                      const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<D *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) +=
        *(l_ptr + l->cnt_from_index(sv)) * *(r_ptr + r->cnt_from_index(sv));
  });
}

template <typename ToType, typename FromType>
void tensor_clone(TensorImpl *to, const TensorImpl *from) {
  if (to->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of output has not been implemented",
                to_string(to->dlayout)));
  if (from->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of input has not been implemented",
                to_string(from->dlayout)));
  ToType *to_ptr = reinterpret_cast<ToType *>(to->data_->get_blob());
  FromType *from_ptr = reinterpret_cast<FromType *>(from->data_->get_blob());
  for_each_sizevec(to->view->sizes, [=](const SizeVec &sv) {
    *(to_ptr + to->cnt_from_index(sv)) = *(from_ptr + from->cnt_from_index(sv));
  });
}

generate_binary_op_helper(tensor_add);
generate_binary_op_helper(tensor_mul);
generate_unary_op_helper(tensor_fill);
generate_unary_op_helper(tensor_to_type);
generate_unary_op_helper(tensor_sum);
generate_unary_op_helper(tensor_clone);
generate_ffi_op_helper(tensor_mul_acc_f);

std::shared_ptr<TensorImpl> operator+(TensorImpl &l, TensorImpl &r) {
  return apply_binary_operator<tensor_add_helper, AddBack>(
      l.shared_from_this(), r.shared_from_this());
}

std::shared_ptr<TensorImpl> operator*(TensorImpl &l, TensorImpl &r) {
  return apply_binary_operator<tensor_mul_helper, MulBack>(
      l.shared_from_this(), r.shared_from_this());
}

TensorImpl &TensorImpl::operator+=(const TensorImpl &rhs) {
  run_expect(
      !requires_grad,
      "This inplace operation cannot perform on a tensor that requires grad.");
  tensor_add_helper::dispatch(dtype, rhs.dtype)(this, this,
                                                const_cast<TensorImpl *>(&rhs));
  return *this;
}

std::shared_ptr<TensorImpl> TensorImpl::ones(const SizeVec &sizes,
                                             types dtype) {
  auto one_ = create(i8, SizeVec{});
  *reinterpret_cast<char *>(one_->data_->get_blob()) = 1;
  auto ret = create(dtype, make_shared<StridedLayout>(sizes));
  tensor_fill_helper::dispatch(dtype, i8)(ret.get(), one_.get());
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::zeros(const SizeVec &sizes,
                                              types dtype) {
  auto ret = create(dtype, StridedLayout{sizes});
  ret->data_->zero_();
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::zeros_like(const TensorImpl &t) {
  return TensorImpl::zeros(t.view->sizes, t.dtype);
}

std::shared_ptr<TensorImpl> TensorImpl::ones_like(const TensorImpl &t) {
  return TensorImpl::ones(t.view->sizes, t.dtype);
}

std::shared_ptr<TensorImpl>
TensorImpl::from_blob(void *data, const SizeVec &sizes, types dtype) {
  auto ret = create(dtype, StridedLayout{sizes});
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

std::shared_ptr<TensorImpl> TensorImpl::type(types t) {
  auto tf = create(t, StridedLayout{view->sizes});
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
    grad = create(dtype, view);
    grad->data_->zero_();
  } else if (!grad->data_->is_alloc()) {
    grad->data_->alloc();
    grad->data_->zero_();
  }
  if (tf_o == nullptr) {
    if (tf_w == nullptr)
      return;
    *grad += *tf_w;
  } else if (is_float(tf_w->type()))
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
      if (it->grad_fn.get() == nullptr)
        continue;
      ++it->grad_fn->n_outway;
      if (__LIKELY(it->grad_fn->back_version < Backward::global_back_version)) {
        it->grad_fn->back_version = Backward::global_back_version;
        q.push(it.get());
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

std::shared_ptr<TensorImpl> TensorImpl::sum() {
  INNC::types dst_t;
  if (dtype <= i64)
    dst_t = i64;
  else
    dst_t = f64;
  auto tf = zeros(SizeVec{}, dst_t);
  tensor_sum_helper::dispatch(dst_t, dtype)(tf.get(), this);
  if (requires_grad) {
    tf->requires_grad = true;
    tf->grad_fn.reset(new SumBack(tf.get(), {shared_from_this()}));
  }
  return tf;
}

void TensorImpl::zero_grad() const noexcept {
  if (grad.get() == nullptr)
    return;
  grad->data_->zero_();
}

void share_grad_storage(TensorImpl &to, TensorImpl &from,
                        const std::shared_ptr<Layout> &view) {
  if (from.grad == nullptr) {
    from.grad = TensorImpl::create(from.dtype, from.view, false);
  }
  to.grad = TensorImpl::create(to.dtype, view, from.grad->data_);
}

void share_grad_storage(TensorImpl &to, TensorImpl &from) {
  share_grad_storage(to, from, to.view);
}

std::shared_ptr<TensorImpl> TensorImpl::operator[](const std::string &slice) {
  if (dlayout == layouts::strided) {
    std::vector<std::string> each_dim = ssplit(slice, ',');
    SizeVec _sizes;
    DiffVec _strides;
    auto view_s = dynamic_cast<StridedLayout *>(view.get());
    auto &sizes = view_s->sizes;
    auto &strides = view_s->strides;
    auto offset = view_s->offset;
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
    auto ret = create(
        dtype, std::make_unique<StridedLayout>(_sizes, _strides, _offset),
        data_);
    if (!requires_grad)
      return ret;
    ret->requires_grad = true;
    ret->grad_fn.reset(new NoBack(ret.get(), {shared_from_this()}));
    share_grad_storage(*ret, *this);
    return ret;
  } else {
    throw std::runtime_error("Not implemented");
  }
}

std::shared_ptr<TensorImpl>
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
  auto view_s = dynamic_cast<StridedLayout *>(input->view.get());
  SizeVec _sizes = view_s->sizes;
  DiffVec _strides = view_s->strides;
  std::swap(_sizes[dim0], _sizes[dim1]);
  std::swap(_strides[dim0], _strides[dim1]);
  auto ret =
      create(input->dtype,
             std::make_unique<StridedLayout>(_sizes, _strides, view_s->offset),
             input->data_);
  if (!input->requires_grad)
    return ret;
  ret->requires_grad = true;
  ret->grad_fn.reset(new NoBack(ret.get(), {input}));
  share_grad_storage(*ret, *input);
  return ret;
}

bool TensorImpl::is_contiguous() const noexcept {
  return view->is_contiguous();
}

std::shared_ptr<TensorImpl> TensorImpl::contiguous() {
  return view->contiguous_from(*this);
}

SizeVec DiffVec_to_SizeVec(const DiffVec &sizes, size_t numel = 0) {
  DiffVec sizes_ = sizes;
  if (sizes_.size() == 0) {
    run_expect(
        numel == 1,
        "You cannot turning a empty DiffVec to SizeVec with (numel != 1).");
    return {};
  }
  size_t minus_one_exist = false;
  size_t minus_one_index;
  for (size_t i = 0; i < sizes_.size(); i++) {
    if (sizes_[i] == -1) {
      run_expect(!minus_one_exist,
                 "Cannot have more than two minus ones in size.");
      minus_one_exist = true;
      minus_one_index = i;
    }
  }
  if (minus_one_exist) {
    size_t s = 1;
    for (size_t i = 0; i < sizes_.size(); i++)
      if (i != minus_one_index)
        s *= sizes_[i];
    run_expect(numel % s == 0, "The input of sizes and numel is illegal.");
    sizes_[minus_one_index] = numel / s;
  }
  SizeVec sizes__;
  for (size_t i = 0; i < sizes_.size(); i++) {
    sizes__.push_back(sizes_[i]);
  }
  return sizes__;
}

std::shared_ptr<TensorImpl>
TensorImpl::reshape(const std::shared_ptr<TensorImpl> &input,
                    const SizeVec &sizes) {
  size_t dim_sizes = 1;
  for (size_t i = 0; i < sizes.size(); i++)
    dim_sizes *= sizes[i];
  if (input->dlayout != layouts::strided) {
    throw std::logic_error(
        sformat("This layout %s has not been implemented", layouts::sparse));
  }
  StridedLayout *view_s = dynamic_cast<StridedLayout *>(input->view.get());
  run_expect(dim_sizes == input->numel(),
             sformat("An impossible reshape. The shape of input: (%s), Actual "
                     "input of reshape: (%s)",
                     view_s->sizes.to_string(), sizes.to_string()));

  
  bool data_continuous = true;
  SizeVec sizes_;
  sizes_.resize(view_s->sizes.size());
  DiffVec strides = view_s->strides;
  std::vector<size_t> indexOrder(strides.size());
    for (size_t i = 0; i < indexOrder.size(); ++i) {
        indexOrder[i] = i;
    }
  std::sort(indexOrder.begin(), indexOrder.end(), [&strides](size_t i, size_t j) { return strides[i] < strides[j]; });
  for (size_t i = 0; i < indexOrder.size(); ++i) {
        sizes_[indexOrder[i]] = view_s->sizes[indexOrder[i]];
        strides[indexOrder[i]] = view_s->strides[indexOrder[i]];
  }
  
  for (size_t i = 1; i < strides.size(); i++)
    if (strides[i - 1] != (strides[i] * (long long)sizes_[i]))
      data_continuous = false;
  if (data_continuous) {
    auto sn = sizes.size();
    if (sn != 0) {
      strides.resize(sn);
      strides[sn - 1] = view_s->strides[view_s->strides.size() - 1];
      for (size_t idx = sn - 1; idx > 0; --idx) {
        strides[idx - 1] = strides[idx] * sizes[idx];
      }
    }
    auto tf =
        create(input->dtype,
               std::make_unique<StridedLayout>(sizes, strides, view_s->offset),
               input->data_);
    if (!input->requires_grad)
      return tf;
    tf->requires_grad = true;
    tf->grad_fn.reset(new ReshapeBack(tf.get(), {input}));
    share_grad_storage(*tf, *input);
    return tf;
  } else {
    auto m_tf = input->clone();
    auto tf =
        create(input->dtype, make_shared<StridedLayout>(sizes), m_tf->data_);
    if (!input->requires_grad)
      return tf;
    tf->requires_grad = true;
    tf->grad_fn.reset(new ReshapeBack(tf.get(), {m_tf}));
    share_grad_storage(*tf, *m_tf);
    return tf;
  }
}

std::shared_ptr<TensorImpl> TensorImpl::reshape(const SizeVec &sizes) {
  return TensorImpl::reshape(shared_from_this(), sizes);
}

std::shared_ptr<TensorImpl>
TensorImpl::reshape(const std::shared_ptr<TensorImpl> &input,
                    const DiffVec &sizes) {
  return TensorImpl::reshape(input, DiffVec_to_SizeVec(sizes, input->numel()));
}

std::shared_ptr<TensorImpl> TensorImpl::reshape(const DiffVec &sizes) {
  return TensorImpl::reshape(shared_from_this(),
                             DiffVec_to_SizeVec(sizes, numel()));
}

std::shared_ptr<TensorImpl> TensorImpl::clone() {
  auto ret = create(dtype, view->sizes);
  tensor_clone_helper::dispatch(dtype, dtype)(ret.get(), this);
  if (!requires_grad)
    return ret;
  ret->requires_grad = true;
  ret->grad_fn.reset(new CloneBack(ret.get(), {shared_from_this()}));
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::detach() {
  return create(dtype, view, data_, dlayout);
}

} // namespace INNC
