#include "INNC/tensorImpl.hpp"
#include "INNC/dispatcher.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/layouts.hpp"
#include "INNC/ops.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/compile_opt.hpp"
#include "INNC/utils/rand.hpp"
#include "INNC/utils/traits.hpp"
#include "INNC/utils/utils.hpp"
#include <cstring>
#include <queue>

namespace INNC {

size_t TensorImpl::dim() const noexcept { return view->dim(); }

size_t TensorImpl::cnt_from_aug_index(const SizeVec &index) const {
  SizeVec phy_index;
  auto phy_dim = dim();
  auto idx_dim = index.size();
  phy_index.resize(phy_dim);
  for (size_t i = 1; i <= phy_dim; ++i) {
    auto s = view->sizes[phy_dim - i];
    if (__LIKELY(s != 1)) {
      run_expect(index[idx_dim - i] < s, "the index ", index,
                 " is out of range ", view->sizes.to_string());
      phy_index[phy_dim - i] = index[idx_dim - i];
    } else {
      phy_index[phy_dim - i] = 0;
    }
  }
  return TensorImpl::cnt_from_index(phy_index);
}

size_t TensorImpl::cnt_from_index(const SizeVec &index) const {
  if (dlayout == layouts::strided)
    return dynamic_cast<StridedLayout *>(view.get())->cnt_from_index(index);
  else
    throw std::logic_error(
        sformat("This layout %s has not been implemented", layouts::sparse));
}

SizeVec TensorImpl::size() const { return view->sizes; }

SignedVec TensorImpl::stride() const {
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

std::shared_ptr<TensorImpl> TensorImpl::operator+() {
  return shared_from_this();
}

std::shared_ptr<TensorImpl> TensorImpl::operator-() {
  return *create(char(0)) - *this;
}

std::shared_ptr<TensorImpl> operator+(TensorImpl &l, TensorImpl &r) {
  return apply_binary_operator<native::tensor_add_helper, AddBack>(
      l.shared_from_this(), r.shared_from_this());
}

std::shared_ptr<TensorImpl> operator-(TensorImpl &l, TensorImpl &r) {
  return apply_binary_operator<native::tensor_sub_helper, SubBack>(
      l.shared_from_this(), r.shared_from_this(), true);
}

std::shared_ptr<TensorImpl> operator*(TensorImpl &l, TensorImpl &r) {
  return apply_binary_operator<native::tensor_mul_helper, MulBack>(
      l.shared_from_this(), r.shared_from_this());
}

std::shared_ptr<TensorImpl> operator/(TensorImpl &l, TensorImpl &r) {
  return apply_binary_operator<native::tensor_div_helper, DivBack>(
      l.shared_from_this(), r.shared_from_this(), true);
}

std::shared_ptr<TensorImpl> operator<(TensorImpl &l, TensorImpl &r) {
  return apply_cmp_op<native::tensor_lt_helper>(l, r);
}

std::shared_ptr<TensorImpl> operator>(TensorImpl &l, TensorImpl &r) {
  return apply_cmp_op<native::tensor_gt_helper>(l, r);
}

std::shared_ptr<TensorImpl> operator<=(TensorImpl &l, TensorImpl &r) {
  return apply_cmp_op<native::tensor_le_helper>(l, r);
}

std::shared_ptr<TensorImpl> operator>=(TensorImpl &l, TensorImpl &r) {
  return apply_cmp_op<native::tensor_ge_helper>(l, r);
}

std::shared_ptr<TensorImpl> operator==(TensorImpl &l, TensorImpl &r) {
  return apply_cmp_op<native::tensor_eq_helper>(l, r);
}

std::shared_ptr<TensorImpl> operator!=(TensorImpl &l, TensorImpl &r) {
  return apply_cmp_op<native::tensor_ne_helper>(l, r);
}

TensorImpl &TensorImpl::operator+=(const TensorImpl &rhs) {
  run_expect(
      !requires_grad,
      "This inplace operation cannot perform on a tensor that requires grad.");
  native::tensor_add_helper::dispatch(dtype, rhs.dtype)(
      this, this, const_cast<TensorImpl *>(&rhs));
  return *this;
}

std::shared_ptr<TensorImpl> TensorImpl::ones(const SizeVec &sizes,
                                             types dtype) {
  auto one_ = create(i8, SizeVec{});
  *reinterpret_cast<char *>(one_->data_->get_blob()) = 1;
  auto ret = create(dtype, make_shared<StridedLayout>(sizes));
  native::tensor_fill_helper::dispatch(dtype, i8)(ret.get(), one_.get());
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
  run_expect(dim_eq, "Tensors' dimension must match. Trying to add ", lsizes,
             " with ", rsizes, ". Broadcasting has not been supported yet.");
}

size_t TensorImpl::numel() const noexcept { return view->numel(); }

void TensorImpl::release() noexcept { data_->release(); }

INNC::types TensorImpl::type() const { return this->dtype; }

std::shared_ptr<TensorImpl> TensorImpl::type(types t) {
  if (requires_grad && is_int(t))
    throw std::runtime_error(
        "Cannot cast a tensor that requries grad to a integer tensor.");
  if (t == dtype)
    return shared_from_this();
  auto ret = create(t, StridedLayout{view->sizes});
  native::tensor_to_type_helper::dispatch(t, dtype)(ret.get(), this);
  if (!requires_grad)
    return ret;
  ret->requires_grad = true;
  ret->grad_fn.reset(new CloneBack(ret.get(), {shared_from_this()}));
  return ret;
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
  native::tensor_sum_helper::dispatch(dst_t, dtype)(tf.get(), this);
  if (requires_grad) {
    tf->requires_grad = true;
    tf->grad_fn.reset(new SumBack(tf.get(), {shared_from_this()}));
  }
  return tf;
}

std::shared_ptr<TensorImpl> TensorImpl::abs() {
  auto ret = create(dtype, view);
  if (!requires_grad) {
    native::tensor_abs_helper::dispatch(dtype, dtype)(ret.get(), this, nullptr);
  } else {
    auto grad = create(dtype, view);
    native::tensor_abs_helper::dispatch(dtype, dtype)(ret.get(), this,
                                                      grad.get());
    ret->requires_grad = true;
    ret->grad_fn.reset(
        new KnownGradBack(ret.get(), {shared_from_this()}, grad));
  }
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::max() {
  auto ret = zeros(SizeVec{}, dtype);
  if (!requires_grad) {
    native::tensor_max_helper::dispatch(dtype)(ret.get(), this, nullptr);
  } else {
    auto grad = zeros_like(*this);
    native::tensor_max_helper::dispatch(dtype)(ret.get(), this, grad.get());
    ret->requires_grad = true;
    ret->grad_fn.reset(
        new KnownGradBack(ret.get(), {shared_from_this()}, grad));
  }
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::min() {
  auto ret = zeros(SizeVec{}, dtype);
  if (!requires_grad) {
    native::tensor_min_helper::dispatch(dtype)(ret.get(), this, nullptr);
  } else {
    auto grad = zeros_like(*this);
    native::tensor_min_helper::dispatch(dtype)(ret.get(), this, grad.get());
    ret->requires_grad = true;
    ret->grad_fn.reset(
        new KnownGradBack(ret.get(), {shared_from_this()}, grad));
  }
  return ret;
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
    SignedVec _strides;
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
        run_expect(idx >= 0 && idx < static_cast<long long>(sizes[dim]),
                   "index ", idx, " is out of bounds for dimension ", dim,
                   " with size ", sizes[dim]);
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
             "Index out of range dimension ", dim,
             ". Actual input of "
             "transpose: (",
             dim0, ", ", dim1, ")");
  run_expect(
      dim0 != dim1,
      sformat("dim0 and dim1 must be distinguished. But they are both ", dim0));
  auto view_s = dynamic_cast<StridedLayout *>(input->view.get());
  SizeVec _sizes = view_s->sizes;
  SignedVec _strides = view_s->strides;
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

std::shared_ptr<TensorImpl> TensorImpl::transpose(size_t dim0, size_t dim1) {
  return TensorImpl::transpose(shared_from_this(), dim0, dim1);
}
std::shared_ptr<TensorImpl>
TensorImpl::permute(const std::shared_ptr<TensorImpl> &input,
                    const SizeVec dims) {
  run_expect(dims.size() == 2, "the size of dims for permute must have dim 2.");
  return TensorImpl::transpose(input, dims[0], dims[1]);
}
std::shared_ptr<TensorImpl> TensorImpl::permute(const SizeVec dims) {
  return TensorImpl::permute(shared_from_this(), dims);
}

SizeVec regularize_size(const SignedVec &sizes, size_t numel = 0) {
  SignedVec sizes_ = sizes;
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
  size_t numel = 1;
  for (auto i : sizes)
    numel *= i;
  if (input->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("This layout %s has not been implemented", layouts::sparse));

  StridedLayout *view_s = dynamic_cast<StridedLayout *>(input->view.get());
  run_expect(numel == input->numel(),
             "An impossible reshape. The shape of input: (", view_s->sizes,
             "), Actual "
             "input of reshape: (",
             sizes, ")");

  if (numel == 1) {
    SignedVec strides;
    strides.resize(sizes.size(), 1);
    auto tf =
        create(input->dtype,
               std::make_unique<StridedLayout>(sizes, strides, view_s->offset),
               input->data_);
    if (!input->requires_grad)
      return tf;
    tf->requires_grad = true;
    tf->grad_fn.reset(new NoBack(tf.get(), {input}));
    share_grad_storage(*tf, *input);
  }

  bool isometry = true;
  SizeVec sizes_;
  sizes_.resize(view_s->dim());
  SignedVec strides = view_s->strides;
  std::vector<size_t> indexOrder(strides.size());
  for (size_t i = 0; i < indexOrder.size(); ++i)
    indexOrder[i] = i;
  std::sort(indexOrder.begin(), indexOrder.end(),
            [&strides](size_t i, size_t j) { return strides[i] < strides[j]; });
  for (size_t i = 0; i < indexOrder.size(); ++i) {
    sizes_[indexOrder[i]] = view_s->sizes[indexOrder[i]];
    strides[indexOrder[i]] = view_s->strides[indexOrder[i]];
  }

  for (size_t i = 1; i < strides.size(); i++)
    if (strides[i - 1] != (strides[i] * (long long)sizes_[i]))
      isometry = false;
  std::shared_ptr<TensorImpl> tf;
  auto last_node = input;
  if (isometry) {
    auto sn = sizes.size();
    if (sn != 0) {
      strides.resize(sn);
      strides[sn - 1] = view_s->strides[view_s->strides.size() - 1];
      for (size_t idx = sn - 1; idx > 0; --idx) {
        strides[idx - 1] = strides[idx] * sizes[idx];
      }
    }
    tf = create(input->dtype,
                std::make_unique<StridedLayout>(sizes, strides, view_s->offset),
                input->data_);
  } else {
    auto m_tf = input->clone();
    tf = create(input->dtype, make_shared<StridedLayout>(sizes), m_tf->data_);
    last_node = m_tf;
  }
  if (!input->requires_grad)
    return tf;
  tf->requires_grad = true;
  tf->grad_fn.reset(new NoBack(tf.get(), {last_node}));
  share_grad_storage(*tf, *last_node);
  return tf;
}

std::shared_ptr<TensorImpl> TensorImpl::reshape(const SizeVec &sizes) {
  return TensorImpl::reshape(shared_from_this(), sizes);
}

std::shared_ptr<TensorImpl>
TensorImpl::reshape(const std::shared_ptr<TensorImpl> &input,
                    const SignedVec &sizes) {
  return TensorImpl::reshape(input, regularize_size(sizes, input->numel()));
}

std::shared_ptr<TensorImpl> TensorImpl::reshape(const SignedVec &sizes) {
  return TensorImpl::reshape(shared_from_this(),
                             regularize_size(sizes, numel()));
}

std::shared_ptr<TensorImpl> TensorImpl::reshape_as(const TensorImpl &t) {
  return TensorImpl::reshape(shared_from_this(), t.view->sizes);
}

bool TensorImpl::is_contiguous() const noexcept {
  return view->is_contiguous();
}

std::shared_ptr<TensorImpl> TensorImpl::contiguous() {
  return view->contiguous_from(*this);
}

std::shared_ptr<TensorImpl> TensorImpl::clone() {
  auto ret = create(dtype, view->sizes);
  native::tensor_clone_helper::dispatch(dtype, dtype)(ret.get(), this);
  if (!requires_grad)
    return ret;
  ret->requires_grad = true;
  ret->grad_fn.reset(new CloneBack(ret.get(), {shared_from_this()}));
  return ret;
}

std::shared_ptr<TensorImpl> TensorImpl::detach() {
  return create(dtype, view, data_, dlayout);
}

bool TensorImpl::all() const {
  if (dlayout == layouts::strided) {
    for (auto r : view->sizes) {
      if (r == 0)
        return true;
    }
    if (__LIKELY(dim() != 0)) {
      SizeVec sv;
      auto last_idx = dim() - 1;
      sv.resize(last_idx + 1, 0);
      while (true) {
        size_t ptr = last_idx;
        while (sv[ptr] == view->sizes[ptr]) {
          if (ptr == 0)
            return true;
          sv[ptr] = 0;
          ++sv[--ptr];
        }
        if (!*reinterpret_cast<bool *>(
                (data_->get_blob() + cnt_from_index(sv))))
          return false;
        ++sv[last_idx];
      };
    } else {
      return *reinterpret_cast<bool *>(
          (data_->get_blob() + cnt_from_index(SizeVec{})));
    }
  } else {
    throw std::logic_error(
        "Layouts except StridedLayout have not been implemented yet.");
  }
}

std::shared_ptr<TensorImpl> TensorImpl::randn(const SizeVec &sizes,
                                              types dtype) {
  run_expect(INNC::is_float(dtype), "Tensors with integer type ",
             INNC::to_string(dtype),
             " cannot be generated from a normal distribution");
  if (dtype == INNC::f32) {
    std::normal_distribution<float> gen_norm{};
    auto ret = create(INNC::f32, StridedLayout{sizes});
    auto dptr = reinterpret_cast<float *>(ret->data_->get_blob());
    for (size_t i = 0; i < ret->numel(); ++i) {
      *(dptr + i) = gen_norm(rng);
    }
    return ret;
  } else {
    std::normal_distribution<double> gen_norm{};
    auto ret = create(INNC::f32, StridedLayout{sizes});
    auto dptr = reinterpret_cast<double *>(ret->data_->get_blob());
    for (size_t i = 0; i < ret->numel(); ++i) {
      *(dptr + i) = gen_norm(rng);
    }
    return ret;
  }
}

std::shared_ptr<TensorImpl> TensorImpl::randn_like(const TensorImpl &t) {
  return randn(t.size(), t.dtype);
}
} // namespace INNC
