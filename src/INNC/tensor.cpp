#include "INNC/tensor.hpp"
#include "INNC/exceptions.hpp"
#include "INNC/function.hpp"
#include "INNC/storage.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/utils.hpp"
#include <cstring>
#include <queue>

namespace INNC
{

#define generate_unary_op_list(op)                                           \
  class op##_helper                                                          \
  {                                                                          \
  public:                                                                    \
    constexpr static void (*spec_list[types::Count][types::Count])(          \
        TensorFrame * to, TensorFrame *from) = {                             \
        {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,        \
         op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,       \
         op<std::int8_t, float>, op<std::int8_t, double>},                   \
        {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,      \
         op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,     \
         op<std::int16_t, float>, op<std::int16_t, double>},                 \
        {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,      \
         op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,     \
         op<std::int32_t, float>, op<std::int32_t, double>},                 \
        {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,      \
         op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,     \
         op<std::int64_t, float>, op<std::int64_t, double>},                 \
        {op<float, std::int8_t>, op<float, std::int16_t>,                    \
         op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>, \
         op<float, double>},                                                 \
        {op<double, std::int8_t>, op<double, std::int16_t>,                  \
         op<double, std::int32_t>, op<double, std::int64_t>,                 \
         op<double, float>, op<double, double>}};                            \
  }

// the type on the left would be the type of the return value
#define generate_binary_op_list(op)                                          \
  class op##_helper                                                          \
  {                                                                          \
  public:                                                                    \
    constexpr static void (*spec_list[types::Count][types::Count])(          \
        TensorFrame * dst, TensorFrame *l, TensorFrame *r) = {               \
        {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,        \
         op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,       \
         op<std::int8_t, float>, op<std::int8_t, double>},                   \
        {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,      \
         op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,     \
         op<std::int16_t, float>, op<std::int16_t, double>},                 \
        {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,      \
         op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,     \
         op<std::int32_t, float>, op<std::int32_t, double>},                 \
        {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,      \
         op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,     \
         op<std::int64_t, float>, op<std::int64_t, double>},                 \
        {op<float, std::int8_t>, op<float, std::int16_t>,                    \
         op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>, \
         op<float, double>},                                                 \
        {op<double, std::int8_t>, op<double, std::int16_t>,                  \
         op<double, std::int32_t>, op<double, std::int64_t>,                 \
         op<double, float>, op<double, double>}};                            \
  }

  size_t TensorFrame::cnt_from_index(const SizeVec &index) const noexcept
  {
    assertm(index.size() == this->sizes.size(), "index mismatches sizes");
    size_t pos = this->offset;
    for (size_t i = 0; i < index.size(); ++i)
    {
      pos += this->strides[i] * index[i];
    }
    return pos;
  }

  std::string TensorFrame::to_string_helper(std::ptrdiff_t offset,
                                            size_t l) const
  {
    if (l == this->sizes.size())
    {
      void *ptr = this->data_.get() + offset;
      return INNC::innc_type_to_string(ptr, this->dtype);
    }
    bool begin = true;
    std::string ret = "[";
    for (size_t i = 0; i < this->sizes[l]; ++i)
    {
      std::ptrdiff_t sub_offset =
          offset + i * this->strides[l] * INNC::size_of(this->dtype);
      if (begin)
      {
        ret += to_string_helper(sub_offset, l + 1);
        begin = false;
        continue;
      }
      ret += ", " + to_string_helper(sub_offset, l + 1);
    }
    return ret + "]";
  }

  std::string TensorFrame::to_string() const
  {
    if (this->sizes.size() == 0)
      return INNC::innc_type_to_string(
          data_.get() + offset * INNC::size_of(dtype), dtype);
    return to_string_helper(offset * INNC::size_of(dtype));
  }

  TensorFrame::~TensorFrame() = default;

  TensorFrame::TensorFrame(TensorFrame &&t) = default;

  std::unique_ptr<TensorFrame>
  TensorFrame::make_tf(types dtype, is_same_wo_cvref<SizeVec> auto &&sizes)
  {
    run_expect(sizes.size() != 0, "`size` must have at least one dimension.");
    SizeVec strides;
    strides.resize(sizes.size());
    strides[sizes.size() - 1] = 1;
    for (int i = sizes.size() - 1; i >= 1; --i)
    {
      strides[i - 1] = strides[i] * sizes[i];
    }
    return std::make_unique<TensorFrame>(
        dtype, std::forward<decltype(sizes)>(sizes), strides, 0);
  }

  TensorFrame::TensorFrame(types dtype, const SizeVec &sizes,
                           const SizeVec &strides, const size_t offset,
                           bool prealloc)
      : dtype(dtype), sizes(sizes), strides(strides), offset(offset)
  {
    size_t space = size_of(dtype);
    for (auto s : sizes)
    {
      space *= s;
    }
    if (prealloc)
      data_.create(space);
    this->requires_grad = false;
    this->retain_grad = false;
    this->_version = 0;
    this->grad_fn.reset();
  }

  // TODO 2: fast path, concurrency, iterator, Broadcasting
  void for_each_sizevec(const SizeVec &range, auto op)
  {
    [op, &range]()
    {
      SizeVec sv;
      auto last_idx = range.size() - 1;
      sv.resize(last_idx + 1);
      for (auto &it : sv)
        it = 0;
      while (true)
      {
        size_t ptr = last_idx;
        while (sv[ptr] == range[ptr])
        {
          if (ptr == 0)
            return;
          sv[ptr] = 0;
          ++sv[--ptr];
        }
        op(sv);
        ++sv[last_idx];
      };
    }();
  }

  template <typename L, typename R>
  void tensor_add(TensorFrame *dst, TensorFrame *l, TensorFrame *r)
  {
    for_each_sizevec(dst->sizes, [=](const SizeVec &sv)
                     { *(reinterpret_cast<L *>(dst->data_.get()) + dst->cnt_from_index(sv)) =
                           *(reinterpret_cast<L *>(l->data_.get()) + l->cnt_from_index(sv)) +
                           *(reinterpret_cast<R *>(r->data_.get()) + r->cnt_from_index(sv)); });
  }

  template <typename L, typename R>
  void tensor_mul(TensorFrame *dst, TensorFrame *l, TensorFrame *r)
  {
    for_each_sizevec(dst->sizes, [=](const SizeVec &sv)
                     { *(reinterpret_cast<L *>(dst->data_.get()) + dst->cnt_from_index(sv)) =
                           *(reinterpret_cast<L *>(l->data_.get()) + l->cnt_from_index(sv)) *
                           *(reinterpret_cast<R *>(r->data_.get()) + r->cnt_from_index(sv)); });
  }

  template <typename TensorType, typename NumberType>
  void tensor_fill(TensorFrame *tdata, TensorFrame *ndata)
  {
    TensorType num = *reinterpret_cast<NumberType *>(ndata->data_.get());
    for_each_sizevec(tdata->sizes, [tdata, num](const SizeVec &sv)
                     { *(reinterpret_cast<TensorType *>(tdata->data_.get()) +
                         tdata->cnt_from_index(sv)) = num; });
  }

  template <typename ToType, typename FromType>
  void tensor_to_type(TensorFrame *todata, TensorFrame *fromdata)
  {
    for_each_sizevec(todata->sizes, [=](const SizeVec &sv)
                     { *(reinterpret_cast<ToType *>(todata->data_.get()) +
                         todata->cnt_from_index(sv)) =
                           *(reinterpret_cast<FromType *>(fromdata->data_.get()) +
                             fromdata->cnt_from_index(sv)); });
  }

  template <typename ToType, typename FromType>
  void tensor_sum(TensorFrame *todata, TensorFrame *fromdata)
  {
    SizeVec sv;
    auto last_idx = fromdata->sizes.size() - 1;
    sv.resize(last_idx + 1);
    for (auto &it : sv)
      it = 0;
    while (true)
    {
      size_t ptr = last_idx;
      while (sv[ptr] == fromdata->sizes[ptr])
      {
        if (ptr == 0)
          return;
        sv[ptr] = 0;
        ++sv[--ptr];
      }
      *reinterpret_cast<ToType *>(todata->data_.get()) +=
          *(reinterpret_cast<FromType *>(fromdata->data_.get()) +
            fromdata->cnt_from_index(sv));
      ++sv[last_idx];
    }
  }

  generate_binary_op_list(tensor_add);
  generate_binary_op_list(tensor_mul);
  generate_unary_op_list(tensor_fill);
  generate_unary_op_list(tensor_to_type);
  generate_unary_op_list(tensor_sum);

  std::unique_ptr<TensorFrame> TensorFrame::ones(const SizeVec &sizes,
                                                 types dtype)
  {
    auto ret = make_tf(dtype, sizes);
    auto one_ = TensorFrame::make_tf(i8, SizeVec{1});
    *one_->data_.get() = 1;
    tensor_fill_helper::spec_list[dtype][i8](ret.get(), one_.get());
    return ret;
  }

  std::unique_ptr<TensorFrame> TensorFrame::zeros(const SizeVec &sizes,
                                                  types dtype)
  {
    auto ret = make_tf(dtype, sizes);
    auto start = ret->data_.get();
    std::fill(start, start + ret->data_.get_size(), 0);
    return ret;
  }

  std::unique_ptr<TensorFrame> TensorFrame::zeros_like(const TensorFrame &t)
  {
    return TensorFrame::zeros(t.sizes, t.dtype);
  }

  std::unique_ptr<TensorFrame> TensorFrame::ones_like(const TensorFrame &t)
  {
    return TensorFrame::ones(t.sizes, t.dtype);
  }

  std::unique_ptr<TensorFrame>
  TensorFrame::from_blob(void *data, const SizeVec &sizes, types dtype)
  {
    auto ret = make_tf(dtype, sizes);
    std::memcpy(ret->data_.get(), data, ret->numel() * INNC::size_of(dtype));
    return ret;
  }

  void check_same_size(const TensorFrame &lhs, const TensorFrame &rhs)
  {
    bool dim_eq = true;
    if (lhs.sizes.size() == rhs.sizes.size())
    {
      for (size_t i = 0; i < lhs.sizes.size(); ++i)
        if (lhs.sizes[i] != rhs.sizes[i])
        {
          dim_eq = false;
          break;
        }
    }
    else
      dim_eq = false;
    if (!dim_eq)
      std::cerr << "Tensors' dimension must match. Trying to add " +
                       lhs.sizes.to_string() + " with " + rhs.sizes.to_string()
                << std::endl
                << "Broadcasting has not been supported yet.";
  }

  TensorFrame &TensorFrame::operator+=(const TensorFrame &rhs)
  {
    run_expect(
        !requires_grad,
        "This inplace operation cannot perform on a tensor that requires grad.");
    tensor_add_helper::spec_list[dtype][rhs.dtype](
        this, this, const_cast<TensorFrame *>(&rhs));
    return *this;
  }

  template <typename ForwardType>
  std::unique_ptr<TensorFrame> apply_no_grad_binary_op(TensorFrame &lhs,
                                                       TensorFrame &rhs)
  {
    types lt = INNC::larger_type(lhs.dtype, rhs.dtype);
    auto ret = TensorFrame::make_tf(lt, lhs.sizes);
    if (lhs.dtype == lt)
    {
      ForwardType::spec_list[lhs.dtype][rhs.dtype](ret.get(), &lhs, &rhs);
    }
    else
    {
      ForwardType::spec_list[rhs.dtype][lhs.dtype](ret.get(), &rhs, &lhs);
    }
    return ret;
  }

  template <typename ForwardType, typename BackwardType>
  std::unique_ptr<TensorFrame>
  apply_binary_operator(const std::shared_ptr<TensorFrame> &lhs,
                        const std::shared_ptr<TensorFrame> &rhs)
  {
    check_same_size(*lhs.get(), *rhs.get());
    auto ret = apply_no_grad_binary_op<ForwardType>(*lhs.get(), *rhs.get());
    if (!lhs->requires_grad && !rhs->requires_grad)
      return ret;
    ret->requires_grad = true;
    ret->grad_fn.reset(new BackwardType(ret.get(), {lhs, rhs}));
    return ret;
  }

  size_t TensorFrame::numel() const noexcept
  {
    size_t ret = 1;
    for (auto i : this->sizes)
      ret *= i;
    return ret;
  }

  void TensorFrame::release() noexcept { data_.reset(); }

  inline INNC::types TensorFrame::type() const { return this->dtype; }

  std::unique_ptr<TensorFrame> TensorFrame::type(types t)
  {
    auto tf = make_tf(t, sizes);
    tensor_to_type_helper::spec_list[t][dtype](tf.get(), this);
    return tf;
  }

  void TensorFrame::try_accumulate_grad(TensorFrame &tf)
  {
    if (!requires_grad)
      return;
    if (grad_fn.get() != nullptr)
      ++grad_fn->accumulated_n_outway;
    if (grad.get() == nullptr)
      grad = TensorFrame::zeros_like(*this);
    *grad += tf;
  }

  void refresh_n_outway(TensorFrame *tf)
  {
    std::queue<TensorFrame *> q;
    tf->grad_fn->n_outway = 0;
    tf->grad_fn->back_version = ++Backward::global_back_version;
    q.push(tf);
    while (!q.empty())
    {
      auto t = q.front();
      q.pop();
      for (const auto &it : t->grad_fn->input_tfs)
      {
        if (it->grad_fn.get() != nullptr &&
            it->grad_fn->back_version < Backward::global_back_version)
        {
          it->grad_fn->back_version = Backward::global_back_version;
          it->grad_fn->n_outway = 0;
          q.push(it.get());
        }
      }
    }
    tf->grad_fn->back_version = ++Backward::global_back_version;
    q.push(tf);
    while (!q.empty())
    {
      auto t = q.front();
      q.pop();
      for (const auto &it : t->grad_fn->input_tfs)
      {
        if (it->grad_fn.get() != nullptr)
        {
          ++it->grad_fn->n_outway;
          if (it->grad_fn->back_version < Backward::global_back_version)
          {
            it->grad_fn->back_version = Backward::global_back_version;
            q.push(it.get());
          }
        }
      }
    }
  }

  void TensorFrame::backward()
  {
    run_expect(requires_grad,
               "Cannot backward from a tensor that does not require grad.");
    run_expect(grad_fn.get() != nullptr,
               "Cannot backward from a tensor that has no grad_func");
    run_expect(numel() == 1,
               "Only scalar number could be the start of a backward propagation");
    refresh_n_outway(this);
    std::queue<TensorFrame *> q; // TODO 3 multiprocessing
    q.push(this);
    grad = ones_like(*this);
    grad_fn->back_version = ++Backward::global_back_version;
    while (!q.empty())
    {
      auto t = q.front();
      q.pop();
      t->grad_fn->step_back();
      if (!t->retain_grad)
        t->grad.reset();
      for (const auto &it : t->grad_fn->input_tfs)
      {
        if (it->grad_fn.get() != nullptr &&
            it->grad_fn->back_version < Backward::global_back_version &&
            it->grad_fn->is_ready())
        {
          it->grad_fn->back_version = Backward::global_back_version;
          q.push(it.get());
        }
      }
    }
  }

  Tensor::Tensor() = default;
  Tensor::Tensor(Tensor &&t) = default;
  Tensor::Tensor(std::unique_ptr<TensorFrame> &tf) : fptr(std::move(tf)){};
  Tensor::Tensor(std::unique_ptr<TensorFrame> &&tf) : fptr(std::move(tf)){};
  Tensor::Tensor(std::shared_ptr<TensorFrame> &tf) : fptr(tf){};
  Tensor &Tensor::operator=(const Tensor &t) = default;
  Tensor &Tensor::operator=(Tensor &&t) = default;
  Tensor::Tensor(const SizeVec &sizes, types dtype)
      : fptr(TensorFrame::make_tf(dtype, sizes)) {}
  Tensor::~Tensor() = default;

  void Tensor::backward() { fptr->backward(); }

  std::string Tensor::to_string() const
  {
    if (fptr == nullptr)
      return "[]";
    return fptr->to_string();
  }

  const SizeVec Tensor::size() const { return fptr->sizes; }

  const SizeVec Tensor::stride() const { return fptr->strides; }

  Tensor Tensor::zeros(const SizeVec &size, types t)
  {
    return Tensor(TensorFrame::zeros(size, t));
  }

  Tensor Tensor::ones(const SizeVec &size, types t)
  {
    return Tensor(TensorFrame::ones(size, t));
  }

  Tensor Tensor::zeros_like(const Tensor &t)
  {
    return Tensor(TensorFrame::zeros_like(*t.fptr));
  }

  Tensor Tensor::ones_like(const Tensor &t)
  {
    return Tensor(TensorFrame::ones_like(*t.fptr));
  }

  Tensor Tensor::from_blob(void *data, const SizeVec &sizes, types dtype)
  {
    return Tensor(TensorFrame::from_blob(data, sizes, dtype));
  }

  size_t Tensor::numel() const noexcept { return fptr->numel(); }

  void Tensor::release() noexcept { fptr->release(); }

  INNC::types Tensor::type() const { return fptr->type(); }

  Tensor Tensor::type(types t) { return Tensor(fptr->type(t)); }

  Tensor &Tensor::operator+=(const Tensor &rhs)
  {
    *fptr += *rhs.fptr;
    return *this;
  }

  Tensor operator+(const Tensor &lhs, const Tensor &rhs)
  {
    auto tf =
        apply_binary_operator<tensor_add_helper, AddBack>(lhs.fptr, rhs.fptr);
    return Tensor(tf);
  }

  Tensor operator*(const Tensor &lhs, const Tensor &rhs)
  {
    auto tf =
        apply_binary_operator<tensor_mul_helper, MulBack>(lhs.fptr, rhs.fptr);
    return Tensor(tf);
  }

  bool Tensor::requires_grad() const noexcept { return fptr->requires_grad; }

  void Tensor::requires_grad(bool b)
  {
    if (b)
      run_expect(fptr->type() == f32 || fptr->type() == f64,
                 "Only matrices of floating point number can require grad");
    if (!b)
      fptr.reset();
    fptr->requires_grad = b;
  }

  bool Tensor::retain_grad() const noexcept { return fptr->retain_grad; }
  void Tensor::retain_grad(bool b) noexcept { fptr->retain_grad = b; }

  Tensor Tensor::grad() const noexcept
  {
    if (fptr->grad.get() == nullptr)
      return Tensor();
    return Tensor(fptr->grad);
  }

  std::unique_ptr<TensorFrame> TensorFrame::sum() const
  {
    INNC::types dst_t;
    if (dtype <= i64)
      dst_t = i64;
    else
      dst_t = f64;
    auto tf = zeros(SizeVec{1}, dst_t);
    tensor_sum_helper::spec_list[dst_t][dtype](tf.get(),
                                               const_cast<TensorFrame *>(this));
    return tf;
  }

  Tensor Tensor::sum() const
  {
    auto tf = fptr->sum();
    if (fptr->requires_grad)
    {
      tf->requires_grad = true;
      tf->grad_fn.reset(new SumBack(tf.get(), {fptr}));
    }
    return Tensor(tf);
  }

  void TensorFrame::zero_grad() const noexcept
  {
    if (grad.get() == nullptr)
      return;
    auto d_begin = grad->data_.get();
    std::fill(d_begin, d_begin + INNC::size_of(dtype) * numel(), 0);
  }

  void Tensor::zero_grad() const noexcept { fptr->zero_grad(); }

  std::unique_ptr<TensorFrame> TensorFrame::operator[](const std::string &slice)
  {
    std::vector<std::string> each_dim = ssplit(slice, ',');
    SizeVec _sizes;
    SizeVec _strides;
    size_t _offset = offset;
    run_expect(each_dim.size() <= this->sizes.size(),
               "Dimension of slicing is larger than sizes.");
    for (size_t dim = 0; dim < this->sizes.size(); ++dim)
    {
      std::vector<std::string> split_slice;
      if (dim < each_dim.size())
      {
        split_slice = ssplit(each_dim[dim], ':');
        for (auto &s : split_slice)
          trim_(s);
      }
      else
        split_slice = {"", "", ""};
      if (split_slice.size() == 1)
      { // like a[-2]
        long long idx = std::stoi(split_slice[0]);
        if (idx < 0)
          idx += sizes[dim];
        run_expect(
            idx >= 0 && idx < static_cast<long long>(sizes[dim]),
            sformat("index %d is out of bounds for dimension %lu with size %lu",
                    idx, dim, sizes[dim]));
        _offset += strides[dim] * idx;
      }
      else
      { // like a[-2:]
        if (sizes[dim] == 0)
        {
          _sizes.push_back(0);
          _strides.push_back(0);
          continue;
        }
        for (int i = split_slice.size(); i < 3; ++i)
          split_slice.push_back("");
        long long step, beg, end;
        if (split_slice[2].size() != 0)
        {
          step = stoll(split_slice[2]);
          run_expect(step != 0, "slice step cannot be zero");
        }
        else
          step = 1;
        if (split_slice[0].size() != 0)
        {
          beg = stoll(split_slice[0]);
          if (beg < 0)
            beg += sizes[dim];
          if (beg < 0)
          {
            if (step > 0)
              beg = 0;
            else
            {
              _sizes.push_back(0);
              _strides.push_back(0);
              continue;
            }
          }
          if (beg >= static_cast<long long>(sizes[dim]))
          {
            if (step > 0)
            {
              _sizes.push_back(0);
              _strides.push_back(0);
              continue;
            }
            else
              beg = sizes[dim] - 1;
          }
        }
        else
          beg = step > 0 ? 0 : sizes[dim] - 1;
        if (split_slice[1].size() != 0)
        {
          end = stoll(split_slice[1]);
          if (end < 0)
            end += sizes[dim];
          if (step > 0)
          {
            if (end <= 0)
            {
              _sizes.push_back(0);
              _strides.push_back(0);
              continue;
            }
            if (end > static_cast<long long>(sizes[dim]))
              end = sizes[dim];
          }
          else
          {
            if (end < 0)
              end = -1;
            if (end >= static_cast<long long>(sizes[dim]) - 1)
            {
              _sizes.push_back(0);
              _strides.push_back(0);
              continue;
            }
          }
        }
        else
          end = step > 0 ? sizes[dim] : -1;
        if ((step > 0 && beg >= end) || (step < 0 && beg <= end))
        {
          _sizes.push_back(0);
          _strides.push_back(0);
          continue;
        }
        _sizes.push_back((std::abs(end - beg) - 1) / std::abs(step) + 1);
        _strides.push_back(step * strides[dim]);
        _offset += beg * strides[dim];
      } // like a[-2:]
    }
    auto ret = make_unique<TensorFrame>(dtype, _sizes, _strides, _offset, false);
    ret->data_ = data_;
    ret->grad = grad;
    return ret;
  }

  Tensor Tensor::operator[](const std::string &slice)
  {
    return Tensor((*this->fptr)[slice]);
  }

  std::unique_ptr<TensorFrame> INNC::TensorFrame::reshape_without_grad(const TensorFrame &input, const SizeVec &sizes)
  {
    size_t input_num = input.numel();
    size_t output_num = 1;
    for (auto i : sizes)
    {
      output_num *= i;
    }

    run_expect(input_num == output_num, "You're turning the tensor into an impossible shape.");
    // Create a tensorframe of a given shape
    auto tf = TensorFrame::ones(sizes, input.type());
    std::memcpy(tf.get()->data_.get() + tf.get()->offset, input.data_.get() + input.offset, output_num * size_of(input.type()));
    return tf;
  }

  Tensor Tensor::reshape(const Tensor &input, const SizeVec &sizes)
  {
    auto tf = TensorFrame::reshape(input.fptr, sizes);

    return Tensor(tf);
  }

  // Tensor Tensor::reshape(const Tensor &input, const std::initializer_list<int> &sizes) {
  //   auto tf = TensorFrame::reshape(input.fptr, new SizeVec(sizes));

  //   return Tensor(tf);
  // }

  std::unique_ptr<TensorFrame> INNC::TensorFrame::reshape(const std::shared_ptr<TensorFrame> &input, const SizeVec &sizes)
  {
    auto tf = reshape_without_grad(*input.get(), sizes);

    if (!input.get()->requires_grad)
      return tf;
    tf->requires_grad = true;
    tf->grad_fn.reset(new ReshapeBack(tf.get(), {input}));
    return tf;
  }

  Tensor Tensor::reshape(const Tensor &input, const std::string &sizes)
  {
    auto tf = TensorFrame::reshape(input.fptr, sizes);

    return Tensor(tf);
  }

  std::unique_ptr<TensorFrame> INNC::TensorFrame::reshape(const std::shared_ptr<TensorFrame> &input, const std::string &sizes)
  {
    std::string size_text = sizes;
    SizeVec sizes_;
    size_text.erase(std::remove(size_text.begin(), size_text.end(), '['), size_text.end());
    size_text.erase(std::remove(size_text.begin(), size_text.end(), ']'), size_text.end());
    size_text.erase(std::remove(size_text.begin(), size_text.end(), ' '), size_text.end());

    sizes_.resize(0);

    bool have_minus_one = false;
    size_t minus_one_index;

    size_t startPos = 0;
    size_t endPos;

    while ((endPos = size_text.find(',', startPos)) != std::string::npos)
    {
      std::string numberStr = size_text.substr(startPos, endPos - startPos);

      if (numberStr[0] == '-'){
        run_expect(!have_minus_one, "You cannot have two dimensions be -1");
        have_minus_one = true;
        minus_one_index = sizes_.size();
        sizes_.push_back(0);

        startPos = endPos + 1;
      }else{
        int number = std::stoi(numberStr);
        sizes_.push_back(number);
        startPos = endPos + 1;
      }
    }

    std::string lastNumberStr = size_text.substr(startPos);
    if (lastNumberStr[0] == '-') {
      run_expect(!have_minus_one, "You cannot have two dimensions be -1");
        have_minus_one = true;
        minus_one_index = sizes_.size();
    }
    int lastNumber = std::stoi(lastNumberStr);
    sizes_.push_back(lastNumber);

    if (have_minus_one){
      int s = 1;
      for (size_t i = 0; i < sizes_.size(); i++){
        if (i != minus_one_index){
          s *= sizes_[i];
        }
      }
      sizes_[minus_one_index] = (input->numel()/s);
    }

    auto tf = reshape_without_grad(*input.get(), sizes_);

    if (!input.get()->requires_grad)
      return tf;
    tf->requires_grad = true;
    tf->grad_fn.reset(new ReshapeBack(tf.get(), {input}));
    return tf;
  }


} // namespace INNC
