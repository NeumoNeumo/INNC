#pragma once
#include "INNC/dispatcher.hpp"
#include "INNC/tensorImpl.hpp"
#include "INNC/types.hpp"
#include "INNC/utils/utils.hpp"
#include <iostream>

namespace INNC {
namespace native {

template <typename L, typename R>
void tensor_add(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<L *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(broadcast_range(l->size(), r->size()),
                   [=](const SizeVec &sv) {
                     *(dst_ptr + dst->cnt_from_aug_index(sv)) =
                         *(l_ptr + l->cnt_from_aug_index(sv)) +
                         *(r_ptr + r->cnt_from_aug_index(sv));
                   });
}

template <typename L, typename R>
void tensor_sub(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr =
      reinterpret_cast<innc_common_t<L, R> *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(broadcast_range(l->size(), r->size()),
                   [=](const SizeVec &sv) {
                     *(dst_ptr + dst->cnt_from_aug_index(sv)) =
                         *(l_ptr + l->cnt_from_aug_index(sv)) -
                         *(r_ptr + r->cnt_from_aug_index(sv));
                   });
}

template <typename L, typename R>
void tensor_mul(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<L *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) *
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename L, typename R>
void tensor_div(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr =
      reinterpret_cast<innc_common_t<L, R> *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) /
        *(r_ptr + r->cnt_from_aug_index(sv));
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

template <typename ToType, typename FromType>
void tensor_mean(TensorImpl *to, const TensorImpl *from) {
  std::cout << "opop" << to->to_string() << from->to_string() << std::endl;
  if (from->view->numel() == 0)
    return;
  SizeVec sv;
  ToType n = from->view->numel();
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
        if (ptr == 0){
        *to_ptr = *(to_ptr) / n;
          return;
        }
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
  *to_ptr = *(to_ptr) / n;
}

template <typename D, typename L, typename R>
void tensor_mul_acc_f(TensorImpl *dst, const TensorImpl *l,
                      const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<D *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  auto rg = broadcast_range(dst->size(), l->size());
  rg = broadcast_range(rg, r->size());
  for_each_sizevec(rg, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) +=
        *(l_ptr + l->cnt_from_aug_index(sv)) *
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename ToType, typename FromType>
void tensor_clone(TensorImpl *to, const TensorImpl *from) {
  if (to->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of output has not been implemented",
                to_string(to->dlayout).c_str()));
  if (from->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of input has not been implemented",
                to_string(from->dlayout).c_str()));
  ToType *to_ptr = reinterpret_cast<ToType *>(to->data_->get_blob());
  FromType *from_ptr = reinterpret_cast<FromType *>(from->data_->get_blob());
  for_each_sizevec(to->view->sizes, [=](const SizeVec &sv) {
    *(to_ptr + to->cnt_from_index(sv)) = *(from_ptr + from->cnt_from_index(sv));
  });
}

template <typename ToType, typename FromType>
void tensor_abs(TensorImpl *to, const TensorImpl *from,
                const TensorImpl *grad = nullptr) {
  if (to->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of output has not been implemented",
                to_string(to->dlayout).c_str()));
  if (from->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of input has not been implemented",
                to_string(from->dlayout).c_str()));
  ToType *to_ptr = reinterpret_cast<ToType *>(to->data_->get_blob());
  FromType *from_ptr = reinterpret_cast<FromType *>(from->data_->get_blob());
  if (grad == nullptr)
    for_each_sizevec(to->view->sizes, [=](const SizeVec &sv) {
      FromType tmp = *(from_ptr + from->cnt_from_index(sv));
      *(to_ptr + to->cnt_from_index(sv)) = tmp >= 0 ? tmp : -tmp;
    });
  else {
    FromType *grad_ptr = reinterpret_cast<FromType *>(grad->data_->get_blob());
    for_each_sizevec(to->view->sizes, [=](const SizeVec &sv) {
      FromType tmp = *(from_ptr + from->cnt_from_index(sv));
      if (tmp >= 0) {
        *(grad_ptr + grad->cnt_from_index(sv)) = 1;
        *(to_ptr + to->cnt_from_index(sv)) = tmp;
      } else {
        *(grad_ptr + grad->cnt_from_index(sv)) = -1;
        *(to_ptr + to->cnt_from_index(sv)) = -tmp;
      }
    });
  }
}

template <typename T>
void tensor_max(TensorImpl *to, const TensorImpl *from,
                const TensorImpl *grad = nullptr) {
  if (to->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of output has not been implemented",
                to_string(to->dlayout).c_str()));
  if (from->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of input has not been implemented",
                to_string(from->dlayout).c_str()));
  T *to_ptr = reinterpret_cast<T *>(to->data_->get_blob());
  T *from_ptr = reinterpret_cast<T *>(from->data_->get_blob());
  *to_ptr = *from_ptr;
  SizeVec max_sv;
  for_each_sizevec(from->view->sizes, [=, &max_sv](const SizeVec &sv) {
    T t = *(from_ptr + from->cnt_from_index(sv));
    if (__LIKELY(*to_ptr >= t))
      return;
    *to_ptr = t;
    max_sv = sv;
  });
  if (grad != nullptr)
    *(reinterpret_cast<T *>(grad->data_->get_blob()) +
      grad->cnt_from_index(max_sv)) = 1;
}

template <typename T>
void tensor_min(TensorImpl *to, const TensorImpl *from,
                const TensorImpl *grad = nullptr) {
  if (to->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of output has not been implemented",
                to_string(to->dlayout).c_str()));
  if (from->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of input has not been implemented",
                to_string(from->dlayout).c_str()));
  T *to_ptr = reinterpret_cast<T *>(to->data_->get_blob());
  T *from_ptr = reinterpret_cast<T *>(from->data_->get_blob());
  *to_ptr = *from_ptr;
  SizeVec min_sv;
  for_each_sizevec(from->view->sizes, [=, &min_sv](const SizeVec &sv) {
    T t = *(from_ptr + from->cnt_from_index(sv));
    if (__LIKELY(*to_ptr <= t))
      return;
    *to_ptr = t;
    min_sv = sv;
  });
  if (grad != nullptr)
    *(reinterpret_cast<T *>(grad->data_->get_blob()) +
      grad->cnt_from_index(min_sv)) = 1;
}

template <typename L, typename R>
void tensor_lt(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<char *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) <
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename L, typename R>
void tensor_gt(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<char *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) >
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename L, typename R>
void tensor_le(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<char *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) <=
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename L, typename R>
void tensor_ge(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<char *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) >=
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename L, typename R>
void tensor_eq(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<char *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) ==
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename L, typename R>
void tensor_ne(TensorImpl *dst, const TensorImpl *l, const TensorImpl *r) {
  auto dst_ptr = reinterpret_cast<char *>(dst->data_->get_blob());
  auto l_ptr = reinterpret_cast<L *>(l->data_->get_blob());
  auto r_ptr = reinterpret_cast<R *>(r->data_->get_blob());
  for_each_sizevec(dst->view->sizes, [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) =
        *(l_ptr + l->cnt_from_aug_index(sv)) !=
        *(r_ptr + r->cnt_from_aug_index(sv));
  });
}

template <typename D, typename L, typename R>
void tensor_div_back_numerator(TensorImpl *dst, const TensorImpl *out_grad,
                               const TensorImpl *den) {
  auto dst_ptr = reinterpret_cast<D *>(dst->data_->get_blob());
  auto out_ptr = reinterpret_cast<L *>(out_grad->data_->get_blob());
  auto den_ptr = reinterpret_cast<R *>(den->data_->get_blob());
  for_each_sizevec(out_grad->size(), [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_aug_index(sv)) +=
        *(out_ptr + out_grad->cnt_from_aug_index(sv)) /
        *(den_ptr + den->cnt_from_aug_index(sv));
  });
}

template <typename Den, typename Quot>
void tensor_div_back_denominator(TensorImpl *dst, const TensorImpl *outgrad,
                                 const TensorImpl *quot,
                                 const TensorImpl *den) {
  auto dst_ptr = reinterpret_cast<Den *>(dst->data_->get_blob());
  auto outgrad_ptr = reinterpret_cast<Quot *>(outgrad->data_->get_blob());
  auto quot_ptr = reinterpret_cast<Quot *>(quot->data_->get_blob());
  auto den_ptr = reinterpret_cast<Den *>(den->data_->get_blob());
  for_each_sizevec(quot->size(), [=](const SizeVec &sv) {
    *(dst_ptr + dst->cnt_from_index(sv)) -=
        *(outgrad_ptr + outgrad->cnt_from_aug_index(sv)) *
        *(quot_ptr + quot->cnt_from_aug_index(sv)) /
        *(den_ptr + den->cnt_from_aug_index(sv));
  });
}

// copy the data from onr tensor
template <typename ToType, typename FromType>
void tensor_cat(TensorImpl *to, const TensorImpl *from, size_t offset) {
  if (to->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of output has not been implemented",
                to_string(to->dlayout).c_str()));
  if (from->dlayout != layouts::strided)
    throw std::logic_error(
        sformat("The layout %s of input has not been implemented",
                to_string(from->dlayout).c_str()));
  ToType *to_ptr = reinterpret_cast<ToType *>(to->data_->get_blob());
  FromType *from_ptr = reinterpret_cast<FromType *>(from->data_->get_blob());
  for_each_sizevec(from->view->sizes, [=](const SizeVec &sv) {
    *(to_ptr + offset + to->cnt_from_index(sv)) =
        *(from_ptr + from->cnt_from_index(sv));
  });
}

generate_binary_op_helper(tensor_add);
generate_binary_op_helper(tensor_sub);
generate_binary_op_helper(tensor_mul);
generate_binary_op_helper(tensor_div);
generate_binary_op_helper(tensor_lt);
generate_binary_op_helper(tensor_gt);
generate_binary_op_helper(tensor_le);
generate_binary_op_helper(tensor_ge);
generate_binary_op_helper(tensor_eq);
generate_binary_op_helper(tensor_ne);
generate_unary_op_helper(tensor_fill);
generate_unary_op_helper(tensor_to_type);
generate_unary_op_helper(tensor_sum);
generate_unary_op_helper(tensor_mean);
generate_unary_op_helper(tensor_clone);
generate_unary_grad_op_helper(tensor_abs);
generate_ffi_op_helper(tensor_mul_acc_f);
generate_ffi_op_helper(tensor_div_back_numerator);
generate_ff_op4_helper(tensor_div_back_denominator);
generate_i_op2_helper(tensor_max);
generate_i_op2_helper(tensor_min);
generate_unary_offset_op_helper(tensor_cat);

} // namespace native
} // namespace INNC
