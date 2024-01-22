namespace INNC {
#define generate_unary_grad_op_helper(op)                                      \
  class op##_helper {                                                          \
    constexpr static void (*spec_list[types::Count][types::Count])(            \
        TensorImpl * to, const TensorImpl *from, const TensorImpl *grad) = {   \
        {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,          \
         op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,         \
         op<std::int8_t, float>, op<std::int8_t, double>},                     \
        {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,        \
         op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,       \
         op<std::int16_t, float>, op<std::int16_t, double>},                   \
        {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,        \
         op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,       \
         op<std::int32_t, float>, op<std::int32_t, double>},                   \
        {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,        \
         op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,       \
         op<std::int64_t, float>, op<std::int64_t, double>},                   \
        {op<float, std::int8_t>, op<float, std::int16_t>,                      \
         op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>,   \
         op<float, double>},                                                   \
        {op<double, std::int8_t>, op<double, std::int16_t>,                    \
         op<double, std::int32_t>, op<double, std::int64_t>,                   \
         op<double, float>, op<double, double>}};                              \
                                                                               \
  public:                                                                      \
    static auto dispatch(types l_t, types r_t) { return spec_list[l_t][r_t]; } \
  }

#define generate_unary_op_helper(op)                                           \
  class op##_helper {                                                          \
    constexpr static void (*spec_list[types::Count][types::Count])(            \
        TensorImpl * to, const TensorImpl *from) = {                           \
        {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,          \
         op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,         \
         op<std::int8_t, float>, op<std::int8_t, double>},                     \
        {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,        \
         op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,       \
         op<std::int16_t, float>, op<std::int16_t, double>},                   \
        {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,        \
         op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,       \
         op<std::int32_t, float>, op<std::int32_t, double>},                   \
        {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,        \
         op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,       \
         op<std::int64_t, float>, op<std::int64_t, double>},                   \
        {op<float, std::int8_t>, op<float, std::int16_t>,                      \
         op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>,   \
         op<float, double>},                                                   \
        {op<double, std::int8_t>, op<double, std::int16_t>,                    \
         op<double, std::int32_t>, op<double, std::int64_t>,                   \
         op<double, float>, op<double, double>}};                              \
                                                                               \
  public:                                                                      \
    static auto dispatch(types l_t, types r_t) { return spec_list[l_t][r_t]; } \
  }

// the type on the left would be the type of the return value
#define generate_binary_op_helper(op)                                          \
  class op##_helper {                                                          \
    constexpr static void (*spec_list[types::Count][types::Count])(            \
        TensorImpl * dst, const TensorImpl *l, const TensorImpl *r) = {        \
        {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,          \
         op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,         \
         op<std::int8_t, float>, op<std::int8_t, double>},                     \
        {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,        \
         op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,       \
         op<std::int16_t, float>, op<std::int16_t, double>},                   \
        {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,        \
         op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,       \
         op<std::int32_t, float>, op<std::int32_t, double>},                   \
        {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,        \
         op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,       \
         op<std::int64_t, float>, op<std::int64_t, double>},                   \
        {op<float, std::int8_t>, op<float, std::int16_t>,                      \
         op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>,   \
         op<float, double>},                                                   \
        {op<double, std::int8_t>, op<double, std::int16_t>,                    \
         op<double, std::int32_t>, op<double, std::int64_t>,                   \
         op<double, float>, op<double, double>}};                              \
                                                                               \
  public:                                                                      \
    static auto dispatch(types l_t, types r_t) { return spec_list[l_t][r_t]; } \
  }

// float, float, int/float
#define generate_ffi_op_helper(op)                                             \
  class op##_helper {                                                          \
    constexpr static void (                                                    \
        *spec_list[float_type_n_][float_type_n_][types::Count])(               \
        TensorImpl * dst, const TensorImpl *l, const TensorImpl *r) = {        \
        {{op<float, float, std::int8_t>, op<float, float, std::int16_t>,       \
          op<float, float, std::int32_t>, op<float, float, std::int64_t>,      \
          op<float, float, float>, op<float, float, double>},                  \
         {op<float, double, std::int8_t>, op<float, double, std::int16_t>,     \
          op<float, double, std::int32_t>, op<float, double, std::int64_t>,    \
          op<float, double, float>, op<float, double, double>}},               \
        {{op<double, float, std::int8_t>, op<double, float, std::int16_t>,     \
          op<double, float, std::int32_t>, op<double, float, std::int64_t>,    \
          op<double, float, float>, op<double, float, double>},                \
         {op<double, double, std::int8_t>, op<double, double, std::int16_t>,   \
          op<double, double, std::int32_t>, op<double, double, std::int64_t>,  \
          op<double, double, float>, op<double, double, double>}}};            \
                                                                               \
  public:                                                                      \
    static auto dispatch(types dst_t, types l_t, types r_t) {                  \
      return spec_list[dst_t - float_type_idx_start_]                          \
                      [l_t - float_type_idx_start_][r_t];                      \
    }                                                                          \
  };

// cat
#define generate_unary_offset_op_helper(op)                                    \
  class op##_helper {                                                          \
    constexpr static void (*spec_list[types::Count][types::Count])(            \
        TensorImpl * to, const TensorImpl *from, const size_t offset) = {      \
        {op<std::int8_t, std::int8_t>, op<std::int8_t, std::int16_t>,          \
         op<std::int8_t, std::int32_t>, op<std::int8_t, std::int64_t>,         \
         op<std::int8_t, float>, op<std::int8_t, double>},                     \
        {op<std::int16_t, std::int8_t>, op<std::int16_t, std::int16_t>,        \
         op<std::int16_t, std::int32_t>, op<std::int16_t, std::int64_t>,       \
         op<std::int16_t, float>, op<std::int16_t, double>},                   \
        {op<std::int32_t, std::int8_t>, op<std::int32_t, std::int16_t>,        \
         op<std::int32_t, std::int32_t>, op<std::int32_t, std::int64_t>,       \
         op<std::int32_t, float>, op<std::int32_t, double>},                   \
        {op<std::int64_t, std::int8_t>, op<std::int64_t, std::int16_t>,        \
         op<std::int64_t, std::int32_t>, op<std::int64_t, std::int64_t>,       \
         op<std::int64_t, float>, op<std::int64_t, double>},                   \
        {op<float, std::int8_t>, op<float, std::int16_t>,                      \
         op<float, std::int32_t>, op<float, std::int64_t>, op<float, float>,   \
         op<float, double>},                                                   \
        {op<double, std::int8_t>, op<double, std::int16_t>,                    \
         op<double, std::int32_t>, op<double, std::int64_t>,                   \
         op<double, float>, op<double, double>}};                              \
                                                                               \
  public:                                                                      \
    static auto dispatch(types l_t, types r_t) { return spec_list[l_t][r_t]; } \
  }

// float, float
#define generate_ff_op4_helper(op)                                             \
  class op##_helper {                                                          \
    constexpr static void (*spec_list[float_type_n_][float_type_n_])(          \
        TensorImpl * dst, const TensorImpl *i1, const TensorImpl *i2,          \
        const TensorImpl *i3) = {{op<float, float>, op<float, double>},        \
                                 {op<double, float>, op<double, double>}};     \
                                                                               \
  public:                                                                      \
    static auto dispatch(types l_t, types r_t) {                               \
      return spec_list[l_t - float_type_idx_start_]                            \
                      [r_t - float_type_idx_start_];                           \
    }                                                                          \
  }

#define generate_i_op2_helper(op)                                              \
  class op##_helper {                                                          \
    constexpr static void (*spec_list[types::Count])(TensorImpl * dst,         \
                                                     const TensorImpl *i1,     \
                                                     const TensorImpl *i2) = { \
        op<std::int8_t>,  op<std::int16_t>, op<std::int32_t>,                  \
        op<std::int64_t>, op<float>,        op<double>};                       \
                                                                               \
  public:                                                                      \
    static auto dispatch(types t) { return spec_list[t]; }                     \
  }

} // namespace INNC
