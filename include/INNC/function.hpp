#pragma once

#include "storage.hpp"
#include "tensor.hpp"
#include <array>
#include <memory>

namespace INNC {
class Tensor;
class Backward {
public:
  std::vector<std::shared_ptr<INNC::TensorImpl>>
      input_tfs; // TODO stricter encapsulation
  TensorImpl *this_tf;
  size_t n_outway;
  size_t accumulated_n_outway; // TODO partial graph
  static size_t global_back_version;
  size_t back_version;

  Backward(TensorImpl *this_tf,
           const std::vector<std::shared_ptr<INNC::TensorImpl>> &input_tfs);
  TensorImpl &get_out_grad();
  inline bool is_ready() { return accumulated_n_outway == n_outway; }
  virtual void step_back() = 0;
  virtual ~Backward();
};

class AddBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class SubBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class MulBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class DivBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class SumBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class NoBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class CloneBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

class CatBack : public Backward {
public:
  using Backward::Backward;
  void step_back() override;
};

} // namespace INNC
