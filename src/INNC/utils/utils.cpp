#include "INNC/utils/utils.hpp"
#include "INNC/exceptions.hpp"
#include <stdexcept>

namespace INNC {
std::vector<std::string> ssplit(const std::string &str, const char delimiter) {
  std::vector<std::string> result;
  if (str.size() == 0)
    return result;
  size_t prev = 0, next = 0;
  do {
    if (str[next] == delimiter) {
      result.push_back(str.substr(prev, next - prev));
      prev = next + 1;
    }
  } while (++next < str.size());
  result.push_back(str.substr(prev));
  return result;
}

SizeVec broadcast_range(const SizeVec *u, const SizeVec *v) {
  if (u->size() < v->size())
    std::swap(u, v);
  auto dim_u = u->size();
  auto dim_v = v->size();
  SizeVec ret;
  ret.resize(dim_u);
  for (size_t i = 1; i <= dim_u; ++i) {
    if (i <= dim_v) {
      auto su = (*u)[dim_u - i];
      auto sv = (*v)[dim_v - i];
      run_expect(su != 1 || sv != 1 || su == sv,
                 sformat("Size %s and Size %s are not broadcastable.",
                         u->to_string(), v->to_string()));
      ret[dim_u - i] = std::max(su, sv);
    } else {
      ret[dim_u - i] = (*u)[dim_u - i];
    }
  }
  return ret;
}

SizeVec broadcast_range(const SizeVec &u, const SizeVec &v) {
  return broadcast_range(&u, &v);
}
} // namespace INNC
