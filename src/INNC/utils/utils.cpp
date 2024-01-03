#include "INNC/utils/utils.hpp"
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

} // namespace INNC
