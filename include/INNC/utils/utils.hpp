#pragma once
#include <stdexcept>
#include <string>
#include <vector>

namespace INNC {
std::vector<std::string> ssplit(const std::string &str, const char delimiter);

template <typename... Args>
std::string sformat(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  char *buf = new char[size];
  std::snprintf(buf, size, format.c_str(), args...);
  return std::string(buf, buf + size - 1);
}

static const char *ws = " \t\n\r\f\v";

inline std::string &ltrim_(std::string &s) {
  return s.erase(0, s.find_first_not_of(ws));
}

inline std::string &rtrim_(std::string &s) {
  return s.erase(s.find_last_not_of(ws) + 1);
}

inline std::string &trim_(std::string &s) { return ltrim_(rtrim_(s)); }

} // namespace INNC
