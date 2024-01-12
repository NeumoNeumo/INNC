#pragma once

#include <cassert>
#include <sstream>
#define assertm(exp, msg) assert(((void)(msg), exp))

template <typename... Args> void run_expect(bool expr, const Args &...args) {
  if (expr)
    return;
  std::stringstream ss;
  (ss << ... << args);
  throw std::runtime_error(ss.str());
}
