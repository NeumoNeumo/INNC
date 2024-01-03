#pragma once

#include <cassert>
#include <exception>
#include <iostream>
// TODO 3 More exceptions
#define assertm(exp, msg) assert(((void)(msg), exp))
#define run_expect(exp, msg)                                                   \
  if (!(exp)) {                                                                \
    throw std::runtime_error(msg);                                             \
  }
namespace INNC {}
