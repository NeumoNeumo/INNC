#pragma once

#include <cassert>
#include <exception>
#include <iostream>
// TODO 3 More exceptions
#define assertm(exp, msg) assert(((void)(msg), exp))
#define run_expect(exp, msg)                                                   \
  if (!(exp)) {                                                                \
    std::cerr << msg << std::endl;                                             \
    abort();                                                                   \
  }
namespace INNC {}
