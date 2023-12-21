#pragma once

#include <cassert>
#include <exception>
#include <iostream>
#define assertm(exp, msg) assert(((void)(msg), exp))
#define run_expect(exp, msg)                                                   \
  if (!(exp)) {                                                                \
    std::cerr << msg << std::endl;                                             \
    abort();                                                                   \
  }
namespace INNC {}
