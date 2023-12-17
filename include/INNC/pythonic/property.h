#pragma once
#include <string>

template <typename T> class Number {
protected:
  int n;
public:
  virtual T &operator+(const int num) {
    n += num;
    return *dynamic_cast<T *>(this);
  }
  virtual T &operator=(const int a) {
    n = a;
    return *dynamic_cast<T *>(this);
  }
  std::string str() const;
};
