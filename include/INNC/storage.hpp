#pragma once

#include <memory>

class UntypedStorage : public std::shared_ptr<uint8_t> {
  size_t size; // in unit of bypes
public:
  void create(size_t size);
  size_t get_size() const noexcept;
};
