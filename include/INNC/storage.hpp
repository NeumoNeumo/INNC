#pragma once

#include <memory>

class UntypedStorage {
  size_t size; // in unit of bypes
  std::unique_ptr<uint8_t[]> blob;

public:
  UntypedStorage(size_t size, bool prealloc = true);
  void alloc();
  size_t get_size() const noexcept;
  void zero_() const noexcept;
  uint8_t *get_blob() const noexcept;
  void reset_blob(uint8_t ptr[]);
  void reset_blob(std::unique_ptr<uint8_t[]> &&ptr);
  void release() noexcept;
  bool is_alloc() const noexcept;
};
