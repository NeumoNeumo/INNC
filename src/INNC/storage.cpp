#include "INNC/storage.hpp"

void UntypedStorage::create(size_t size) {
  this->size = size;
  this->reset(new uint8_t[size]);
}

size_t UntypedStorage::get_size() const noexcept { return this->size; }
