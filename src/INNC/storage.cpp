#include "INNC/storage.hpp"
#include <cstring>

void UntypedStorage::alloc() { blob.reset(new uint8_t[size]); }

UntypedStorage::UntypedStorage(size_t size, bool prealloc) {
  this->size = size;
  if (prealloc)
    blob.reset(new uint8_t[size]);
}

size_t UntypedStorage::get_size() const noexcept { return this->size; }

void UntypedStorage::zero_() const noexcept {
  if (blob.get() == nullptr)
    return;
  std::memset(blob.get(), 0, size);
}

uint8_t *UntypedStorage::get_blob() { return blob.get(); }

void UntypedStorage::release() noexcept { blob.reset(); }

bool UntypedStorage::is_alloc() const noexcept { return blob.get() != nullptr; }
