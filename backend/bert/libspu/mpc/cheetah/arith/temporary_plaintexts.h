#pragma once

#include <cstddef>
#include <vector>
#include "seal/plaintext.h"

namespace spu::mpc::cheetah {

inline std::vector<seal::Plaintext> MakeTemporaryPlaintexts(std::size_t count) {
  auto pool=seal::MemoryPoolHandle::New(true);
  std::vector<seal::Plaintext> values;
  values.reserve(count);
  for(std::size_t i=0;i<count;++i)values.emplace_back(pool);
  return values;
}

}
