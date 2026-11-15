#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace spu::vit_resource {
struct Budget {
  uint64_t beaver64;
  uint64_t beaver32;

  std::vector<std::array<uint64_t,2>> lanes;
  uint64_t resident_bytes() const {
    uint64_t n=24*beaver64+12*beaver32;
    for(const auto& q:lanes)n+=32*q[0]+49*q[1];
    return n;
  }
};
inline Budget Parse(const std::vector<uint64_t>& q,size_t lanes) {
  if(lanes==0||lanes>48||q.size()!=2+2*lanes)
    throw std::invalid_argument("lane stock geometry mismatch");
  for(auto v:q)if(v>=(uint64_t{1}<<32))
    throw std::invalid_argument("lane stock item count outside bound");
  Budget out{q[0],q[1],{}};
  for(size_t i=0;i<lanes;++i)out.lanes.push_back({q[2+2*i],q[3+2*i]});
  return out;
}
}
