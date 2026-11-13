#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace spu::mpc::cheetah {

inline std::vector<std::vector<int64_t>> ActiveSchedule(const std::vector<int64_t>& ks) {
  int64_t maximum=0;
  for(auto k:ks){if(k<0||k>8)throw std::invalid_argument("public K outside 0..8");maximum=std::max(maximum,k);}
  std::vector<std::vector<int64_t>> result(maximum);
  for(int64_t i=0;i<static_cast<int64_t>(ks.size());++i)
    for(int64_t round=0;round<ks[i];++round)result[round].push_back(i);
  return result;
}
}
