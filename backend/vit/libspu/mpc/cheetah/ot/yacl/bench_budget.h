#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
namespace spu::mpc::cheetah {
class BenchBudget {
 bool active_=false;uint64_t cap_=0,used_=0;
 public:
 void arm(uint64_t n){active_=true;cap_=n;used_=0;}
 void disarm(){active_=false;}
 void consume(uint64_t n){if(!active_)return;if(n>cap_-used_)throw std::runtime_error("online preprocessing stock exhausted; refill forbidden: request="+std::to_string(n)+",remaining="+std::to_string(cap_-used_)+",capacity="+std::to_string(cap_));used_+=n;}
 bool active()const{return active_;}
 uint64_t used()const{return used_;}
 uint64_t remaining()const{return cap_-used_;}
};
}
