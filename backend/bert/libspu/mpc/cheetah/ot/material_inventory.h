#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <set>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/falloc.h>

namespace spu::mpc::cheetah {
inline int64_t MaterialNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct GenerationInterval { int64_t begin, end; };
struct MaterialUsage {
  uint64_t requested=0, stock_served=0, stock_remaining=0, generated=0;
  uint64_t stock_capacity=0;
  uint64_t stock_resident_bytes=0,stock_disk_bytes=0;
  uint64_t rmrc1_requested=0,rmrc2_requested=0;
  uint64_t rmrc1_served=0,rmrc2_served=0,rmrc_stock_remaining=0;
  std::vector<GenerationInterval> generations;
};
inline int64_t GenerationUnionNs(const std::vector<GenerationInterval>& input,
                                 int64_t begin,int64_t end) {
  std::vector<GenerationInterval> spans;
  for(auto span:input) {
    span.begin=std::max(span.begin,begin);span.end=std::min(span.end,end);
    if(span.begin<span.end)spans.push_back(span);
  }
  std::sort(spans.begin(),spans.end(),[](auto a,auto b){return a.begin<b.begin;});
  int64_t total=0,position=begin;
  for(auto span:spans){total+=std::max(int64_t{0},span.end-std::max(position,span.begin));position=std::max(position,span.end);}
  return total;
}

class InventoryStoragePolicy {
 public:
  static InventoryStoragePolicy& Get(){static InventoryStoragePolicy value;return value;}
  bool Bounded() const{return bounded_;}
  uint64_t Limit() const{return limit_;}
  size_t Chunk() const{return chunk_;}
  bool Acquire(uint64_t bytes){std::lock_guard guard(lock_);if(bytes>limit_-used_)return false;used_+=bytes;return true;}
  void Return(uint64_t bytes){std::lock_guard guard(lock_);used_-=bytes;}
  void Admit(uint64_t ot_bytes,uint64_t largest_lane,uint64_t beaver_bytes) const {
    if(!bounded_)return;
    std::ifstream input("/proc/meminfo");std::string line;uint64_t available=0;
    while(std::getline(input,line))if(line.rfind("MemAvailable:",0)==0)available=std::stoull(line.substr(13))*1024;
    const auto resident=std::min(ot_bytes,limit_);
    if(available<resident+2*largest_lane+4*beaver_bytes+(8ULL<<30))
      throw std::runtime_error("bounded inventory RAM admission failed including generator transient");
    uint64_t disk=0;std::set<dev_t> seen;
    for(const auto& dir:dirs_){struct stat st{};struct statvfs fs{};
      if(stat(dir.c_str(),&st)==0 && statvfs(dir.c_str(),&fs)==0 && seen.insert(st.st_dev).second){
        const auto free=uint64_t(fs.f_bavail)*fs.f_frsize;
        if(free>headroom_)disk+=free-headroom_;
      }
    }
    const auto need=ot_bytes>limit_?ot_bytes-limit_+chunk_:0;
    if(disk<need)throw std::runtime_error("bounded inventory disk admission failed");
  }
  int Open(uint64_t maximum_bytes) const {
    for(const auto& dir:dirs_){struct statvfs fs{};
      if(statvfs(dir.c_str(),&fs)!=0)continue;
      if(uint64_t(fs.f_bavail)*fs.f_frsize<maximum_bytes+headroom_)continue;
      std::string name=dir+"/cdi-stock-XXXXXX";
      std::vector<char> path(name.begin(),name.end());path.push_back('\0');
      int fd=mkstemp(path.data());if(fd<0)continue;
      if(fchmod(fd,0600)!=0 || fcntl(fd,F_SETFD,FD_CLOEXEC)!=0){close(fd);unlink(path.data());continue;}
      if(unlink(path.data())!=0){close(fd);throw std::runtime_error("cannot unlink private stock");}
      return fd;
    }
    throw std::runtime_error("no private stock directory with sufficient headroom");
  }
 private:
  InventoryStoragePolicy(){
    if(const char* bytes=std::getenv("CDI_MATERIAL_RAM_BYTES")){
      bounded_=true;limit_=std::stoull(bytes);
      if(const char* chunk=std::getenv("CDI_MATERIAL_CHUNK_BYTES"))chunk_=std::stoull(chunk);
      if(!limit_ || !chunk_ || chunk_>(64ULL<<20))throw std::runtime_error("invalid inventory storage budget");
      const char* raw=std::getenv("CDI_MATERIAL_SPILL_DIRS");
      if(!raw || !*raw)throw std::runtime_error("bounded inventory requires explicit private spill directories");
      std::string dirs(raw);size_t begin=0;
      while(begin<dirs.size()){auto end=dirs.find(':',begin);auto dir=dirs.substr(begin,end-begin);
        struct stat st{};if(dir.empty() || dir[0]!='/' || stat(dir.c_str(),&st)!=0 || !S_ISDIR(st.st_mode))
          throw std::runtime_error("invalid private stock directory");
        dirs_.push_back(dir);if(end==std::string::npos)break;begin=end+1;
      }
    }
  }
  bool bounded_=false;
  uint64_t limit_=std::numeric_limits<uint64_t>::max(),used_=0;
  size_t chunk_=64ULL<<20;
  const uint64_t headroom_=4ULL<<30;
  std::vector<std::string> dirs_;
  std::mutex lock_;
};

template<class T> class OneUseInventory {
 public:
  OneUseInventory()=default;
  OneUseInventory(const OneUseInventory&)=delete;
  OneUseInventory& operator=(const OneUseInventory&)=delete;
  ~OneUseInventory(){Release();}
  void Install(std::vector<T> values) {
    if(armed_)throw std::runtime_error("material inventory is already armed");
    auto& policy=InventoryStoragePolicy::Get();
    total_=values.size();cursor_=0;
    try {
      if(!policy.Bounded()){
        if(!policy.Acquire(values.size()*sizeof(T)))throw std::runtime_error("inventory size overflow");
        resident_=values.size()*sizeof(T);
        blocks_.push_back({std::move(values),0,total_,0,false});
      }else{
        const size_t chunk=std::max(size_t{1},policy.Chunk()/sizeof(T));
        for(size_t offset=0;offset<values.size();){
          const auto count=std::min(chunk,values.size()-offset);const auto bytes=count*sizeof(T);
          if(policy.Acquire(bytes)){
            resident_+=bytes;
            blocks_.push_back({std::vector<T>(values.begin()+offset,values.begin()+offset+count),0,count,0,false});
          }else{
            if(fd_<0)fd_=policy.Open(values.size()*sizeof(T));
            Transfer(false,values.data()+offset,bytes,file_end_);
            blocks_.push_back({{},file_end_,count,0,true});file_end_+=bytes;disk_+=bytes;
          }
          offset+=count;
        }
        if(fd_>=0){
          if(fdatasync(fd_)!=0)throw std::runtime_error("private stock flush failed");
          posix_fadvise(fd_,0,0,POSIX_FADV_DONTNEED);
        }
      }
      armed_=true;
    }catch(...){Release();throw;}
  }
  void Take(T* output,size_t count) {
    if(!armed_ || count>Remaining())throw std::runtime_error("prepared material inventory exhausted");
    size_t done=0;
    while(done<count){
      auto& block=blocks_.front();const auto n=std::min(count-done,block.size-block.read);
      if(block.disk)Transfer(true,output+done,n*sizeof(T),block.offset+block.read*sizeof(T));
      else std::copy_n(block.values.data()+block.read,n,output+done);
      block.read+=n;cursor_+=n;done+=n;
      if(block.read==block.size){
        const auto bytes=block.size*sizeof(T);
        if(block.disk){

          if(fallocate(fd_,FALLOC_FL_PUNCH_HOLE|FALLOC_FL_KEEP_SIZE,block.offset,bytes)!=0)
            throw std::runtime_error("private stock consumed-block release failed");
          posix_fadvise(fd_,block.offset,bytes,POSIX_FADV_DONTNEED);disk_-=bytes;
        }else{InventoryStoragePolicy::Get().Return(bytes);resident_-=bytes;}
        blocks_.pop_front();
      }
    }
  }
  size_t Remaining() const{return total_-cursor_;}
  bool Armed() const{return armed_;}
  size_t Capacity() const{return total_;}
  size_t ResidentBytes() const{return resident_;}
  size_t DiskBytes() const{return disk_;}
  void Release(){
    if(resident_)InventoryStoragePolicy::Get().Return(resident_);
    blocks_.clear();if(fd_>=0)close(fd_);
    fd_=-1;resident_=disk_=file_end_=total_=cursor_=0;armed_=false;
  }
 private:
  void Transfer(bool read,void* buffer,size_t bytes,uint64_t offset){
    size_t done=0;auto* data=static_cast<char*>(buffer);
    while(done<bytes){
      auto n=read?pread(fd_,data+done,bytes-done,offset+done):pwrite(fd_,data+done,bytes-done,offset+done);
      if(n<0 && errno==EINTR)continue;
      if(n<=0)throw std::runtime_error(std::string("private stock I/O failed: ")+strerror(errno));
      done+=size_t(n);
    }
  }
  struct Block{std::vector<T> values;uint64_t offset;size_t size,read;bool disk;};
  std::deque<Block> blocks_;
  int fd_=-1;
  size_t cursor_=0,total_=0,resident_=0,disk_=0;
  uint64_t file_end_=0;
  bool armed_=false;
};
}
