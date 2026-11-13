// Copyright 2026

#pragma once

#include <cstdint>
#include <array>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/value.h"
#include "libspu/mpc/kernel.h"
#include "libspu/mpc/cheetah/ot/material_inventory.h"

namespace spu::mpc::cheetah {

class CdiTimingState final : public State {
 public:
  static constexpr char kBindName[] = "CdiTiming";
  void Observe(const std::string& scope, int64_t site, const std::string& action, SPUContext* ctx=nullptr);
  bool InInference() const { return !stack_.empty() && stack_.front().scope=="inference"; }
  int64_t NextGelu(){return next_gelu_++;}
  int64_t NextEqual(){return next_equal_++;}
  void ArmMaterials(int64_t epoch) { SPU_ENFORCE(epoch>=0 && stack_.empty() && !material_epochs_.count(epoch),"invalid/reused material preparation epoch");material_epochs_.insert(epoch);material_epoch_=epoch;material_consumed_=false; }
  void ReleaseMaterials(int64_t epoch,SPUContext* ctx);
  std::vector<std::array<uint64_t,2>> LastRequirements(){std::lock_guard guard(mutex_);return last_requirements_;}
  std::array<uint64_t,2> LastBeaverRequirements(){std::lock_guard guard(mutex_);return last_beaver_requirements_;}
 private:
  struct Entry {
    std::string scope;
    int64_t site;
    std::chrono::steady_clock::time_point start;
    int64_t start_ns;
    std::vector<MaterialUsage> resources;
    std::array<MaterialUsage,2> beavers;
  };
  std::mutex mutex_;
  std::vector<Entry> stack_;
  std::vector<std::array<uint64_t,2>> last_requirements_;
  std::array<uint64_t,2> last_beaver_requirements_{};
  int64_t next_gelu_=0,next_equal_=0,material_epoch_=-1;
  bool material_consumed_=false;
  std::unordered_set<int64_t> material_epochs_;
};

class CdiScope final : public Kernel {
 public:
  static constexpr char kBindName[] = "cdi_scope";
  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

class CdiMaterialPrepare final : public Kernel {
 public:
  static constexpr char kBindName[] = "cdi_material_prepare";
  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

}
