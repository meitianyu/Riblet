#include "libspu/core/cpp_prepare_observer.h"

// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Modified for Riblet; see NOTICE. */

#pragma once

#include <memory>
#include <mutex>
#include <map>
#include "libspu/mpc/cheetah/ot/yacl/bench_budget.h"

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"
#include "libspu/mpc/cheetah/arith/cheetah_dot.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetah {

class CheetahMulState : public State {
 private:
  mutable std::mutex lock_;
  // a[2] = a[0] * a[1]
  mutable int64_t cached_sze_{0};
  struct RingStock {
    BenchBudget budget;
    std::array<NdArrayRef,3> shares;
    int64_t available=0;
  };
  std::map<FieldType,RingStock> prepared_;
  bool stock_required_=false;
  FieldType field_{FT_INVALID};
  NdArrayRef cached_beaver_[3];

  std::unique_ptr<CheetahMul> mul_prot_;
  std::shared_ptr<yacl::link::Context> duplx_;

  // NOTE(juhou): make sure the lock is obtained
  void makeSureCacheSize(FieldType, int64_t numel);

  explicit CheetahMulState(std::unique_ptr<CheetahMul> mul_prot)
      : mul_prot_(std::move(mul_prot)) {}

 public:
  static constexpr char kBindName[] = "CheetahMul";

  explicit CheetahMulState(const std::shared_ptr<yacl::link::Context>& lctx,
                           bool enable_mul_lsb_error = false) {
    mul_prot_ = std::make_unique<CheetahMul>(lctx, enable_mul_lsb_error);
    duplx_ = lctx->Spawn();
  }

  ~CheetahMulState() override = default;

  CheetahMul* get() { return mul_prot_.get(); }
  void BenchPrepare(FieldType field,int64_t n) {

spu::lifecycle::Scope scope_beaver_reserve_f32((field == FM32 ? "beaver.reserve.f32" : "beaver.reserve.f64"), "unclassified");
scope_beaver_reserve_f32.Outcome("prepared");

    std::unique_lock guard(lock_);
    SPU_ENFORCE((field==FM32||field==FM64)&&n>=0&&!prepared_.count(field),"invalid/repeated ring reservation");
    cached_sze_=0;field_=field;
    for(auto& v:cached_beaver_)v=ring_zeros(field,{0});
    if(n)makeSureCacheSize(field,n);
    RingStock pool;pool.available=cached_sze_;pool.budget.arm(n);
    for(size_t i=0;i<3;++i){pool.shares[i]=std::move(cached_beaver_[i]);cached_beaver_[i]=ring_zeros(field,{0});}
    prepared_.emplace(field,std::move(pool));cached_sze_=0;stock_required_=true;
  }
  void BenchRelease() {

spu::lifecycle::Scope scope_beaver_pool_release("beaver.pool-release", "A");
scope_beaver_pool_release.Outcome("retired");

    std::unique_lock guard(lock_);prepared_.clear();cached_sze_=0;
    if(field_!=FT_INVALID)for(auto& v:cached_beaver_)v=ring_zeros(field_,{0});

    stock_required_=true;
  }
  bool BenchArmed() const { return !prepared_.empty(); }
  uint64_t BenchUsed() const { uint64_t n=0;for(const auto& [f,p]:prepared_)n+=p.budget.used();return n; }
  uint64_t BenchRemaining() const { uint64_t n=0;for(const auto& [f,p]:prepared_)n+=p.budget.remaining();return n; }
  uint64_t BenchRemaining(FieldType field) const { auto p=prepared_.find(field);return p==prepared_.end()?0:p->second.budget.remaining(); }

  std::shared_ptr<yacl::link::Context> duplx() { return duplx_; }

  std::array<NdArrayRef, 3> TakeCachedBeaver(FieldType field, int64_t num);
};

class CheetahDotState : public State {
 private:
  std::unique_ptr<CheetahDot> dot_prot_;

  explicit CheetahDotState(std::unique_ptr<CheetahDot> dot_prot)
      : dot_prot_(std::move(dot_prot)) {}

 public:
  static constexpr char kBindName[] = "CheetahDot";

  explicit CheetahDotState(const std::shared_ptr<yacl::link::Context>& lctx,
                           bool disable_matmul_pack = false) {
    dot_prot_ = std::make_unique<CheetahDot>(lctx, disable_matmul_pack);
  }

  ~CheetahDotState() override = default;

  CheetahDot* get() { return dot_prot_.get(); }
};

class CheetahOTState : public State {
 private:
  using ProtPtr = std::shared_ptr<BasicOTProtocols>;

  mutable std::mutex lock_;

  static constexpr size_t kMaxOTParallel = 48;

  size_t maximum_instances_ = 0;
  std::vector<ProtPtr> basic_ot_prot_;
  CheetahOtKind ot_kind_;
  bool strict_inventory_=false;

 public:
  static constexpr char kBindName[] = "CheetahOT";

  explicit CheetahOTState(size_t maximum_instances, CheetahOtKind ot_kind)
      : maximum_instances_(std::min(kMaxOTParallel, maximum_instances)),
        basic_ot_prot_(maximum_instances_),
        ot_kind_(ot_kind) {
    SPU_ENFORCE(maximum_instances_ > 0);
    std::string ot_type;
    switch (ot_kind_) {
      default:
      case CheetahOtKind::YACL_Ferret:
        ot_type = "yacl_ferret";
        break;
      case CheetahOtKind::EMP_Ferret:
        ot_type = "emp_ferret";
        break;
      case CheetahOtKind::YACL_Softspoken:
        ot_type = "yacl_softspoken";
        break;
    }
    SPDLOG_DEBUG("CHEETAH: Uses {} OT", ot_type);
  }

  ~CheetahOTState() override = default;

  size_t maximum_instances() const { return maximum_instances_; }
  void SetStrictInventory(bool v) { strict_inventory_=v; }

  void LazyInit(Communicator* comm, size_t idx = 0) {

spu::lifecycle::Scope scope_ot_lane_initialize("OT.lane-initialize", "R-S", [&] { return static_cast<uint64_t>((comm->lctx())->GetStats()->sent_bytes); }, "yacl-shared-inclusive");

    SPU_ENFORCE(idx < maximum_instances_, "idx={} out-of-bound", idx);
    std::lock_guard guard(lock_);
    if (basic_ot_prot_[idx]) {

scope_ot_lane_initialize.Outcome("cache-hit");

      return;
    }
    SPU_ENFORCE(!strict_inventory_,"online OT lane was not prepared");
    // NOTE(lwj): create a separated link for OT
    // We **do not** block on the OT link since the message volume is small for
    // LPN-based OTe
    auto link = comm->lctx()->Spawn();
    link->SetThrottleWindowSize(0);
    auto _comm = std::make_shared<Communicator>(std::move(link));
    basic_ot_prot_[idx] =
        std::make_shared<BasicOTProtocols>(std::move(_comm), ot_kind_);
  }

  std::shared_ptr<BasicOTProtocols> get(size_t idx = 0) {
    SPU_ENFORCE(idx < maximum_instances_, "idx={} out-of-bound", idx);
    SPU_ENFORCE(basic_ot_prot_[idx], "call LazyInit first");
    return basic_ot_prot_[idx];
  }
};

}  // namespace spu::mpc::cheetah
