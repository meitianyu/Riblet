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

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/object.h"
#include "libspu/mpc/cheetah/arith/cheetah_dot.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"

#include "libspu/spu.pb.h"

namespace spu::mpc::cheetah {

class CheetahMulState : public State {
 private:
  mutable std::mutex lock_;
  // a[2] = a[0] * a[1]
  mutable int64_t cached_sze_{0};
  FieldType field_{FT_INVALID};
  NdArrayRef cached_beaver_[3];
  MaterialUsage usage_;
  bool strict_inventory_=false;
  struct PreparedPool {
    int64_t remaining=0;
    std::array<NdArrayRef,3> shares;
  };
  std::map<FieldType,PreparedPool> prepared_pools_;
  std::map<FieldType,MaterialUsage> field_usage_;

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

  std::shared_ptr<yacl::link::Context> duplx() { return duplx_; }

  std::array<NdArrayRef, 3> TakeCachedBeaver(FieldType field, int64_t num);
  void ReserveBeavers(FieldType field,int64_t count);
  void ReleasePreparedBeavers();
  MaterialUsage Resources() const {
    std::lock_guard guard(lock_);auto result=usage_;result.stock_remaining=cached_sze_;
    for(const auto& [field,pool]:prepared_pools_)result.stock_remaining+=pool.remaining;
    return result;
  }
  MaterialUsage Resources(FieldType field) const {
    std::lock_guard guard(lock_);
    auto it=field_usage_.find(field);
    auto result=it==field_usage_.end()?MaterialUsage{}:it->second;
    if(auto pool=prepared_pools_.find(field);pool!=prepared_pools_.end())result.stock_remaining=pool->second.remaining;
    else result.stock_remaining=field_==field?cached_sze_:0;
    return result;
  }
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
  std::shared_ptr<BasicOTProtocols> Existing(size_t idx) const {
    std::lock_guard guard(lock_);return basic_ot_prot_.at(idx);
  }
  void SetStrictInventory(bool value) { strict_inventory_=value; }

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
