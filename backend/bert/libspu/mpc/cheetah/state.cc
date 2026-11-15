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

#include "libspu/mpc/cheetah/state.h"

#include <future>

#include "spdlog/spdlog.h"

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

void CheetahMulState::makeSureCacheSize(FieldType field, int64_t numel) {

spu::lifecycle::Scope scope_beaver_generate_mixed("beaver.generate.mixed", "unclassified");

  // NOTE(juhou): make sure the lock is obtained
  SPU_ENFORCE(numel > 0);
  if (field_ != field) {
    // drop all previous cache
    cached_sze_ = 0;
  }

  if (cached_sze_ >= numel) {
    return;
  }
  SPU_ENFORCE(!strict_inventory_,"prepared Beaver inventory exhausted");
  const auto material_begin=MaterialNowNs();

  //  create one batch OLE which then converted to Beavers
  //  Math:
  //   Alice samples rand0 and views it as rand0 = a0||b0
  //   Bob samples rand1 and views it as rand1 = b1||a0
  //   The multiplication rand0 * rand1 gives two cross term a0*b1||a1*b0
  //   Then the beaver (a0, b0, c0) and (a1, b1, c1)
  //   where c0 = a0*b0 + <a0*b1> + <a1*b0>
  //         c1 = a1*b1 + <a0*b1> + <a1*b0>

spu::lifecycle::Scope scope_beaver_key_initialization("beaver.key-initialization", "R-S");

  mul_prot_->LazyInitKeys(field);

scope_beaver_key_initialization.Stop();

  const int rank = mul_prot_->Rank();
  const int64_t ole_sze = mul_prot_->OLEBatchSize();
  const int64_t num_ole = CeilDiv<size_t>(2 * numel, ole_sze);
  const int64_t num_beaver = (num_ole * ole_sze) / 2;

spu::lifecycle::Scope scope_beaver_random_masks("beaver.random-masks", "F");

  auto rand = ring_rand(field, {num_ole * ole_sze});

scope_beaver_random_masks.Stop();

spu::lifecycle::Scope scope_beaver_ole("beaver.OLE", "F");

  auto cross = mul_prot_->MulOLE(rand, rank == 0);

scope_beaver_ole.Stop();

spu::lifecycle::Scope scope_beaver_local_assemble("beaver.local-assemble", "A");

  NdArrayRef beaver[3];
  NdArrayRef a0b1;
  NdArrayRef a1b0;
  if (rank == 0) {
    beaver[0] = rand.slice({0}, {num_beaver}, {1});
    beaver[1] = rand.slice({num_beaver}, {num_beaver * 2}, {1});
  } else {
    beaver[0] = rand.slice({num_beaver}, {num_beaver * 2}, {1});
    beaver[1] = rand.slice({0}, {num_beaver}, {1});
  }
  a0b1 = cross.slice({0}, {num_beaver}, {1});
  a1b0 = cross.slice({num_beaver}, {num_beaver * 2}, {1});

  beaver[2] =
      ring_add(ring_add(cross.slice({0}, {num_beaver}, {1}),
                        cross.slice({num_beaver}, {2 * num_beaver}, {1})),
               ring_mul(beaver[0], beaver[1]));

scope_beaver_local_assemble.Stop();

spu::lifecycle::Scope scope_beaver_pool_allocation_copy("beaver.pool-allocation-copy", "A");

  DISPATCH_ALL_FIELDS(field, "makeSureCacheSize", [&]() {
    for (size_t i : {0, 1, 2}) {
      auto tmp = ring_zeros(field, {num_beaver + cached_sze_});
      NdArrayView<ring2k_t> new_cache(tmp);

      // concate two array
      if (cached_sze_ > 0) {
        NdArrayView<const ring2k_t> old_cache(cached_beaver_[i]);
        pforeach(0, cached_sze_,
                 [&](int64_t j) { new_cache[j] = old_cache[j]; });
      }

      NdArrayView<const ring2k_t> _beaver(beaver[i]);
      pforeach(0, num_beaver,
               [&](int64_t j) { new_cache[cached_sze_ + j] = _beaver[j]; });
      cached_beaver_[i] = tmp;
    }
  });

scope_beaver_pool_allocation_copy.Stop();

  field_ = field;
  cached_sze_ += num_beaver;
  usage_.generated+=num_beaver;
  const GenerationInterval interval{material_begin,MaterialNowNs()};
  usage_.generations.push_back(interval);
  field_usage_[field].generated+=num_beaver;
  field_usage_[field].generations.push_back(interval);

  SPU_ENFORCE(cached_sze_ >= numel);
}

std::array<NdArrayRef, 3> CheetahMulState::TakeCachedBeaver(FieldType field,
                                                            int64_t numel) {

spu::lifecycle::Scope scope_beaver_consume_f32((field == FM32 ? "beaver.consume.f32" : "beaver.consume.f64"), "F");
scope_beaver_consume_f32.Outcome("consumed");

  SPU_ENFORCE(numel > 0);
  std::unique_lock guard(lock_);
  usage_.requested+=numel;
  field_usage_[field].requested+=numel;
  if(strict_inventory_){
    auto it=prepared_pools_.find(field);
    SPU_ENFORCE(it!=prepared_pools_.end() && it->second.remaining>=numel,
                "prepared Beaver inventory exhausted for field {}",static_cast<int>(field));
    auto& pool=it->second;
    std::array<NdArrayRef,3> ret;
    for(size_t i=0;i<3;++i){
      ret[i]=pool.shares[i].slice({0},{numel},{1});
      pool.shares[i]=pool.remaining==numel
          ? NdArrayRef(pool.shares[i].eltype(),{0})
          : pool.shares[i].slice({numel},{pool.remaining},{1});
    }
    pool.remaining-=numel;
    usage_.stock_served+=numel;field_usage_[field].stock_served+=numel;
    return ret;
  }
  makeSureCacheSize(field, numel);

  std::array<NdArrayRef, 3> ret;
  for (size_t i : {0, 1, 2}) {
    SPU_ENFORCE(cached_beaver_[i].numel() >= numel);
    ret[i] = cached_beaver_[i].slice({0}, {numel}, {1});
    if (cached_sze_ == numel) {
      // empty cache now
      // NOTE(lwj): should use `{0}` as empty while `{}` is a scalar
      cached_beaver_[i] = NdArrayRef(cached_beaver_[i].eltype(), {0});
    } else {
      cached_beaver_[i] = cached_beaver_[i].slice({numel}, {cached_sze_}, {1});
    }
  }

  cached_sze_ -= numel;
  return ret;
}

void CheetahMulState::ReserveBeavers(FieldType field,int64_t count) {

spu::lifecycle::Scope scope_beaver_reserve_f32((field == FM32 ? "beaver.reserve.f32" : "beaver.reserve.f64"), "unclassified");
scope_beaver_reserve_f32.Outcome("prepared");

  std::lock_guard guard(lock_);
  SPU_ENFORCE(count>=0,"negative material quota");
  SPU_ENFORCE(field==FM32 || field==FM64,"prepared ring must be FM32 or FM64");
  SPU_ENFORCE(!prepared_pools_.count(field),"release prepared ring before reserving it again");
  const bool was_strict=strict_inventory_;
  strict_inventory_=false;

  for(auto& buffer:cached_beaver_)buffer=ring_zeros(field,{0});
  cached_sze_=0;field_=field;
  try {
    if(count)makeSureCacheSize(field,count);
    PreparedPool pool;pool.remaining=cached_sze_;
    for(size_t i=0;i<3;++i){pool.shares[i]=std::move(cached_beaver_[i]);cached_beaver_[i]=ring_zeros(field,{0});}
    prepared_pools_.emplace(field,std::move(pool));cached_sze_=0;
    strict_inventory_=true;
  }catch(...){
    cached_sze_=0;for(auto& buffer:cached_beaver_)buffer=ring_zeros(field,{0});
    strict_inventory_=was_strict;throw;
  }
}

void CheetahMulState::ReleasePreparedBeavers() {

spu::lifecycle::Scope scope_beaver_pool_release("beaver.pool-release", "A");
scope_beaver_pool_release.Outcome("retired");

  std::lock_guard guard(lock_);
  prepared_pools_.clear();
  if(field_!=FT_INVALID)for(auto& buffer:cached_beaver_)buffer=ring_zeros(field_,{0});
  cached_sze_=0;strict_inventory_=false;
}

}  // namespace spu::mpc::cheetah
