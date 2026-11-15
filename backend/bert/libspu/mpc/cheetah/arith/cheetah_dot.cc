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

#include "libspu/mpc/cheetah/arith/cheetah_dot.h"

#include <algorithm>
#include <future>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/types/span.h"
#include "seal/batchencoder.h"
#include "seal/context.h"
#include "seal/decryptor.h"
#include "seal/encryptor.h"
#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/keygenerator.h"
#include "seal/publickey.h"
#include "seal/secretkey.h"
#include "seal/util/locks.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "spdlog/spdlog.h"
#include "yacl/link/link.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/arith/matmat_prot.h"
#include "libspu/mpc/cheetah/arith/temporary_plaintexts.h"
#include "libspu/mpc/cheetah/rlwe/lwe_ct.h"
#include "libspu/mpc/cheetah/rlwe/modswitch_helper.h"
#include "libspu/mpc/cheetah/rlwe/packlwes.h"
#include "libspu/mpc/cheetah/rlwe/utils.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

enum class CipherPackingType {
  lwes,
  rlwes,
  none,
};

static std::string ToString(CipherPackingType type) {
  switch (type) {
    case CipherPackingType::rlwes:
      return "interleave";
    case CipherPackingType::lwes:
      return "pack_lwes";
    default:
    case CipherPackingType::none:
      return "none";
  }
}

static void NimbusNegacyclicRightShiftInplace(
    RLWECt& ciphertext, size_t shift, const seal::SEALContext& context) {
  if (shift == 0 || ciphertext.size() == 0) {
    return;
  }

  const auto context_data = context.get_context_data(ciphertext.parms_id());
  SPU_ENFORCE(context_data != nullptr, "invalid Nimbus COP ciphertext");
  SPU_ENFORCE(!ciphertext.is_ntt_form(),
              "Nimbus COP packing requires non-NTT ciphertexts");

  const size_t degree = ciphertext.poly_modulus_degree();
  SPU_ENFORCE(shift < degree);
  std::vector<uint64_t> wrapped(shift);
  const auto& modulus = context_data->parms().coeff_modulus();
  for (size_t component = 0; component < ciphertext.size(); ++component) {
    uint64_t* coefficients = ciphertext.data(component);
    for (const auto& prime : modulus) {
      std::copy_n(coefficients + degree - shift, shift, wrapped.data());
      for (size_t index = degree - shift; index > 0; --index) {
        coefficients[index - 1 + shift] = coefficients[index - 1];
      }
      for (size_t index = 0; index < shift; ++index) {
        coefficients[index] = seal::util::negate_uint_mod(wrapped[index], prime);
      }
      coefficients += degree;
    }
  }
}

struct CheetahDot::Impl : public EnableCPRNG {
 public:
  const bool kUseModDownOptimization = true;
  static constexpr size_t kCtAsyncParallel = 16;

  explicit Impl(std::shared_ptr<yacl::link::Context> lctx,
                bool disable_matmul_pack)
      : lctx_(std::move(lctx)), disable_pack_(disable_matmul_pack) {}

  ~Impl() = default;

  // Compute C = A*B where |A|=dims[0]xdims[1], |B|=dims[1]xdims[2
  NdArrayRef DotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                    const Shape3D &dim3, bool is_self_lhs);

  // dim4 = [B, M, K, L]
  // LHS.shape BxMxK, RHS.shape BxKxL
  // Out.shape BxMxL
  NdArrayRef BatchDotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                         const Shape4D &dim4, bool is_self_lhs);

  NdArrayRef doDotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                      const Shape3D &dim3, bool is_self_lhs);

  NdArrayRef doBatchDotOLE(const NdArrayRef &prv_mat, yacl::link::Context *conn,
                           const Shape4D &dim4, bool is_self_lhs);

  void NimbusCopSetup(const NdArrayRef& lhs, const Shape3D& dims,
                      const std::string& plan_sha256, int64_t model_epoch,
                      int64_t family_slot);

  NdArrayRef NimbusCopOnline(const NdArrayRef& rhs, const Shape3D& dims,
                             const std::string& plan_sha256,
                             int64_t model_epoch, int64_t request_epoch,
                             int64_t family_slot);

  void NimbusCopRequestEnd(const std::string& plan_sha256,
                           int64_t model_epoch, int64_t request_epoch,
                           int64_t family_slot);

  void NimbusCopRelease(const std::string& plan_sha256, int64_t model_epoch,
                        int64_t family_slot);

  void doDotOLESenderSendStep(const NdArrayRef &prv_mat, const Shape3D &dim3,
                              bool is_self_lhs, CipherPackingType cptype,
                              yacl::link::Context *conn);

  void doDotOLEReceiverRecvStep(const NdArrayRef &prv_mat, const Shape3D &dim3,
                                bool is_self_lhs, CipherPackingType cptype,
                                absl::Span<RLWECt> result_cts,
                                yacl::link::Context *conn);

  NdArrayRef doDotOLESenderRecvStep(FieldType field, size_t batch_size,
                                    MatMatProtocol::Meta meta,
                                    size_t num_ct_to_recv,
                                    CipherPackingType cptype,
                                    yacl::link::Context *conn);

  NdArrayRef doDotOLEReceiverSendStep(FieldType field, size_t batch_size,
                                      MatMatProtocol::Meta meta,
                                      absl::Span<RLWECt> ct_array_to_pack,
                                      CipherPackingType cptype,
                                      yacl::link::Context *conn,
                                      size_t bytes_recv);

  bool IsPackingEnabled(uint32_t ring_bitlen) const {
    return DecideSEALParameters(ring_bitlen).use_special_prime();
  }

  seal::EncryptionParameters DecideSEALParameters(uint32_t ring_bitlen) const {
    auto scheme_type = seal::scheme_type::ckks;
    auto parms = seal::EncryptionParameters(scheme_type);

    size_t poly_deg;
    std::vector<int> modulus_bits;
    // NOTE(lwj): we need Q=sum(modulus_bits) > 2*k for multiplying two
    // k-bit elements.
    // 1. We need the (N, Q) pair satisifies the security.
    //    Check `seal/util/globals.cpp` for the recommendation HE parameters.
    // 2. We prefer modulus_bits[i] to be around 49-bit aiming to use AVX512
    // for acceleration if avaiable.
    // 3. We need some bits for margin. That is Q > 2*k + margin for errorless
    // w.h.p. We set margin=32bit
    if (ring_bitlen <= 32) {
      poly_deg = 4096;
      // ~ 64 + 32 bit
      modulus_bits = {59, 37};
      parms.set_use_special_prime(false);
    } else if (ring_bitlen <= 64) {
      poly_deg = 8192;
      //  ~ 128 + 32 bit
      modulus_bits = {59, 55, 49, 55};
      parms.set_use_special_prime(true);
    } else {
      poly_deg = 16384;
      // ~ 256 + 30 bit
      modulus_bits = {59, 59, 59, 59, 49, 55};
      parms.set_use_special_prime(true);
    }

    parms.set_poly_modulus_degree(poly_deg);
    parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_deg, modulus_bits));
    return parms;
  }

  void LazyInit(size_t field_bitlen, bool need_galois_key = false);

  void LazyInitGaloisKey(size_t field_bitlen);

  // Enc(Delta*m) -> Enc(Delta*m - r), r where r is sampled from Rq
  // One share of `m` is round(r/Delta) mod t
  // The other share of `m` is Dec(Enc(Delta*m - r))
  void H2A(absl::Span<RLWECt> ct, absl::Span<RLWEPt> rnd_mask,
           size_t target_modulus_size, const seal::PublicKey &pk,
           const seal::SEALContext &context) {
    seal::Evaluator evaluator(context);
    size_t num_poly = ct.size();
    SPU_ENFORCE(num_poly > 0);
    SPU_ENFORCE_EQ(rnd_mask.size(), num_poly);

    constexpr int64_t heuristic_group = 4;
    yacl::parallel_for(
        0, num_poly, heuristic_group, [&](size_t bgn, size_t end) {
          RLWECt zero_ct;
          for (size_t idx = bgn; idx < end; ++idx) {
            // NOTE(lwj): we hope the final ct is in the non-ntt form
            // We perform the intt before the modulus down which is faster
            // than modulus down then intt.
            InvNttInplace(ct[idx], context);

            ModulusSwtichInplace(ct[idx], target_modulus_size, context);

            // TODO(lwj): improve the performance of pk encryption of zero.
            // ct <- ct + enc(0)
            if (0 == zero_ct.size()) {
              seal::util::encrypt_zero_asymmetric(
                  pk, context, ct[idx].parms_id(), ct[idx].is_ntt_form(),
                  zero_ct);
            }

            evaluator.add_inplace(ct[idx], zero_ct);
            SPU_ENFORCE(!ct[idx].is_ntt_form());

            // sample r <- Rq
            // (ct[0] - r, ct[1]) <- ct
            UniformPoly(context, &rnd_mask[idx], ct[idx].parms_id());
            SubPlainInplace(ct[idx], rnd_mask[idx], context);
          }
        });
  }

 private:
  struct NimbusCopCacheKey {
    std::string plan_sha256;
    int64_t model_epoch;
    int64_t family_slot;

    bool operator==(const NimbusCopCacheKey& other) const {
      return plan_sha256 == other.plan_sha256 &&
             model_epoch == other.model_epoch &&
             family_slot == other.family_slot;
    }
  };

  struct NimbusCopCacheKeyHash {
    size_t operator()(const NimbusCopCacheKey& key) const {
      size_t value = std::hash<std::string>{}(key.plan_sha256);
      const auto combine = [&value](size_t part) {
        value ^= part + 0x9e3779b9 + (value << 6) + (value >> 2);
      };
      combine(std::hash<int64_t>{}(key.model_epoch));
      combine(std::hash<int64_t>{}(key.family_slot));
      return value;
    }
  };

  struct NimbusCopCacheEntry {
    FieldType field = FT_INVALID;
    Shape3D dims{};
    NdArrayRef clear_lhs;
    std::vector<RLWECt> encrypted_lhs;
    std::unordered_set<int64_t> used_request_epochs;
    std::unordered_set<int64_t> live_request_epochs;
    std::mutex request_mutex;
  };

  std::shared_ptr<yacl::link::Context> lctx_;
  bool disable_pack_ = false;
  std::unordered_map<NimbusCopCacheKey, std::shared_ptr<NimbusCopCacheEntry>,
                     NimbusCopCacheKeyHash>
      nimbus_cop_cache_;
  std::mutex nimbus_cop_cache_mutex_;

  // field_bitlen -> functor mapping
  std::unordered_map<size_t, std::shared_ptr<seal::SEALContext>> seal_cntxts_;
  std::unordered_map<size_t, std::shared_ptr<seal::SecretKey>> secret_keys_;
  std::unordered_map<size_t, std::shared_ptr<seal::PublicKey>> peer_pub_keys_;
  std::unordered_map<size_t, std::shared_ptr<seal::GaloisKeys>>
      peer_galois_keys_;
  // ModulusSwitchHelper for encoding
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> ecd_mswh_;
  // ModulusSwitchHelper for decoding
  std::unordered_map<size_t, std::shared_ptr<ModulusSwitchHelper>> dcd_mswh_;
  std::unordered_map<size_t, std::shared_ptr<seal::Decryptor>> decryptors_;
};

void CheetahDot::Impl::LazyInitGaloisKey(size_t field_bitlen) {

spu::lifecycle::Scope scope_he_dot_galois_key("HE.dot-galois-key", "R-S", [&] { return static_cast<uint64_t>((lctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");

  if (peer_galois_keys_.find(field_bitlen) != peer_galois_keys_.end()) {
    return;
  }
  auto kv = seal_cntxts_.find(field_bitlen);
  SPU_ENFORCE(kv != seal_cntxts_.end());
  const auto &this_context = *kv->second;
  const auto &this_rlwe_sk = *secret_keys_.find(field_bitlen)->second;

  seal::GaloisKeys gk;
  GenerateGaloisKeyForPacking(this_context, this_rlwe_sk,
                              /*seed*/ true, &gk);

  auto gk_buf = EncodeSEALObject(gk);
  int nxt_rank = lctx_->NextRank();
  std::shared_ptr<seal::GaloisKeys> peer_galois_key;
  if (nxt_rank == 0) {
    lctx_->Send(nxt_rank, gk_buf, "Rank0 send galois key");
    auto recv_gk = lctx_->Recv(nxt_rank, "Rank0 recv galois key");
    peer_galois_key = std::make_shared<seal::GaloisKeys>();
    DecodeSEALObject(recv_gk, this_context, peer_galois_key.get());
  } else {
    auto recv_gk = lctx_->Recv(nxt_rank, "Rank1 recv galois key");
    lctx_->Send(nxt_rank, gk_buf, "Rank0 send galois key");
    peer_galois_key = std::make_shared<seal::GaloisKeys>();
    DecodeSEALObject(recv_gk, this_context, peer_galois_key.get());
  }
  peer_galois_keys_.emplace(field_bitlen, peer_galois_key);
}

void CheetahDot::Impl::LazyInit(size_t field_bitlen, bool need_galois_keys) {

spu::lifecycle::Scope scope_he_dot_context_key("HE.dot-context-key", "R-S", [&] { return static_cast<uint64_t>((lctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");

if (spu::lifecycle::Global().enabled.load()) scope_he_dot_context_key.Outcome((seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) ? "cache-hit" : "initialized");

  if (seal_cntxts_.find(field_bitlen) != seal_cntxts_.end()) {
    if (need_galois_keys) {
      LazyInitGaloisKey(field_bitlen);
    }
    return;
  }

  auto parms = DecideSEALParameters(field_bitlen);
  auto *this_context =
      new seal::SEALContext(parms, true, seal::sec_level_type::none);
  seal::KeyGenerator keygen(*this_context);
  auto *rlwe_sk = new seal::SecretKey(keygen.secret_key());

  // exchange the public keys
  int nxt_rank = lctx_->NextRank();
  auto pk = keygen.create_public_key();
  auto pk_buf = EncodeSEALObject(pk.obj());
  auto peer_public_key = std::make_shared<seal::PublicKey>();
  if (nxt_rank == 0) {
    lctx_->Send(nxt_rank, pk_buf, "Rank1 send public key");
    auto recv_pk = lctx_->Recv(nxt_rank, "Rank1 recv public key");
    DecodeSEALObject(recv_pk, *this_context, peer_public_key.get());
  } else {
    auto recv_pk = lctx_->Recv(nxt_rank, "Rank0 recv public key");
    lctx_->Send(nxt_rank, pk_buf, "Rank1 send public key");
    DecodeSEALObject(recv_pk, *this_context, peer_public_key.get());
  }

  auto modulus = this_context->first_context_data()->parms().coeff_modulus();
  size_t enc_modulus_sze = modulus.size();
  parms.set_coeff_modulus(modulus);
  seal::SEALContext ecd_ms_context(parms, false, seal::sec_level_type::none);

  if (kUseModDownOptimization) {
    modulus.pop_back();
    if (field_bitlen > 64) {
      // Keep 3 primes are enough for FM128.
      modulus.pop_back();
    }
  }

  size_t dcd_modulus_sze = modulus.size();
  parms.set_use_special_prime(false);
  parms.set_coeff_modulus(modulus);
  seal::SEALContext dcd_ms_context(parms, false, seal::sec_level_type::none);

  seal_cntxts_.emplace(field_bitlen, this_context);
  secret_keys_.emplace(field_bitlen, rlwe_sk);
  peer_pub_keys_.emplace(field_bitlen, peer_public_key);

  ecd_mswh_.emplace(field_bitlen,
                    new ModulusSwitchHelper(ecd_ms_context, field_bitlen));
  dcd_mswh_.emplace(field_bitlen,
                    new ModulusSwitchHelper(dcd_ms_context, field_bitlen));

  decryptors_.emplace(field_bitlen,
                      new seal::Decryptor(*this_context, *rlwe_sk));
  if (need_galois_keys) {
    LazyInitGaloisKey(field_bitlen);
  }

  if (lctx_->Rank() == 0) {
    SPDLOG_INFO(
        "CheetahDot uses {}@{} modulus {} degree for {} bit ring (packing={})",
        enc_modulus_sze, dcd_modulus_sze, parms.poly_modulus_degree(),
        field_bitlen, need_galois_keys ? "enabled" : "disabled");
  }
}

void CheetahDot::Impl::NimbusCopSetup(const NdArrayRef& lhs,
                                      const Shape3D& dims,
                                      const std::string& plan_sha256,
                                      int64_t model_epoch,
                                      int64_t family_slot) {

spu::lifecycle::Scope scope_nimbus_model_cache_prepare("Nimbus.model-cache-prepare", "unclassified", [&] { return static_cast<uint64_t>((lctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_nimbus_model_cache_prepare.Outcome("initialized");

  const NimbusCopCacheKey key{plan_sha256, model_epoch, family_slot};
  {
    std::lock_guard guard(nimbus_cop_cache_mutex_);
    SPU_ENFORCE(nimbus_cop_cache_.find(key) == nimbus_cop_cache_.end(),
                "Nimbus COP cache is already initialized for model epoch {} "
                "family slot {}", model_epoch, family_slot);
  }
  SPU_ENFORCE(lhs.eltype().isa<Ring2k>());
  SPU_ENFORCE(lhs.ndim() == 2);
  SPU_ENFORCE_EQ(lhs.numel(), dims[0] * dims[1]);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  SPU_ENFORCE_EQ(field, FieldType::FM64,
                 "Nimbus COP setup currently supports FM64 only");
  const size_t field_bitlen = SizeOf(field) * 8;
  SPU_ENFORCE(dims[0] <= static_cast<int64_t>(
                                  DecideSEALParameters(field_bitlen)
                                      .poly_modulus_degree()),
              "Nimbus COP rows exceed the polynomial degree");

  LazyInit(field_bitlen, false);
  const auto& context = *seal_cntxts_.at(field_bitlen);
  const auto& encode_msh = *ecd_mswh_.at(field_bitlen);
  MatMatProtocol protocol(context, encode_msh, false, true);
  const MatMatProtocol::Meta meta = {.dims = dims};
  const size_t encrypted_count = protocol.GetLeftSize(meta);

spu::lifecycle::Scope scope_nimbus_cache_entry_shape("Nimbus.cache-entry-shape", "A");

  auto entry = std::make_shared<NimbusCopCacheEntry>();
  entry->field = field;
  entry->dims = dims;

scope_nimbus_cache_entry_shape.Stop();

  if (lctx_->Rank() == 0) {
    entry->clear_lhs = ring_add(lhs, ring_zeros(field, lhs.shape()));
    std::vector<RLWEPt> plaintexts(encrypted_count);

spu::lifecycle::Scope scope_nimbus_weight_encode("Nimbus.weight-encode", "R-M");

    protocol.EncodeLHS(lhs, meta, true, absl::MakeSpan(plaintexts));

scope_nimbus_weight_encode.Stop();

    entry->encrypted_lhs.resize(encrypted_count);

spu::lifecycle::Scope scope_nimbus_weight_encrypt("Nimbus.weight-encrypt", "R-M");

    SymmetricRLWEEncrypt(*secret_keys_.at(field_bitlen), context,
                         absl::MakeConstSpan(plaintexts),
                         false, true,
                         absl::MakeSpan(entry->encrypted_lhs));

scope_nimbus_weight_encrypt.Stop();

spu::lifecycle::Scope scope_nimbus_weight_serialize_send("Nimbus.weight-serialize-send", "R-M", [&] { return static_cast<uint64_t>((lctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");

    const int peer = lctx_->NextRank();
    for (size_t index = 0; index < encrypted_count; ++index) {
      auto encoded = EncodeSEALObject(entry->encrypted_lhs[index]);
      if (index + 1 == encrypted_count) {
        lctx_->Send(peer, encoded, "Nimbus COP setup weight");
      } else {
        lctx_->SendAsync(peer, encoded, "Nimbus COP setup weight");
      }
    }

scope_nimbus_weight_serialize_send.Stop();

    entry->encrypted_lhs.clear();
  } else {

spu::lifecycle::Scope scope_nimbus_weight_receive_and_cache("Nimbus.weight-receive-and-cache", "R-M", [&] { return static_cast<uint64_t>((lctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");

    entry->encrypted_lhs.resize(encrypted_count);
    const int peer = lctx_->NextRank();
    for (auto& ciphertext : entry->encrypted_lhs) {
      const auto encoded = lctx_->Recv(peer, "Nimbus COP setup weight");
      DecodeSEALObject(encoded, context, &ciphertext);
      SPU_ENFORCE(!ciphertext.is_ntt_form());
    }

scope_nimbus_weight_receive_and_cache.Stop();

  }
  std::lock_guard guard(nimbus_cop_cache_mutex_);
  const bool inserted = nimbus_cop_cache_.emplace(key, entry).second;
  SPU_ENFORCE(inserted, "Nimbus COP cache was concurrently initialized");
}

NdArrayRef CheetahDot::Impl::NimbusCopOnline(const NdArrayRef& rhs,
                                             const Shape3D& dims,
                                             const std::string& plan_sha256,
                                             int64_t model_epoch,
                                             int64_t request_epoch,
                                             int64_t family_slot) {

spu::lifecycle::Scope scope_nimbus_online_cache_and_he("Nimbus.online-cache-and-HE", "F", [&] { return static_cast<uint64_t>((lctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_nimbus_online_cache_and_he.Outcome("cache-used");

  const NimbusCopCacheKey key{plan_sha256, model_epoch, family_slot};
  std::shared_ptr<NimbusCopCacheEntry> entry;
  {
    std::lock_guard guard(nimbus_cop_cache_mutex_);
    const auto cached = nimbus_cop_cache_.find(key);
    SPU_ENFORCE(cached != nimbus_cop_cache_.end(),
                "Nimbus COP cache is not initialized for model epoch {} "
                "family slot {}", model_epoch, family_slot);
    entry = cached->second;
  }
  {
    std::lock_guard guard(entry->request_mutex);
    const bool inserted =
        entry->used_request_epochs.emplace(request_epoch).second;
    SPU_ENFORCE(inserted,
                "Nimbus COP request epoch {} was already consumed for model "
                "epoch {} family slot {}",
                request_epoch, model_epoch, family_slot);
    entry->live_request_epochs.emplace(request_epoch);
  }
  SPU_ENFORCE(rhs.eltype().isa<Ring2k>());
  SPU_ENFORCE_EQ(rhs.eltype().as<Ring2k>()->field(), entry->field);
  SPU_ENFORCE(rhs.ndim() == 2);
  SPU_ENFORCE_EQ(rhs.numel(), dims[1] * dims[2]);
  for (size_t dimension = 0; dimension < 2; ++dimension) {
    SPU_ENFORCE_EQ(dims[dimension], entry->dims[dimension]);
  }

  const size_t field_bitlen = SizeOf(entry->field) * 8;
  const auto& context = *seal_cntxts_.at(field_bitlen);
  const auto& encode_msh = *ecd_mswh_.at(field_bitlen);
  const auto& decode_msh = *dcd_mswh_.at(field_bitlen);
  MatMatProtocol protocol(context, encode_msh, false, true);
  const MatMatProtocol::Meta meta = {.dims = dims};
  const auto subshape = protocol.GetSubMatShape(meta);
  const size_t degree = context.first_context_data()
                            ->parms()
                            .poly_modulus_degree();
  const size_t row_extent = static_cast<size_t>(subshape[0]);
  SPU_ENFORCE_EQ(degree % row_extent, 0UL);
  const size_t columns_per_ciphertext = degree / row_extent;
  const size_t response_count = CeilDiv<size_t>(dims[2],
                                                columns_per_ciphertext);

  if (lctx_->Rank() == 0) {
    std::vector<RLWEPt> plaintexts(response_count);
    const int peer = lctx_->NextRank();
    for (size_t index = 0; index < response_count; ++index) {
      RLWECt ciphertext;
      const auto encoded = lctx_->Recv(peer, "Nimbus COP online result");
      DecodeSEALObject(encoded, context, &ciphertext);
      NttInplace(ciphertext, context);
      decryptors_.at(field_bitlen)->decrypt(ciphertext, plaintexts[index]);
      InvNttInplace(plaintexts[index], context);
    }
    auto cross_share = protocol.ParseNimbusPackedResult(
        entry->field, meta, absl::MakeConstSpan(plaintexts), decode_msh);
    return ring_add(ring_mmul(entry->clear_lhs, rhs), cross_share);
  }

  std::vector<RLWECt> products(protocol.GetOutSize(meta));
  protocol.Compute(absl::MakeConstSpan(entry->encrypted_lhs), rhs, meta,
                   absl::MakeSpan(products));

  std::vector<RLWECt> packed(response_count);
  seal::Evaluator evaluator(context);
  for (size_t begin = 0; begin < products.size();
       begin += columns_per_ciphertext) {
    const size_t count =
        std::min(columns_per_ciphertext, products.size() - begin);
    auto& accumulator = products[begin];
    for (size_t offset = 1; offset < count; ++offset) {
      auto shifted = products[begin + offset];
      NimbusNegacyclicRightShiftInplace(
          shifted, offset * subshape[0], context);
      CATCH_SEAL_ERROR(evaluator.add_inplace(accumulator, shifted));
    }
    packed[begin / columns_per_ciphertext] = std::move(accumulator);
  }

  std::vector<RLWEPt> masks(response_count);
  H2A(absl::MakeSpan(packed), absl::MakeSpan(masks),
      decode_msh.coeff_modulus_size(), *peer_pub_keys_.at(field_bitlen),
      context);
  const int peer = lctx_->NextRank();
  for (size_t index = 0; index < response_count; ++index) {
    auto encoded = EncodeSEALObject(packed[index]);
    if (index + 1 == response_count) {
      lctx_->Send(peer, encoded, "Nimbus COP online result");
    } else {
      lctx_->SendAsync(peer, encoded, "Nimbus COP online result");
    }
  }
  return protocol.ParseNimbusPackedResult(
      entry->field, meta, absl::MakeConstSpan(masks), decode_msh);
}

void CheetahDot::Impl::NimbusCopRequestEnd(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch, int64_t family_slot) {
  const NimbusCopCacheKey key{plan_sha256, model_epoch, family_slot};
  std::shared_ptr<NimbusCopCacheEntry> entry;
  {
    std::lock_guard guard(nimbus_cop_cache_mutex_);
    const auto cached = nimbus_cop_cache_.find(key);
    SPU_ENFORCE(cached != nimbus_cop_cache_.end(),
                "Nimbus COP cache is not initialized at request end");
    entry = cached->second;
  }
  std::lock_guard guard(entry->request_mutex);
  SPU_ENFORCE(entry->live_request_epochs.erase(request_epoch) == 1,
              "Nimbus COP request epoch {} is not live", request_epoch);
}

void CheetahDot::Impl::NimbusCopRelease(const std::string& plan_sha256,
                                        int64_t model_epoch,
                                        int64_t family_slot) {
  const NimbusCopCacheKey key{plan_sha256, model_epoch, family_slot};
  std::lock_guard cache_guard(nimbus_cop_cache_mutex_);
  const auto cached = nimbus_cop_cache_.find(key);
  SPU_ENFORCE(cached != nimbus_cop_cache_.end(),
              "Nimbus COP cache is not initialized at release");
  {
    std::lock_guard request_guard(cached->second->request_mutex);
    SPU_ENFORCE(cached->second->live_request_epochs.empty(),
                "Nimbus COP cache cannot be released with live requests");
  }
  nimbus_cop_cache_.erase(cached);
}

void CheetahDot::Impl::doDotOLEReceiverRecvStep(const NdArrayRef &prv_mat,
                                                const Shape3D &dim3,
                                                bool is_self_lhs,
                                                CipherPackingType cptype,
                                                absl::Span<RLWECt> result_cts,
                                                yacl::link::Context *conn) {
  int next_rank = conn->NextRank();
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();
  const size_t field_bitlen = SizeOf(field) * 8;

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  const auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  bool disable_pack = cptype == CipherPackingType::none;

  MatMatProtocol matmat_prot(this_context, this_ecd_msh, disable_pack);

  MatMatProtocol::Meta meta;
  meta.dims = dim3;
  auto subshape = matmat_prot.GetSubMatShape(meta);
  const size_t lhs_n = matmat_prot.GetLeftSize(meta, subshape);
  const size_t rhs_n = matmat_prot.GetRightSize(meta, subshape);
  const size_t out_n = matmat_prot.GetOutSize(meta, subshape);
  SPU_ENFORCE_EQ(out_n, result_cts.size());

  // 1. launch IO task to recv ct from peer
  std::vector<RLWECt> enc_mat(is_self_lhs ? rhs_n : lhs_n);
  auto io_task = std::async(std::launch::async, [&]() {
    for (auto &ct : enc_mat) {
      auto ct_s = conn->Recv(next_rank, "recv encrypted mat");
      DecodeSEALObject(ct_s, this_context, &ct);
    }
  });

  // 2. encode the matrix for multiplication
  auto plain_mat=MakeTemporaryPlaintexts(is_self_lhs ? lhs_n : rhs_n);
  if (is_self_lhs) {
    matmat_prot.EncodeLHS(prv_mat, meta, false, absl::MakeSpan(plain_mat));
  } else {
    matmat_prot.EncodeRHS(prv_mat, meta, false, absl::MakeSpan(plain_mat));
  }

  yacl::parallel_for(0, plain_mat.size(), [&](size_t bgn, size_t end) {
    for (size_t i = bgn; i < end; ++i) {
      NttInplace(plain_mat[i], this_context);
    }
  });
  io_task.get();

  if(plain_mat.size()>1024) {
    SPDLOG_INFO("CHEETAH_TEMP_PLAINTEXT receiver count={} scoped_pool_bytes={} global_pool_bytes={}",
                plain_mat.size(),plain_mat.front().pool().alloc_byte_count(),
                seal::MemoryManager::GetPool().alloc_byte_count());
  }

  // 3. HE multiplications
  if (is_self_lhs) {
    matmat_prot.Compute(plain_mat, enc_mat, meta, result_cts);
  } else {
    matmat_prot.Compute(enc_mat, plain_mat, meta, result_cts);
  }
}

NdArrayRef CheetahDot::Impl::doDotOLEReceiverSendStep(
    FieldType field, size_t batch_size, MatMatProtocol::Meta meta,
    absl::Span<RLWECt> ct_array_to_pack, CipherPackingType cptype,
    yacl::link::Context *conn, size_t bytes_recv) {
  int next_rank = conn->NextRank();
  const size_t field_bitlen = SizeOf(field) * 8;
  const auto &this_public_key = *(peer_pub_keys_.find(field_bitlen)->second);
  const auto &this_dcd_msh = *dcd_mswh_.find(field_bitlen)->second;
  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  const auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  bool disable_pack = cptype == CipherPackingType::none;

  MatMatProtocol matmat_prot(this_context, this_ecd_msh, disable_pack);

  auto subshape = matmat_prot.GetSubMatShape(meta);

  size_t out_n = ct_array_to_pack.size();
  size_t num_ct_response = 0;

  yacl::ElapsedTimer pack_timer;
  if (cptype == CipherPackingType::none) {
    SPU_ENFORCE(batch_size == 1, "impossible dispatch here");
    num_ct_response = out_n;
  } else if (cptype == CipherPackingType::rlwes) {
    // BumbleBee's interleave
    const auto &this_galois_key =
        *(peer_galois_keys_.find(field_bitlen)->second);

    const size_t gap = subshape[1];
    const size_t pack_stride = gap;

    PackingHelper pack_helper(gap, this_ecd_msh.coeff_modulus_size(),
                              this_galois_key, this_context);

    for (size_t i = 0; i < out_n; i += pack_stride) {
      size_t this_batch = std::min(out_n - i, pack_stride);
      size_t packed_idx = i / pack_stride;
      pack_helper.PackingWithModulusDrop(
          ct_array_to_pack.subspan(i, this_batch),
          ct_array_to_pack[packed_idx]);
    }

    num_ct_response = CeilDiv(out_n, pack_stride);
  } else {
    // Chen Hao et al's PackLWEs
    SPU_ENFORCE(batch_size == 1, "not implemented yet");
    const auto &this_galois_key =
        *(peer_galois_keys_.find(field_bitlen)->second);

    // Drop some modulus before packing
    yacl::parallel_for(
        0, ct_array_to_pack.size(), [&](int64_t bgn, int64_t end) {
          for (int64_t i = bgn; i < end; ++i) {
            InvNttInplace(ct_array_to_pack[i], this_context);
            ModulusSwtichInplace(ct_array_to_pack[i],
                                 this_dcd_msh.coeff_modulus_size(),
                                 this_context);
          }
        });

    // Extract Phantom LWECt from RLWEs
    size_t out_numel = meta.dims[0] * meta.dims[2];
    size_t aligned_onumel = absl::bit_ceil(out_numel);
    std::vector<PhantomLWECt> _lwes(aligned_onumel);
    auto lwes = absl::MakeSpan(_lwes);
    matmat_prot.WrapPhantomLWEs(meta, ct_array_to_pack,
                                lwes.subspan(0, out_numel));

    size_t pack_stride = lwes[0].poly_modulus_degree();
    std::vector<RLWECt> packed_lwes(CeilDiv(lwes.size(), pack_stride));

    // NOTE: we can not modify the ct_array_to_pack inplace because it wraps the
    // PhantomLWECt
    PackLWEs(lwes, this_galois_key, this_context, absl::MakeSpan(packed_lwes));

    num_ct_response = packed_lwes.size();

    // copy back
    for (size_t i = 0; i < num_ct_response; ++i) {
      ct_array_to_pack[i] = packed_lwes[i];
    }
  }
  double pack_time = pack_timer.CountMs();

  // 4. Random masking to conver HE to AShr
  std::vector<RLWEPt> rnd_polys(num_ct_response);
  H2A({ct_array_to_pack.data(), num_ct_response}, absl::MakeSpan(rnd_polys),
      this_dcd_msh.coeff_modulus_size(), this_public_key, this_context);

  if (cptype == CipherPackingType::none) {
    // If no packing, then clean up un-used coefficients
    // NOTE(lwj): we place Extract **after** H2A for a smaller communication
    matmat_prot.ExtractLWEsInplace(meta, ct_array_to_pack);
  }

  size_t bytes_sent = conn->GetStats()->sent_bytes;
  for (size_t i = 0; i < num_ct_response; ++i) {
    conn->SendAsync(next_rank, EncodeSEALObject(ct_array_to_pack[i]), "");
  }
  bytes_sent = conn->GetStats()->sent_bytes - bytes_sent;

  SPDLOG_INFO(
      "{}@{}x{}x{} => {}x{}x{} Recv {} MiB, Response {} MiB Pack {} ms ({})",
      batch_size, meta.dims[0], meta.dims[1], meta.dims[2], subshape[0],
      subshape[1], subshape[2],
      std::roundf(bytes_recv / 1024. / 1024. * 1000) / 1000.,
      std::roundf(bytes_sent / 1024. / 1024. * 1000) / 1000.,
      std::roundf(pack_time * 1000) / 1000., ToString(cptype));

  switch (cptype) {
    case CipherPackingType::none:
      return matmat_prot.ParseResult(
          field, meta, absl::MakeConstSpan(rnd_polys), this_dcd_msh);
    case CipherPackingType::lwes:
      return matmat_prot.ParsePackLWEsResult(
          field, meta, absl::MakeConstSpan(rnd_polys), this_dcd_msh);
    case CipherPackingType::rlwes:
    default:
      return matmat_prot.ParseBatchPackedResult(field, batch_size, meta,
                                                absl::MakeConstSpan(rnd_polys),
                                                this_dcd_msh);
  }
}

void CheetahDot::Impl::doDotOLESenderSendStep(const NdArrayRef &prv_mat,
                                              const Shape3D &dim3,
                                              bool is_self_lhs,
                                              CipherPackingType cptype,
                                              yacl::link::Context *conn) {
  int next_rank = conn->NextRank();
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();
  const size_t field_bitlen = SizeOf(field) * 8;

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_secret_key = *secret_keys_.find(field_bitlen)->second;
  auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  bool disable_pack = cptype == CipherPackingType::none;
  MatMatProtocol matmat_prot(this_context, this_ecd_msh, disable_pack);

  MatMatProtocol::Meta meta;
  meta.dims = dim3;
  auto subshape = matmat_prot.GetSubMatShape(meta);

  const size_t lhs_n = matmat_prot.GetLeftSize(meta, subshape);
  const size_t rhs_n = matmat_prot.GetRightSize(meta, subshape);

  auto _encoded_mat=MakeTemporaryPlaintexts(is_self_lhs ? lhs_n : rhs_n);
  auto encoded_mat = absl::MakeSpan(_encoded_mat);
  if (is_self_lhs) {
    matmat_prot.EncodeLHS(prv_mat, meta, true, encoded_mat);
  } else {
    matmat_prot.EncodeRHS(prv_mat, meta, true, encoded_mat);
  }

  size_t num_ct_to_send = encoded_mat.size();
  for (size_t i = 0; i < num_ct_to_send; i += kCtAsyncParallel) {
    size_t this_batch = std::min(num_ct_to_send - i, kCtAsyncParallel);
    std::vector<RLWECt> enc_mat(this_batch);
    std::vector<yacl::Buffer> ct_s(this_batch);

    SymmetricRLWEEncrypt(this_secret_key, this_context,
                         encoded_mat.subspan(i, this_batch),
                         /*ntt*/ true,
                         /*seed*/ true, absl::MakeSpan(enc_mat));

    yacl::parallel_for(0, this_batch, [&](size_t bgn, size_t end) {
      for (size_t j = bgn; j < end; ++j) {
        ct_s[j] = EncodeSEALObject(enc_mat[j]);
      }
    });

    for (size_t j = 1; j < this_batch; ++j) {
      conn->SendAsync(next_rank, ct_s[j - 1], "send encrypted mat");
    }
    conn->Send(next_rank, ct_s[this_batch - 1], "send encrypted mat");
  }
}

NdArrayRef CheetahDot::Impl::doDotOLESenderRecvStep(FieldType field,
                                                    size_t batch_size,
                                                    MatMatProtocol::Meta meta,
                                                    size_t num_ct_to_recv,
                                                    CipherPackingType cptype,
                                                    yacl::link::Context *conn) {
  SPU_ENFORCE(batch_size > 0);
  int next_rank = conn->NextRank();
  const size_t field_bitlen = SizeOf(field) * 8;

  const auto &this_context = *seal_cntxts_.find(field_bitlen)->second;
  auto &this_decryptor = decryptors_.find(field_bitlen)->second;
  auto &this_ecd_msh = *ecd_mswh_.find(field_bitlen)->second;
  auto &this_dcd_msh = *dcd_mswh_.find(field_bitlen)->second;

  MatMatProtocol matmat_prot(this_context, this_ecd_msh,
                             cptype == CipherPackingType::none);

  std::vector<RLWECt> recv_ct(kCtAsyncParallel);
  std::vector<RLWEPt> result_poly(num_ct_to_recv);

  for (size_t i = 0; i < num_ct_to_recv; i += kCtAsyncParallel) {
    size_t this_batch = std::min(num_ct_to_recv - i, kCtAsyncParallel);
    for (size_t j = 0; j < this_batch; ++j) {
      auto ct_s = conn->Recv(next_rank, "recv result mat");
      DecodeSEALObject(ct_s, this_context, &recv_ct[j]);
    }

    yacl::parallel_for(0, this_batch, [&](size_t bgn, size_t end) {
      for (size_t j = bgn; j < end; ++j) {
        NttInplace(recv_ct[j], this_context);
        this_decryptor->decrypt(recv_ct[j], result_poly[i + j]);
        // non-ntt form for ParseResult
        InvNttInplace(result_poly[i + j], this_context);
      }
    });
  }

  switch (cptype) {
    case CipherPackingType::none:
      return matmat_prot.ParseResult(
          field, meta, absl::MakeConstSpan(result_poly), this_dcd_msh);
    case CipherPackingType::lwes:
      return matmat_prot.ParsePackLWEsResult(
          field, meta, absl::MakeConstSpan(result_poly), this_dcd_msh);
    case CipherPackingType::rlwes:
    default:
      return matmat_prot.ParseBatchPackedResult(
          field, batch_size, meta, absl::MakeConstSpan(result_poly),
          this_dcd_msh);
  }
}

NdArrayRef CheetahDot::Impl::DotOLE(const NdArrayRef &prv_mat,
                                    yacl::link::Context *conn,
                                    const Shape3D &dim3, bool is_self_lhs) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }
  auto eltype = prv_mat.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(prv_mat.numel() > 0 && prv_mat.ndim() == 2);

  if (is_self_lhs) {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim3[0] * dim3[1]);
  } else {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim3[1] * dim3[2]);
  }

  return doDotOLE(prv_mat, conn, dim3, is_self_lhs);
}

NdArrayRef CheetahDot::Impl::BatchDotOLE(const NdArrayRef &prv_mat,
                                         yacl::link::Context *conn,
                                         const Shape4D &dim4,
                                         bool is_self_lhs) {
  if (conn == nullptr) {
    conn = lctx_.get();
  }
  auto eltype = prv_mat.eltype();
  SPU_ENFORCE(eltype.isa<Ring2k>(), "must be ring_type, got={}", eltype);
  SPU_ENFORCE(prv_mat.numel() > 0 && prv_mat.ndim() == 3);

  if (is_self_lhs) {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim4[0] * dim4[1] * dim4[2]);
  } else {
    SPU_ENFORCE_EQ(prv_mat.numel(), dim4[0] * dim4[2] * dim4[3]);
  }

  auto field = eltype.template as<Ring2k>()->field();
  if (not IsPackingEnabled(8 * SizeOf(field))) {
    // Packing is not supported.
    // Thus just call multiple DotOLEs
    Shape3D dim3 = {dim4[1], dim4[2], dim4[3]};
    int64_t out_numel = dim4[1] * dim4[3];
    NdArrayRef out(eltype, {dim4[0] * out_numel});
    const auto &mat_shape = prv_mat.shape();

    for (int64_t b = 0; b < dim4[0]; ++b) {
      auto one_mat =
          prv_mat
              .slice({b, 0, 0}, {b + 1, mat_shape[1], mat_shape[2]}, {1, 1, 1})
              .reshape({mat_shape[1], mat_shape[2]});
      auto one_out = DotOLE(one_mat, conn, dim3, is_self_lhs);
      auto out_slice =
          out.slice({b * out_numel}, {b * out_numel + out_numel}, {1});
      pforeach(0, one_out.numel(), [&](int64_t i) {
        std::memcpy(&out_slice.at(i), &one_out.at(i), one_out.elsize());
      });
    }

    return out.reshape({dim4[0], dim4[1], dim4[3]});
  }

  return doBatchDotOLE(prv_mat, conn, dim4, is_self_lhs);
}

NdArrayRef CheetahDot::Impl::doBatchDotOLE(const NdArrayRef &prv_mat,
                                           yacl::link::Context *conn,
                                           const Shape4D &dim4,
                                           bool is_self_lhs) {
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();

  const size_t field_bitlen = SizeOf(field) * 8;
  size_t poly_deg = DecideSEALParameters(field_bitlen).poly_modulus_degree();

  const Shape3D dim3 = {dim4[1], dim4[2], dim4[3]};
  const size_t batch_size = dim4[0];
  MatMatProtocol::Meta meta = {.dims = dim3};

  CipherPackingType cptype = CipherPackingType::rlwes;
  auto subshape =
      MatMatProtocol::GetSubMatShape(meta, poly_deg, /*disable_pack*/ false);

  size_t blk0 = CeilDiv(meta.dims[0], subshape[0]);
  size_t blk1 = CeilDiv(meta.dims[1], subshape[1]);
  size_t blk2 = CeilDiv(meta.dims[2], subshape[2]);
  size_t lhs_poly_n = blk0 * blk1;
  size_t rhs_poly_n = blk1 * blk2;
  bool to_encrypt_lhs = lhs_poly_n <= rhs_poly_n;
  bool act_as_encryptor = (is_self_lhs ^ to_encrypt_lhs) == 0;

  LazyInit(field_bitlen, cptype != CipherPackingType::none);
  auto mat_shape = prv_mat.shape();

  if (act_as_encryptor) {
    for (int64_t b = 0; b < (int64_t)batch_size; ++b) {
      auto one_mat =
          prv_mat
              .slice({b, 0, 0}, {b + 1, mat_shape[1], mat_shape[2]}, {1, 1, 1})
              .reshape({mat_shape[1], mat_shape[2]});
      doDotOLESenderSendStep(one_mat, dim3, is_self_lhs, cptype, conn);
    }

    size_t num_ct_to_recv = 0;
    switch (cptype) {
      case CipherPackingType::rlwes:
        num_ct_to_recv = CeilDiv<size_t>(batch_size * blk0 * blk2, subshape[1]);
        break;
      default:
        num_ct_to_recv = blk0 * blk2;
    }
    return doDotOLESenderRecvStep(field, batch_size, meta, num_ct_to_recv,
                                  cptype, conn);
  }

  size_t num_ct_per_batch = blk0 * blk2;
  std::vector<RLWECt> _ct_array_to_pack(batch_size * num_ct_per_batch);
  auto ct_array_to_pack = absl::MakeSpan(_ct_array_to_pack);

  size_t bytes_recv = conn->GetStats()->recv_bytes;
  for (int64_t b = 0; b < (int64_t)batch_size; ++b) {
    auto one_mat =
        prv_mat.slice({b, 0, 0}, {b + 1, mat_shape[1], mat_shape[2]}, {1, 1, 1})
            .reshape({mat_shape[1], mat_shape[2]});
    doDotOLEReceiverRecvStep(
        one_mat, dim3, is_self_lhs, cptype,
        ct_array_to_pack.subspan(b * num_ct_per_batch, num_ct_per_batch), conn);
  }
  bytes_recv = conn->GetStats()->recv_bytes - bytes_recv;

  return doDotOLEReceiverSendStep(field, batch_size, meta, ct_array_to_pack,
                                  cptype, conn, bytes_recv);
}

NdArrayRef CheetahDot::Impl::doDotOLE(const NdArrayRef &prv_mat,
                                      yacl::link::Context *conn,
                                      const Shape3D &dim3, bool is_self_lhs) {
  auto eltype = prv_mat.eltype();
  auto field = eltype.template as<Ring2k>()->field();
  const size_t field_bitlen = SizeOf(field) * 8;
  size_t poly_deg = DecideSEALParameters(field_bitlen).poly_modulus_degree();

  MatMatProtocol::Meta meta = {.dims = dim3};

  // No cipher packing for small HE
  CipherPackingType cptype =
      (IsPackingEnabled(8 * SizeOf(field)) and not disable_pack_)
          ? CipherPackingType::rlwes
          : CipherPackingType::none;

  Shape3D subshape;
  size_t blk[3];
  if (cptype != CipherPackingType::none) {
    // attempt to calculate the cost with packing
    subshape = MatMatProtocol::GetSubMatShape(meta, poly_deg, false);
    for (int i : {0, 1, 2}) {
      blk[i] = CeilDiv(meta.dims[i], subshape[i]);
    }

    if (blk[0] * blk[2] <= 1) {
      // If there is only 1 resultant RLWE;
      // then we just skip any packing
      cptype = CipherPackingType::none;
    } else {
      // dynamic packing type
      double pack_rlwes_cost = subshape[1];
      double pack_lwes_cost = meta.dims[0] * meta.dims[2];
      cptype = pack_rlwes_cost < pack_lwes_cost ? CipherPackingType::rlwes
                                                : CipherPackingType::lwes;
    }
  }

  LazyInit(field_bitlen, cptype != CipherPackingType::none);

  // Update subshape
  subshape = MatMatProtocol::GetSubMatShape(meta, poly_deg,
                                            cptype == CipherPackingType::none);

  for (int i : {0, 1, 2}) {
    blk[i] = CeilDiv(meta.dims[i], subshape[i]);
  }

  size_t lhs_poly_n = blk[0] * blk[1];
  size_t rhs_poly_n = blk[1] * blk[2];
  bool to_encrypt_lhs = lhs_poly_n <= rhs_poly_n;
  bool act_as_encryptor = (is_self_lhs ^ to_encrypt_lhs) == 0;

  if (act_as_encryptor) {
    doDotOLESenderSendStep(prv_mat, dim3, is_self_lhs, cptype, conn);

    size_t num_ct_to_recv = 0;
    switch (cptype) {
      case CipherPackingType::rlwes:
        num_ct_to_recv = CeilDiv<size_t>(blk[0] * blk[2], subshape[1]);
        break;
      case CipherPackingType::lwes:
        num_ct_to_recv = CeilDiv<size_t>(meta.dims[0] * meta.dims[2], poly_deg);
        break;
      default:
      case CipherPackingType::none:
        num_ct_to_recv = blk[0] * blk[2];
        break;
    }

    return doDotOLESenderRecvStep(field, /*batch*/ 1, meta, num_ct_to_recv,
                                  cptype, conn)
        .reshape({dim3[0], dim3[2]});
  }

  std::vector<RLWECt> _ct_array_to_pack(blk[0] * blk[2]);
  auto ct_array_to_pack = absl::MakeSpan(_ct_array_to_pack);
  size_t bytes_recv = conn->GetStats()->recv_bytes;
  doDotOLEReceiverRecvStep(prv_mat, dim3, is_self_lhs, cptype, ct_array_to_pack,
                           conn);
  bytes_recv = conn->GetStats()->recv_bytes - bytes_recv;

  return doDotOLEReceiverSendStep(field, /*batch*/ 1, meta, ct_array_to_pack,
                                  cptype, conn, bytes_recv)
      .reshape({dim3[0], dim3[2]});
}

//////////////////////////////////////////////

CheetahDot::CheetahDot(const std::shared_ptr<yacl::link::Context> &lctx,
                       bool disable_matmul_pack) {
  impl_ = std::make_unique<Impl>(lctx, disable_matmul_pack);
}

CheetahDot::~CheetahDot() = default;

void CheetahDot::LazyInitKeys(FieldType field) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->LazyInit(SizeOf(field) * 8,
                         impl_->IsPackingEnabled(SizeOf(field) * 8));
}

NdArrayRef CheetahDot::DotOLE(const NdArrayRef &inp, yacl::link::Context *conn,
                              const Shape3D &dim3, bool is_self_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  SPU_ENFORCE(conn != nullptr);
  return impl_->DotOLE(inp, conn, dim3, is_self_lhs);
}

NdArrayRef CheetahDot::DotOLE(const NdArrayRef &inp, const Shape3D &dim3,
                              bool is_self_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->DotOLE(inp, nullptr, dim3, is_self_lhs);
}

NdArrayRef CheetahDot::BatchDotOLE(const NdArrayRef &inp,
                                   yacl::link::Context *conn,
                                   const Shape4D &dim4, bool is_self_lhs) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->BatchDotOLE(inp, conn, dim4, is_self_lhs);
}

void CheetahDot::NimbusCopSetup(const NdArrayRef& lhs, const Shape3D& dims,
                                const std::string& plan_sha256,
                                int64_t model_epoch, int64_t family_slot) {
  SPU_ENFORCE(impl_ != nullptr);
  impl_->NimbusCopSetup(lhs, dims, plan_sha256, model_epoch, family_slot);
}

NdArrayRef CheetahDot::NimbusCopOnline(const NdArrayRef& rhs,
                                       const Shape3D& dims,
                                       const std::string& plan_sha256,
                                       int64_t model_epoch,
                                       int64_t request_epoch,
                                       int64_t family_slot) {
  SPU_ENFORCE(impl_ != nullptr);
  return impl_->NimbusCopOnline(rhs, dims, plan_sha256, model_epoch,
                                request_epoch, family_slot);
}

void CheetahDot::NimbusCopRequestEnd(const std::string& plan_sha256,
                                     int64_t model_epoch,
                                     int64_t request_epoch,
                                     int64_t family_slot) {
  SPU_ENFORCE(impl_ != nullptr);
  impl_->NimbusCopRequestEnd(plan_sha256, model_epoch, request_epoch,
                             family_slot);
}

void CheetahDot::NimbusCopRelease(const std::string& plan_sha256,
                                  int64_t model_epoch, int64_t family_slot) {
  SPU_ENFORCE(impl_ != nullptr);
  impl_->NimbusCopRelease(plan_sha256, model_epoch, family_slot);
}

}  // namespace spu::mpc::cheetah
