// Copyright 2026

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/shape.h"
#include "libspu/core/value.h"

namespace spu::mpc::cheetah {

enum class LinearPhase { kSetup, kPreprocess, kOnline };

enum class LinearBackend { kBumbleBee, kOurs };

enum class LinearKernelFamily {
  kMatMulAA,
  kMatMulAV,
  kMatMulVVS,
  kBatchMatMulAA,
  kBatchMatMulAV,
};

struct LinearCallFingerprint {
  int64_t ordinal = 0;
  LinearKernelFamily family = LinearKernelFamily::kMatMulAA;
  Shape lhs_shape;
  Shape rhs_shape;
  Shape output_shape;

  bool operator==(const LinearCallFingerprint& other) const {
    return ordinal == other.ordinal && family == other.family &&
           lhs_shape == other.lhs_shape && rhs_shape == other.rhs_shape &&
           output_shape == other.output_shape;
  }
};

struct LinearAuditSnapshot {
  LinearPhase phase = LinearPhase::kSetup;
  LinearBackend backend = LinearBackend::kBumbleBee;
  std::vector<LinearCallFingerprint> high_level_calls;
  int64_t dot_ole_calls = 0;
  int64_t batch_dot_ole_calls = 0;
};

struct DynamicConsumerKey {
  std::string plan_sha256;
  int64_t request_epoch = 0;
  std::string consumer_id;

  bool operator<(const DynamicConsumerKey& other) const {
    return std::tie(plan_sha256, request_epoch, consumer_id) <
           std::tie(other.plan_sha256, other.request_epoch, other.consumer_id);
  }

  bool operator==(const DynamicConsumerKey& other) const {
    return plan_sha256 == other.plan_sha256 &&
           request_epoch == other.request_epoch &&
           consumer_id == other.consumer_id;
  }
};

struct DynamicRequestContextKey {
  std::string plan_sha256;
  int64_t request_epoch = 0;

  bool operator<(const DynamicRequestContextKey& other) const {
    return std::tie(plan_sha256, request_epoch) <
           std::tie(other.plan_sha256, other.request_epoch);
  }

  bool operator==(const DynamicRequestContextKey& other) const {
    return plan_sha256 == other.plan_sha256 &&
           request_epoch == other.request_epoch;
  }
};

class MJVLinearAuditState final : public State {
 public:
  static constexpr char kBindName[] = "MJVLinearAudit";

  void Begin(LinearPhase phase, LinearBackend backend);
  LinearAuditSnapshot End();

  void RecordHighLevel(LinearKernelFamily family, const Shape& lhs_shape,
                       const Shape& rhs_shape, const Shape& output_shape);
  void RecordHeSubcall(bool batched);
  void Abort();

 private:
  mutable std::mutex mutex_;
  bool active_ = false;
  LinearAuditSnapshot snapshot_;
};

class MJVDynamicLinearState final : public State {
 public:
  static constexpr char kBindName[] = "MJVDynamicLinear";

  struct CorrelationContext {
    NdArrayRef lhs_mask;
    NdArrayRef rhs_mask;
    NdArrayRef correlation;
    Shape lhs_shape;
    Shape rhs_shape;
    bool batched = false;
    int64_t model_epoch = 0;
    std::string physical_family_id;
  };

  void Store(DynamicConsumerKey key, CorrelationContext correlation);
  CorrelationContext Consume(const DynamicConsumerKey& key);
  int64_t LiveCorrelationCount() const;
  void BeginOnline(std::string plan_sha256, int64_t request_epoch);
  void EndOnline(const std::string& plan_sha256, int64_t request_epoch);
  DynamicConsumerKey NextOnlineConsumer(
      const Shape& lhs_shape, const Shape& rhs_shape, bool batched) const;
  void AbortOnline();

 private:
  mutable std::mutex mutex_;
  std::map<DynamicConsumerKey, CorrelationContext> correlations_;
  std::set<DynamicConsumerKey> used_correlation_keys_;
  std::map<DynamicRequestContextKey, std::vector<DynamicConsumerKey>>
      prepared_order_;
  bool online_active_ = false;
  DynamicRequestContextKey online_context_;
  std::vector<DynamicConsumerKey> online_order_;
  int64_t online_cursor_ = 0;
};

class MJVLinearScopeState final : public State {
 public:
  static constexpr char kBindName[] = "MJVLinearScope";

  void RegisterPublicPlan(const std::string& hash,
                          const std::vector<LinearCallFingerprint>& calls);

  void Begin(int64_t request_id, std::string manifest_hash,
             LinearBackend backend);
  LinearBackend ActiveBackend() const;
  std::string ActiveManifestHash() const;
  bool IsActive() const;
  LinearKernelFamily ConsumeExpectedFamily(const Shape& lhs_shape,
                                           const Shape& rhs_shape,
                                           const Shape& output_shape);
  void Finish(int64_t request_id, const LinearAuditSnapshot& snapshot);
  void Abort();

 private:
  mutable std::mutex mutex_;
  bool active_ = false;
  int64_t request_id_ = 0;
  std::string manifest_hash_;
  LinearBackend backend_ = LinearBackend::kBumbleBee;
  int64_t replacement_cursor_ = 0;
  std::map<std::string, std::vector<LinearCallFingerprint>> baselines_;
};

class MJVConsumerPreprocess final : public Kernel {
 public:
  static constexpr char kBindName[] = "mjv_consumer_preprocess";

  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

class MJVDynamicOnline final : public Kernel {
 public:
  static constexpr char kBindName[] = "mjv_dynamic_online";

  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

Value ConsumeNextMJVDynamicOnline(KernelEvalContext* ctx, const Value& lhs,
                                  const Value& rhs,
                                  LinearKernelFamily family);

void VerifyLinearPathParity(
    const std::vector<LinearCallFingerprint>& baseline,
    const std::vector<LinearCallFingerprint>& replacement);

}
