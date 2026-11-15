// Copyright 2026

#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"
#include "libspu/core/shape.h"

namespace spu::mpc::cheetah {

enum class ObjectLifetime {
  kModelStatic,
  kRequestImmutable,
  kAppendOnlyUnsupported,
};

enum class ObjectOperand { kLhs, kRhs };

struct ObjectKey {
  std::string logical_object_id;
  int64_t epoch = 0;
  int64_t segment = 0;

  bool operator<(const ObjectKey& other) const {
    return std::tie(logical_object_id, epoch, segment) <
           std::tie(other.logical_object_id, other.epoch, other.segment);
  }

  bool operator==(const ObjectKey& other) const {
    return logical_object_id == other.logical_object_id &&
           epoch == other.epoch && segment == other.segment;
  }
};

struct ConsumerKey {
  ObjectKey object;
  std::string consumer_id;

  bool operator<(const ConsumerKey& other) const {
    return std::tie(object, consumer_id) <
           std::tie(other.object, other.consumer_id);
  }
};

struct ObjectAuditSnapshot {
  int64_t required_shared_object_contexts = 0;
  int64_t shared_side_logical_openings = 0;
  int64_t shared_side_opened_elements = 0;
  int64_t shared_side_physical_collectives = 0;
  int64_t logical_right_openings = 0;
  int64_t right_opened_elements = 0;
  int64_t physical_right_collectives = 0;
  int64_t physical_family_executions = 0;
  int64_t embedding_physical_executions = 0;
};

struct PhaseBatchAuditSnapshot {
  int64_t model_setup_rpc_calls = 0;
  int64_t request_preprocess_rpc_calls = 0;
};

struct RuntimeObjectAuditV1 {
  static constexpr int64_t kSchemaVersion = 1;
  static constexpr size_t kFieldCount = 33;

  int64_t schema_version = kSchemaVersion;
  int64_t plan_matches = 0;
  int64_t model_epoch = 0;
  int64_t request_epoch = 0;
  int64_t logical_consumer_count = 0;
  int64_t physical_family_count = 0;
  ObjectAuditSnapshot model;
  ObjectAuditSnapshot request;
  PhaseBatchAuditSnapshot phase_batches;
  int64_t consumed_logical_consumers = 0;
  int64_t expected_logical_consumers = 0;
  int64_t fresh_request_masks = 0;
  int64_t static_object_reused = 0;
  int64_t secure_invoke_rpc_calls = 0;
  int64_t protocol_matrix_calls = 0;
  int64_t dot_ole_calls = 0;
  int64_t batch_dot_ole_calls = 0;
  int64_t live_object_contexts = 0;
  int64_t live_consumer_contexts = 0;

  std::vector<int64_t> ToVector() const;
};

struct RuntimeFamilySpec {
  std::string physical_family_id;
  std::string logical_object_id;
  int64_t logical_members = 0;
  ObjectLifetime object_lifetime = ObjectLifetime::kRequestImmutable;
  ObjectOperand object_operand = ObjectOperand::kLhs;
  Shape shared_shape;
  Shape right_shape;
  std::string family_kind;
  bool consumed = false;
};

class MJVObjectContextState final : public State {
 public:
  static constexpr char kBindName[] = "MJVObjectContext";

  void RegisterObject(const ObjectKey& key, ObjectLifetime lifetime,
                      const Shape& shape, std::string mask_digest,
                      bool requires_shared_opening = true);
  void RegisterConsumer(const ConsumerKey& key, const Shape& right_shape);
  void RegisterConsumer(const ConsumerKey& key, const Shape& right_shape,
                        ObjectLifetime right_lifetime, int64_t right_epoch);
  void PrepareEpoch(ObjectLifetime lifetime, int64_t epoch);
  void OpenSharedObjectOnce(const ObjectKey& key);
  void OpenRightOperandOnce(const ConsumerKey& key);
  void OpenRightFamilyOnce(const std::vector<ConsumerKey>& keys);
  void ConsumeConsumer(const ConsumerKey& key);
  void ConsumeFamilyOnce(const std::vector<ConsumerKey>& keys);
  void RecordPhysicalFamilyExecution(ObjectLifetime lifetime, int64_t epoch,
                                     const std::string& family_kind);
  void RecordModelSetupBatch(const std::string& plan_sha256,
                             int64_t model_epoch);
  void RecordRequestPreprocessBatch(const std::string& plan_sha256,
                                    int64_t model_epoch,
                                    int64_t request_epoch);
  PhaseBatchAuditSnapshot AuditPhaseBatches(
      const std::string& plan_sha256, int64_t model_epoch,
      int64_t request_epoch) const;
  void BeginPlanExecution(std::string plan_sha256, int64_t model_epoch,
                          int64_t request_epoch);
  void RecordSecureInvoke();
  void RecordProtocolMatrixCall();
  void RecordHeSubcall(bool batched);
  void ReconcileLinearAudit(bool ours_backend, int64_t high_level_calls,
                            int64_t dot_ole_calls,
                            int64_t batch_dot_ole_calls);
  void RegisterModelFamily(
      const std::string& plan_sha256, int64_t model_epoch,
      std::string physical_family_id, std::string logical_object_id,
      int64_t logical_members, ObjectLifetime object_lifetime,
      ObjectOperand object_operand, const Shape& shared_shape,
      const Shape& right_shape, std::string family_kind);
  void RegisterRequestFamily(const std::string& plan_sha256,
                             int64_t model_epoch, int64_t request_epoch,
                             const std::string& physical_family_id);
  void RegisterRequestFamily(
      const std::string& plan_sha256, int64_t model_epoch,
      int64_t request_epoch, std::string physical_family_id,
      std::string logical_object_id, int64_t logical_members,
      ObjectLifetime object_lifetime, ObjectOperand object_operand,
      const Shape& shared_shape, const Shape& right_shape,
      std::string family_kind);
  void RecordRuntimeFamily(const std::string& plan_sha256,
                           int64_t model_epoch, int64_t request_epoch,
                           const std::string& physical_family_id);
  RuntimeObjectAuditV1 FinalizeRuntimeAudit(
      const std::string& plan_sha256, int64_t model_epoch,
      int64_t request_epoch);
  void EndPlanExecution(const std::string& plan_sha256, int64_t model_epoch,
                        int64_t request_epoch,
                        bool release_model_epoch = false);
  void AbortPlanExecution();
  bool IsPlanExecutionActive() const;
  ObjectAuditSnapshot FinishEpoch(ObjectLifetime lifetime, int64_t epoch);
  void AbortEpoch(ObjectLifetime lifetime, int64_t epoch);

  ObjectAuditSnapshot AuditEpoch(ObjectLifetime lifetime, int64_t epoch) const;
  std::string MaskDigest(const ObjectKey& key) const;
  int64_t LiveObjectCount() const;
  int64_t LiveConsumerCount() const;

 private:
  using EpochKey = std::pair<ObjectLifetime, int64_t>;
  using ModelFamilyKey = std::tuple<std::string, int64_t, std::string>;
  using RequestFamilyKey =
      std::tuple<std::string, int64_t, int64_t, std::string>;
  using ModelObjectKey = std::tuple<std::string, int64_t, std::string>;
  using RequestObjectKey =
      std::tuple<std::string, int64_t, int64_t, std::string>;

  struct ObjectRecord {
    ObjectLifetime lifetime = ObjectLifetime::kRequestImmutable;
    Shape shape;
    std::string mask_digest;
    bool requires_shared_opening = true;
    bool shared_opened = false;
  };

  struct ConsumerRecord {
    Shape right_shape;
    ObjectLifetime right_lifetime = ObjectLifetime::kRequestImmutable;
    int64_t right_epoch = 0;
    bool right_opened = false;
  };

  mutable std::mutex mutex_;
  std::map<ObjectKey, ObjectRecord> objects_;
  std::map<ConsumerKey, ConsumerRecord> consumers_;
  std::set<ObjectKey> retired_objects_;
  std::set<ConsumerKey> retired_consumers_;
  std::set<std::string> request_mask_digests_;
  std::set<EpochKey> active_epochs_;
  std::map<EpochKey, ObjectAuditSnapshot> audits_;
  std::set<std::pair<std::string, int64_t>> model_setup_batches_;
  std::set<std::tuple<std::string, int64_t, int64_t>>
      request_preprocess_batches_;
  bool plan_execution_active_ = false;
  std::string active_plan_sha256_;
  int64_t active_model_epoch_ = 0;
  int64_t active_request_epoch_ = 0;
  int64_t active_expected_consumers_ = 0;
  int64_t active_consumed_consumers_ = 0;
  int64_t active_secure_invoke_rpc_calls_ = 0;
  int64_t active_protocol_matrix_calls_ = 0;
  int64_t active_dot_ole_calls_ = 0;
  int64_t active_batch_dot_ole_calls_ = 0;
  std::set<std::tuple<std::string, int64_t, int64_t>> finalized_audits_;
  std::set<std::pair<std::string, int64_t>> released_model_epochs_;
  std::map<ModelFamilyKey, RuntimeFamilySpec> model_families_;
  std::map<RequestFamilyKey, RuntimeFamilySpec> request_families_;
  std::set<ModelObjectKey> runtime_model_shared_objects_;
  std::set<RequestObjectKey> runtime_request_shared_objects_;
};

NdArrayRef EvaluateMaskedMatmul(const NdArrayRef& correlation,
                                const NdArrayRef& object_mask,
                                const NdArrayRef& right_mask,
                                const NdArrayRef& opened_object_difference,
                                const NdArrayRef& opened_right_difference,
                                ObjectOperand object_operand,
                                bool add_public_cross_term);

}
