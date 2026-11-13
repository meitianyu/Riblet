// Copyright 2026

#include "libspu/mpc/cheetah/mjv_object_context.h"

#include <algorithm>
#include <utility>

#include "libspu/core/prelude.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
namespace {

int64_t Numel(const Shape& shape) {
  SPU_ENFORCE(shape.numel() > 0, "object context shapes must be non-empty");
  return shape.numel();
}

bool IsPlanDigest(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void CheckRuntimeFamily(const RuntimeFamilySpec& family) {
  SPU_ENFORCE(!family.physical_family_id.empty() &&
                  !family.logical_object_id.empty(),
              "runtime family identity must not be empty");
  SPU_ENFORCE(family.logical_members > 0,
              "runtime family logical members must be positive");
  SPU_ENFORCE(
      family.object_lifetime != ObjectLifetime::kAppendOnlyUnsupported,
      "append-only lifetime is deferred from v0.1");
  SPU_ENFORCE(!family.family_kind.empty(),
              "runtime family kind must not be empty");
  Numel(family.shared_shape);
  Numel(family.right_shape);
}

}  

std::vector<int64_t> RuntimeObjectAuditV1::ToVector() const {
  std::vector<int64_t> fields = {
      schema_version,
      plan_matches,
      model_epoch,
      request_epoch,
      logical_consumer_count,
      physical_family_count,
      model.required_shared_object_contexts,
      model.shared_side_logical_openings,
      model.shared_side_opened_elements,
      model.shared_side_physical_collectives,
      model.logical_right_openings,
      model.right_opened_elements,
      model.physical_right_collectives,
      request.required_shared_object_contexts,
      request.shared_side_logical_openings,
      request.shared_side_opened_elements,
      request.shared_side_physical_collectives,
      request.logical_right_openings,
      request.right_opened_elements,
      request.physical_right_collectives,
      request.embedding_physical_executions,
      phase_batches.model_setup_rpc_calls,
      phase_batches.request_preprocess_rpc_calls,
      consumed_logical_consumers,
      expected_logical_consumers,
      fresh_request_masks,
      static_object_reused,
      secure_invoke_rpc_calls,
      protocol_matrix_calls,
      dot_ole_calls,
      batch_dot_ole_calls,
      live_object_contexts,
      live_consumer_contexts,
  };
  SPU_ENFORCE_EQ(fields.size(), kFieldCount);
  return fields;
}

void MJVObjectContextState::RegisterObject(
    const ObjectKey& key, ObjectLifetime lifetime, const Shape& shape,
    std::string mask_digest, bool requires_shared_opening) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(lifetime != ObjectLifetime::kAppendOnlyUnsupported,
              "append-only lifetime is deferred from v0.1");
  SPU_ENFORCE(!key.logical_object_id.empty(),
              "logical object ID must not be empty");
  SPU_ENFORCE(key.epoch >= 0 && key.segment == 0,
              "v0.1 object keys require a nonnegative epoch and segment zero");
  SPU_ENFORCE(!mask_digest.empty(), "mask digest must not be empty");
  Numel(shape);
  SPU_ENFORCE(objects_.find(key) == objects_.end() &&
                  retired_objects_.find(key) == retired_objects_.end(),
              "logical object epoch was already registered");
  if (lifetime == ObjectLifetime::kRequestImmutable) {
    SPU_ENFORCE(request_mask_digests_.insert(mask_digest).second,
                "request mask digest was reused across epochs");
  }
  objects_.emplace(
      key, ObjectRecord{lifetime, shape, std::move(mask_digest),
                        requires_shared_opening, false});
}

void MJVObjectContextState::RegisterConsumer(const ConsumerKey& key,
                                             const Shape& right_shape) {
  ObjectLifetime lifetime;
  {
    std::lock_guard guard(mutex_);
    const auto object = objects_.find(key.object);
    SPU_ENFORCE(object != objects_.end(),
                "consumer references an undeclared object");
    lifetime = object->second.lifetime;
  }
  RegisterConsumer(key, right_shape, lifetime, key.object.epoch);
}

void MJVObjectContextState::RegisterConsumer(
    const ConsumerKey& key, const Shape& right_shape,
    ObjectLifetime right_lifetime, int64_t right_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!key.consumer_id.empty(), "consumer ID must not be empty");
  SPU_ENFORCE(right_lifetime != ObjectLifetime::kAppendOnlyUnsupported,
              "append-only lifetime is deferred from v0.1");
  SPU_ENFORCE(right_epoch >= 0, "right operand epoch must be nonnegative");
  Numel(right_shape);
  SPU_ENFORCE(objects_.find(key.object) != objects_.end(),
              "consumer references an undeclared object");
  SPU_ENFORCE(consumers_.find(key) == consumers_.end() &&
                  retired_consumers_.find(key) == retired_consumers_.end(),
              "consumer was already registered or consumed");
  consumers_.emplace(
      key, ConsumerRecord{right_shape, right_lifetime, right_epoch, false});
}

void MJVObjectContextState::PrepareEpoch(ObjectLifetime lifetime,
                                         int64_t epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(lifetime != ObjectLifetime::kAppendOnlyUnsupported,
              "append-only lifetime is deferred from v0.1");
  const EpochKey epoch_key{lifetime, epoch};
  SPU_ENFORCE(active_epochs_.insert(epoch_key).second,
              "object epoch is already active");
  ObjectAuditSnapshot audit;
  for (const auto& [key, object] : objects_) {
    if (key.epoch == epoch && object.lifetime == lifetime &&
        object.requires_shared_opening) {
      ++audit.required_shared_object_contexts;
    }
  }
  const auto has_right_participants = std::any_of(
      consumers_.begin(), consumers_.end(), [&](const auto& item) {
        return item.second.right_lifetime == lifetime &&
               item.second.right_epoch == epoch;
      });
  SPU_ENFORCE(audit.required_shared_object_contexts > 0 ||
                  has_right_participants,
              "object epoch has no participating contexts");
  audits_[epoch_key] = audit;
}

void MJVObjectContextState::OpenSharedObjectOnce(const ObjectKey& key) {
  std::lock_guard guard(mutex_);
  auto object = objects_.find(key);
  SPU_ENFORCE(object != objects_.end(), "unknown or retired logical object");
  const EpochKey epoch_key{object->second.lifetime, key.epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "logical object epoch is not active");
  SPU_ENFORCE(object->second.requires_shared_opening,
              "logical object does not require a shared-side opening");
  SPU_ENFORCE(!object->second.shared_opened,
              "shared object Gamma was already opened");
  object->second.shared_opened = true;
  auto& audit = audits_.at(epoch_key);
  ++audit.shared_side_logical_openings;
  audit.shared_side_opened_elements += Numel(object->second.shape);
  ++audit.shared_side_physical_collectives;
}

void MJVObjectContextState::OpenRightOperandOnce(const ConsumerKey& key) {
  std::lock_guard guard(mutex_);
  auto consumer = consumers_.find(key);
  SPU_ENFORCE(consumer != consumers_.end(), "unknown or consumed consumer");
  auto object = objects_.find(key.object);
  SPU_ENFORCE(object != objects_.end(), "consumer object is not live");
  const EpochKey epoch_key{consumer->second.right_lifetime,
                           consumer->second.right_epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "consumer epoch is not active");
  SPU_ENFORCE(!consumer->second.right_opened,
              "consumer right operand was already opened");
  consumer->second.right_opened = true;
  auto& audit = audits_.at(epoch_key);
  ++audit.logical_right_openings;
  audit.right_opened_elements += Numel(consumer->second.right_shape);
  ++audit.physical_right_collectives;
}

void MJVObjectContextState::OpenRightFamilyOnce(
    const std::vector<ConsumerKey>& keys) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!keys.empty(), "right family must not be empty");
  const auto first = consumers_.find(keys.front());
  SPU_ENFORCE(first != consumers_.end(), "unknown or consumed consumer");
  const EpochKey epoch_key{first->second.right_lifetime,
                           first->second.right_epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "right family epoch is not active");
  int64_t opened_elements = 0;
  for (const auto& key : keys) {
    auto consumer = consumers_.find(key);
    SPU_ENFORCE(consumer != consumers_.end(), "unknown or consumed consumer");
    SPU_ENFORCE(consumer->second.right_lifetime == epoch_key.first &&
                    consumer->second.right_epoch == epoch_key.second,
                "right family crosses epoch boundaries");
    SPU_ENFORCE(!consumer->second.right_opened,
                "consumer right operand was already opened");
    consumer->second.right_opened = true;
    opened_elements += Numel(consumer->second.right_shape);
  }
  auto& audit = audits_.at(epoch_key);
  audit.logical_right_openings += keys.size();
  audit.right_opened_elements += opened_elements;
  ++audit.physical_right_collectives;
}

void MJVObjectContextState::ConsumeConsumer(const ConsumerKey& key) {
  std::lock_guard guard(mutex_);
  auto consumer = consumers_.find(key);
  SPU_ENFORCE(consumer != consumers_.end(), "consumer was already consumed");
  auto object = objects_.find(key.object);
  SPU_ENFORCE(object != objects_.end(), "consumer object is not live");
  const EpochKey epoch_key{consumer->second.right_lifetime,
                           consumer->second.right_epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "consumer epoch is not active");
  SPU_ENFORCE(object->second.shared_opened,
              "consumer cannot run before shared Gamma opens");
  SPU_ENFORCE(consumer->second.right_opened,
              "consumer cannot run before its right Delta opens");
  if (plan_execution_active_ &&
      consumer->second.right_lifetime == ObjectLifetime::kRequestImmutable &&
      consumer->second.right_epoch == active_request_epoch_) {
    ++active_consumed_consumers_;
  }
  retired_consumers_.insert(key);
  consumers_.erase(consumer);
}

void MJVObjectContextState::ConsumeFamilyOnce(
    const std::vector<ConsumerKey>& keys) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!keys.empty(), "consumer family must not be empty");
  for (const auto& key : keys) {
    auto consumer = consumers_.find(key);
    SPU_ENFORCE(consumer != consumers_.end(), "consumer was already consumed");
    const auto object = objects_.find(key.object);
    SPU_ENFORCE(object != objects_.end(), "consumer object is not live");
    const EpochKey epoch_key{consumer->second.right_lifetime,
                             consumer->second.right_epoch};
    SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
                "consumer family epoch is not active");
    SPU_ENFORCE(object->second.shared_opened,
                "consumer cannot run before shared Gamma opens");
    SPU_ENFORCE(consumer->second.right_opened,
                "consumer cannot run before its right Delta opens");
  }
  for (const auto& key : keys) {
    const auto consumer = consumers_.find(key);
    if (plan_execution_active_ &&
        consumer->second.right_lifetime ==
            ObjectLifetime::kRequestImmutable &&
        consumer->second.right_epoch == active_request_epoch_) {
      ++active_consumed_consumers_;
    }
    retired_consumers_.insert(key);
    consumers_.erase(key);
  }
}

void MJVObjectContextState::RecordPhysicalFamilyExecution(
    ObjectLifetime lifetime, int64_t epoch, const std::string& family_kind) {
  std::lock_guard guard(mutex_);
  const EpochKey epoch_key{lifetime, epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "physical family epoch is not active");
  auto& audit = audits_.at(epoch_key);
  ++audit.physical_family_executions;
  if (family_kind == "embedding_batch") {
    ++audit.embedding_physical_executions;
  }
}

void MJVObjectContextState::RecordModelSetupBatch(
    const std::string& plan_sha256, int64_t model_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(IsPlanDigest(plan_sha256), "invalid MJV object plan digest");
  SPU_ENFORCE(model_epoch >= 0, "model epoch must be nonnegative");
  SPU_ENFORCE(released_model_epochs_.count({plan_sha256, model_epoch}) == 0,
              "model epoch was already released");
  SPU_ENFORCE(model_setup_batches_.emplace(plan_sha256, model_epoch).second,
              "model setup batch was already recorded for this plan epoch");
}

void MJVObjectContextState::RecordRequestPreprocessBatch(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(IsPlanDigest(plan_sha256), "invalid MJV object plan digest");
  SPU_ENFORCE(model_epoch >= 0 && request_epoch >= 0,
              "model and request epochs must be nonnegative");
  SPU_ENFORCE(request_preprocess_batches_
                  .emplace(plan_sha256, model_epoch, request_epoch)
                  .second,
              "request preprocess batch was already recorded for this plan epoch");
}

PhaseBatchAuditSnapshot MJVObjectContextState::AuditPhaseBatches(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch) const {
  std::lock_guard guard(mutex_);
  return {
      static_cast<int64_t>(
          model_setup_batches_.count({plan_sha256, model_epoch})),
      static_cast<int64_t>(request_preprocess_batches_.count(
          {plan_sha256, model_epoch, request_epoch})),
  };
}

void MJVObjectContextState::BeginPlanExecution(std::string plan_sha256,
                                               int64_t model_epoch,
                                               int64_t request_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(!plan_execution_active_, "MJV object plan is already active");
  SPU_ENFORCE(IsPlanDigest(plan_sha256), "invalid MJV object plan digest");
  SPU_ENFORCE(model_epoch >= 0 && request_epoch >= 0,
              "MJV object plan epochs must be nonnegative");
  SPU_ENFORCE(
      finalized_audits_.count({plan_sha256, model_epoch, request_epoch}) == 0,
      "MJV object plan epoch audit was already finalized");
  SPU_ENFORCE(released_model_epochs_.count({plan_sha256, model_epoch}) == 0,
              "model epoch was already released");
  plan_execution_active_ = true;
  active_plan_sha256_ = std::move(plan_sha256);
  active_model_epoch_ = model_epoch;
  active_request_epoch_ = request_epoch;
  active_expected_consumers_ = 0;
  for (const auto& [key, family] : request_families_) {
    if (std::get<0>(key) == active_plan_sha256_ &&
        std::get<1>(key) == model_epoch && std::get<2>(key) == request_epoch) {
      active_expected_consumers_ += family.logical_members;
    }
  }
  if (active_expected_consumers_ == 0) {
    active_expected_consumers_ = std::count_if(
        consumers_.begin(), consumers_.end(), [&](const auto& item) {
          return item.second.right_lifetime ==
                     ObjectLifetime::kRequestImmutable &&
                 item.second.right_epoch == request_epoch;
        });
  }
  active_consumed_consumers_ = 0;
  active_secure_invoke_rpc_calls_ = 0;
  active_protocol_matrix_calls_ = 0;
  active_dot_ole_calls_ = 0;
  active_batch_dot_ole_calls_ = 0;
}

void MJVObjectContextState::RecordSecureInvoke() {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_, "MJV object plan is not active");
  ++active_secure_invoke_rpc_calls_;
}

void MJVObjectContextState::RecordProtocolMatrixCall() {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_, "MJV object plan is not active");
  ++active_protocol_matrix_calls_;
}

void MJVObjectContextState::RecordHeSubcall(bool batched) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_, "MJV object plan is not active");
  if (batched) {
    ++active_batch_dot_ole_calls_;
  } else {
    ++active_dot_ole_calls_;
  }
}

void MJVObjectContextState::ReconcileLinearAudit(
    bool ours_backend, int64_t high_level_calls, int64_t dot_ole_calls,
    int64_t batch_dot_ole_calls) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_, "MJV object plan is not active");
  SPU_ENFORCE(high_level_calls >= 0 && dot_ole_calls >= 0 &&
                  batch_dot_ole_calls >= 0,
              "linear audit counters must be nonnegative");
  if (ours_backend) {
    SPU_ENFORCE_EQ(dot_ole_calls, 0,
                   "Ours online linear audit contains DotOLE calls");
    SPU_ENFORCE_EQ(batch_dot_ole_calls, 0,
                   "Ours online linear audit contains BatchDotOLE calls");
    SPU_ENFORCE_EQ(active_protocol_matrix_calls_, high_level_calls,
                   "Ours runtime family count differs from linear audit");
    return;
  }
  SPU_ENFORCE_EQ(active_protocol_matrix_calls_, 0,
                 "baseline protocol calls were already recorded");
  active_protocol_matrix_calls_ = high_level_calls;
  active_dot_ole_calls_ = dot_ole_calls;
  active_batch_dot_ole_calls_ = batch_dot_ole_calls;
}

void MJVObjectContextState::RegisterModelFamily(
    const std::string& plan_sha256, int64_t model_epoch,
    std::string physical_family_id, std::string logical_object_id,
    int64_t logical_members, ObjectLifetime object_lifetime,
    ObjectOperand object_operand, const Shape& shared_shape,
    const Shape& right_shape, std::string family_kind) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(IsPlanDigest(plan_sha256) && model_epoch >= 0,
              "invalid model runtime family epoch");
  RuntimeFamilySpec family{std::move(physical_family_id),
                           std::move(logical_object_id),
                           logical_members,
                           object_lifetime,
                           object_operand,
                           shared_shape,
                           right_shape,
                           std::move(family_kind),
                           false};
  CheckRuntimeFamily(family);
  SPU_ENFORCE(released_model_epochs_.count({plan_sha256, model_epoch}) == 0,
              "model epoch was already released");
  const ModelFamilyKey key{plan_sha256, model_epoch,
                           family.physical_family_id};
  SPU_ENFORCE(model_families_.emplace(key, family).second,
              "model runtime family was already registered");

  auto& audit = audits_[{ObjectLifetime::kModelStatic, model_epoch}];
  if (family.object_lifetime == ObjectLifetime::kModelStatic &&
      family.object_operand == ObjectOperand::kRhs) {
    const ModelObjectKey object_key{plan_sha256, model_epoch,
                                    family.logical_object_id};
    SPU_ENFORCE(runtime_model_shared_objects_.insert(object_key).second,
                "model shared object was already opened");
    ++audit.required_shared_object_contexts;
    ++audit.shared_side_logical_openings;
    audit.shared_side_opened_elements += Numel(family.shared_shape);
    ++audit.shared_side_physical_collectives;
  } else {
    audit.logical_right_openings += family.logical_members;
    audit.right_opened_elements += Numel(family.right_shape);
    ++audit.physical_right_collectives;
  }
}

void MJVObjectContextState::RegisterRequestFamily(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch, const std::string& physical_family_id) {
  std::lock_guard guard(mutex_);
  const auto model = model_families_.find(
      {plan_sha256, model_epoch, physical_family_id});
  SPU_ENFORCE(model != model_families_.end(),
              "request family has no model setup family");
  const RequestFamilyKey key{plan_sha256, model_epoch, request_epoch,
                             physical_family_id};
  SPU_ENFORCE(request_families_.emplace(key, model->second).second,
              "request runtime family was already registered");
}

void MJVObjectContextState::RegisterRequestFamily(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch, std::string physical_family_id,
    std::string logical_object_id, int64_t logical_members,
    ObjectLifetime object_lifetime, ObjectOperand object_operand,
    const Shape& shared_shape, const Shape& right_shape,
    std::string family_kind) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(IsPlanDigest(plan_sha256) && model_epoch >= 0 &&
                  request_epoch >= 0,
              "invalid request runtime family epoch");
  RuntimeFamilySpec family{std::move(physical_family_id),
                           std::move(logical_object_id),
                           logical_members,
                           object_lifetime,
                           object_operand,
                           shared_shape,
                           right_shape,
                           std::move(family_kind),
                           false};
  CheckRuntimeFamily(family);
  const RequestFamilyKey key{plan_sha256, model_epoch, request_epoch,
                             family.physical_family_id};
  SPU_ENFORCE(request_families_.emplace(key, std::move(family)).second,
              "request runtime family was already registered");
}

void MJVObjectContextState::RecordRuntimeFamily(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch, const std::string& physical_family_id) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_ && active_plan_sha256_ == plan_sha256 &&
                  active_model_epoch_ == model_epoch &&
                  active_request_epoch_ == request_epoch,
              "runtime family plan or epoch is not active");
  auto family = request_families_.find(
      {plan_sha256, model_epoch, request_epoch, physical_family_id});
  SPU_ENFORCE(family != request_families_.end(),
              "runtime family was not registered");
  SPU_ENFORCE(!family->second.consumed,
              "runtime family was already consumed");
  family->second.consumed = true;

  auto& audit = audits_[{ObjectLifetime::kRequestImmutable, request_epoch}];
  if (family->second.object_lifetime == ObjectLifetime::kRequestImmutable) {
    const RequestObjectKey object_key{plan_sha256, model_epoch, request_epoch,
                                      family->second.logical_object_id};
    if (runtime_request_shared_objects_.insert(object_key).second) {
      ++audit.required_shared_object_contexts;
      ++audit.shared_side_logical_openings;
      audit.shared_side_opened_elements += Numel(family->second.shared_shape);
      ++audit.shared_side_physical_collectives;
    }
  }
  if (family->second.family_kind == "embedding_batch" ||
      family->second.family_kind == "attention_matmul") {
    audit.logical_right_openings += family->second.logical_members;
    audit.right_opened_elements += Numel(family->second.right_shape);
    ++audit.physical_right_collectives;
  }
  ++audit.physical_family_executions;
  if (family->second.family_kind == "embedding_batch") {
    ++audit.embedding_physical_executions;
  }
  active_consumed_consumers_ += family->second.logical_members;
  ++active_protocol_matrix_calls_;
}

RuntimeObjectAuditV1 MJVObjectContextState::FinalizeRuntimeAudit(
    const std::string& plan_sha256, int64_t model_epoch,
    int64_t request_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_, "MJV object plan is not active");
  SPU_ENFORCE(active_plan_sha256_ == plan_sha256 &&
                  active_model_epoch_ == model_epoch &&
                  active_request_epoch_ == request_epoch,
              "MJV object plan or epoch changed before audit");
  SPU_ENFORCE_EQ(active_consumed_consumers_, active_expected_consumers_,
                 "MJV runtime audit has incomplete consumers");
  SPU_ENFORCE(
      finalized_audits_.emplace(plan_sha256, model_epoch, request_epoch).second,
      "MJV runtime audit was already finalized");

  RuntimeObjectAuditV1 result;
  result.plan_matches = 1;
  result.model_epoch = model_epoch;
  result.request_epoch = request_epoch;
  result.logical_consumer_count = active_consumed_consumers_;
  const auto model_audit = audits_.find({ObjectLifetime::kModelStatic,
                                         model_epoch});
  if (model_audit != audits_.end()) {
    result.model = model_audit->second;
  }
  const auto request_audit = audits_.find(
      {ObjectLifetime::kRequestImmutable, request_epoch});
  if (request_audit != audits_.end()) {
    result.request = request_audit->second;
  }
  result.physical_family_count = result.request.physical_family_executions;
  result.phase_batches = {
      static_cast<int64_t>(
          model_setup_batches_.count({plan_sha256, model_epoch})),
      static_cast<int64_t>(request_preprocess_batches_.count(
          {plan_sha256, model_epoch, request_epoch})),
  };
  result.consumed_logical_consumers = active_consumed_consumers_;
  result.expected_logical_consumers = active_expected_consumers_;
  result.fresh_request_masks = 1;
  result.static_object_reused = !runtime_model_shared_objects_.empty() ||
                                std::any_of(
      objects_.begin(), objects_.end(), [&](const auto& item) {
        return item.first.epoch == model_epoch &&
               item.second.lifetime == ObjectLifetime::kModelStatic &&
               item.second.shared_opened;
      });
  result.secure_invoke_rpc_calls = active_secure_invoke_rpc_calls_;
  result.protocol_matrix_calls = active_protocol_matrix_calls_;
  result.dot_ole_calls = active_dot_ole_calls_;
  result.batch_dot_ole_calls = active_batch_dot_ole_calls_;
  result.live_object_contexts =
      objects_.size() + runtime_model_shared_objects_.size();
  int64_t pending_runtime_consumers = 0;
  for (const auto& [key, family] : request_families_) {
    if (std::get<0>(key) == plan_sha256 && std::get<1>(key) == model_epoch &&
        std::get<2>(key) == request_epoch && !family.consumed) {
      pending_runtime_consumers += family.logical_members;
    }
  }
  result.live_consumer_contexts =
      consumers_.size() + pending_runtime_consumers;
  return result;
}

void MJVObjectContextState::EndPlanExecution(const std::string& plan_sha256,
                                             int64_t model_epoch,
                                             int64_t request_epoch,
                                             bool release_model_epoch) {
  std::lock_guard guard(mutex_);
  SPU_ENFORCE(plan_execution_active_, "MJV object plan is not active");
  SPU_ENFORCE(active_plan_sha256_ == plan_sha256 &&
                  active_model_epoch_ == model_epoch &&
                  active_request_epoch_ == request_epoch,
              "MJV object plan or epoch changed before end");
  std::set<int64_t> request_epochs;
  if (release_model_epoch) {
    for (const auto& [key, family] : request_families_) {
      if (std::get<0>(key) == plan_sha256 &&
          std::get<1>(key) == model_epoch) {
        SPU_ENFORCE(family.consumed,
                    "model epoch has an unconsumed runtime family");
        request_epochs.insert(std::get<2>(key));
      }
    }
    SPU_ENFORCE(
        released_model_epochs_.emplace(plan_sha256, model_epoch).second,
        "model epoch was already released");
  }

  plan_execution_active_ = false;
  active_plan_sha256_.clear();
  active_model_epoch_ = 0;
  active_request_epoch_ = 0;
  active_expected_consumers_ = 0;
  active_consumed_consumers_ = 0;
  if (!release_model_epoch) {
    return;
  }

  for (auto it = model_families_.begin(); it != model_families_.end();) {
    if (std::get<0>(it->first) == plan_sha256 &&
        std::get<1>(it->first) == model_epoch) {
      it = model_families_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = request_families_.begin(); it != request_families_.end();) {
    if (std::get<0>(it->first) == plan_sha256 &&
        std::get<1>(it->first) == model_epoch) {
      it = request_families_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = runtime_model_shared_objects_.begin();
       it != runtime_model_shared_objects_.end();) {
    if (std::get<0>(*it) == plan_sha256 && std::get<1>(*it) == model_epoch) {
      it = runtime_model_shared_objects_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = runtime_request_shared_objects_.begin();
       it != runtime_request_shared_objects_.end();) {
    if (std::get<0>(*it) == plan_sha256 && std::get<1>(*it) == model_epoch) {
      it = runtime_request_shared_objects_.erase(it);
    } else {
      ++it;
    }
  }
  model_setup_batches_.erase({plan_sha256, model_epoch});
  for (auto it = request_preprocess_batches_.begin();
       it != request_preprocess_batches_.end();) {
    if (std::get<0>(*it) == plan_sha256 && std::get<1>(*it) == model_epoch) {
      it = request_preprocess_batches_.erase(it);
    } else {
      ++it;
    }
  }
  audits_.erase({ObjectLifetime::kModelStatic, model_epoch});
  for (const auto epoch : request_epochs) {
    audits_.erase({ObjectLifetime::kRequestImmutable, epoch});
  }
}

void MJVObjectContextState::AbortPlanExecution() {
  std::lock_guard guard(mutex_);
  plan_execution_active_ = false;
  active_plan_sha256_.clear();
  active_expected_consumers_ = 0;
  active_consumed_consumers_ = 0;
}

bool MJVObjectContextState::IsPlanExecutionActive() const {
  std::lock_guard guard(mutex_);
  return plan_execution_active_;
}

ObjectAuditSnapshot MJVObjectContextState::FinishEpoch(
    ObjectLifetime lifetime, int64_t epoch) {
  std::lock_guard guard(mutex_);
  const EpochKey epoch_key{lifetime, epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "object epoch is not active");
  for (const auto& [key, consumer] : consumers_) {
    (void)key;
    if (consumer.right_epoch == epoch &&
        consumer.right_lifetime == lifetime) {
      SPU_THROW("object epoch has unconsumed consumers");
    }
  }
  for (const auto& [key, object] : objects_) {
    if (key.epoch == epoch && object.lifetime == lifetime &&
        object.requires_shared_opening) {
      SPU_ENFORCE(object.shared_opened,
                  "object epoch has an unopened shared context");
    }
  }
  const auto audit = audits_.at(epoch_key);
  SPU_ENFORCE_EQ(audit.shared_side_logical_openings,
                 audit.required_shared_object_contexts,
                 "shared object opening mismatch");
  active_epochs_.erase(epoch_key);
  if (lifetime == ObjectLifetime::kRequestImmutable) {
    for (auto it = objects_.begin(); it != objects_.end();) {
      if (it->first.epoch == epoch && it->second.lifetime == lifetime) {
        retired_objects_.insert(it->first);
        it = objects_.erase(it);
      } else {
        ++it;
      }
    }
  }
  return audit;
}

void MJVObjectContextState::AbortEpoch(ObjectLifetime lifetime, int64_t epoch) {
  std::lock_guard guard(mutex_);
  const EpochKey epoch_key{lifetime, epoch};
  SPU_ENFORCE(active_epochs_.find(epoch_key) != active_epochs_.end(),
              "object epoch is not active");
  for (auto it = consumers_.begin(); it != consumers_.end();) {
    if (it->second.right_epoch == epoch &&
        it->second.right_lifetime == lifetime) {
      retired_consumers_.insert(it->first);
      it = consumers_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = objects_.begin(); it != objects_.end();) {
    if (it->first.epoch == epoch && it->second.lifetime == lifetime) {
      retired_objects_.insert(it->first);
      it = objects_.erase(it);
    } else {
      ++it;
    }
  }
  active_epochs_.erase(epoch_key);
}

ObjectAuditSnapshot MJVObjectContextState::AuditEpoch(
    ObjectLifetime lifetime, int64_t epoch) const {
  std::lock_guard guard(mutex_);
  const auto found = audits_.find({lifetime, epoch});
  SPU_ENFORCE(found != audits_.end(), "unknown object epoch audit");
  return found->second;
}

std::string MJVObjectContextState::MaskDigest(const ObjectKey& key) const {
  std::lock_guard guard(mutex_);
  const auto found = objects_.find(key);
  SPU_ENFORCE(found != objects_.end(), "unknown or retired logical object");
  return found->second.mask_digest;
}

int64_t MJVObjectContextState::LiveObjectCount() const {
  std::lock_guard guard(mutex_);
  return objects_.size();
}

int64_t MJVObjectContextState::LiveConsumerCount() const {
  std::lock_guard guard(mutex_);
  return consumers_.size();
}

NdArrayRef EvaluateMaskedMatmul(
    const NdArrayRef& correlation, const NdArrayRef& object_mask,
    const NdArrayRef& right_mask, const NdArrayRef& opened_object_difference,
    const NdArrayRef& opened_right_difference, ObjectOperand object_operand,
    bool add_public_cross_term) {
  auto output = correlation.clone();
  if (object_operand == ObjectOperand::kLhs) {
    ring_add_(output,
              ring_mmul(opened_object_difference, right_mask));
    ring_add_(output, ring_mmul(object_mask, opened_right_difference));
    if (add_public_cross_term) {
      ring_add_(output, ring_mmul(opened_object_difference,
                                  opened_right_difference));
    }
  } else {
    ring_add_(output,
              ring_mmul(right_mask, opened_object_difference));
    ring_add_(output, ring_mmul(opened_right_difference, object_mask));
    if (add_public_cross_term) {
      ring_add_(output, ring_mmul(opened_right_difference,
                                  opened_object_difference));
    }
  }
  return output;
}

}
