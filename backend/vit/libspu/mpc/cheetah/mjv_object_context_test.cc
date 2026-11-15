// Copyright 2026

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/mjv_object_context.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
namespace {

ObjectKey RequestKey(std::string logical_object_id, int64_t epoch) {
  return {std::move(logical_object_id), epoch, 0};
}

ConsumerKey Consumer(const ObjectKey& object, std::string consumer_id) {
  return {object, std::move(consumer_id)};
}

void ExpectRingEqual(const NdArrayRef& actual, const NdArrayRef& expected) {
  ASSERT_EQ(actual.shape(), expected.shape());
  DISPATCH_ALL_FIELDS(FieldType::FM64, "mjv-object-context-test", [&]() {
    auto actual_view = NdArrayView<ring2k_t>(actual);
    auto expected_view = NdArrayView<ring2k_t>(expected);
    for (int64_t index = 0; index < expected.numel(); ++index) {
      EXPECT_EQ(actual_view[index], expected_view[index]);
    }
  });
}

TEST(MJVObjectContextTest, SameObjectThreeConsumersOpenGammaOnce) {
  MJVObjectContextState state;
  const auto object = RequestKey("request.hidden.0", 7);
  state.RegisterObject(object, ObjectLifetime::kRequestImmutable, {128, 768},
                       "request-mask-7");
  const auto query = Consumer(object, "layer0.q");
  const auto key = Consumer(object, "layer0.k");
  const auto value = Consumer(object, "layer0.v");
  state.RegisterConsumer(query, {768, 768});
  state.RegisterConsumer(key, {768, 768});
  state.RegisterConsumer(value, {768, 768});

  state.PrepareEpoch(ObjectLifetime::kRequestImmutable, 7);
  state.OpenSharedObjectOnce(object);
  for (const auto* consumer : {&query, &key, &value}) {
    state.OpenRightOperandOnce(*consumer);
    state.ConsumeConsumer(*consumer);
  }
  const auto audit =
      state.FinishEpoch(ObjectLifetime::kRequestImmutable, 7);

  EXPECT_EQ(audit.required_shared_object_contexts, 1);
  EXPECT_EQ(audit.shared_side_logical_openings, 1);
  EXPECT_EQ(audit.shared_side_physical_collectives, 1);
  EXPECT_EQ(audit.logical_right_openings, 3);
  EXPECT_EQ(audit.physical_right_collectives, 3);
}

TEST(MJVObjectContextTest, DynamicMasksAreFreshAcrossRequestEpochs) {
  MJVObjectContextState state;
  const auto first = RequestKey("request.hidden.0", 11);
  const auto second = RequestKey("request.hidden.0", 12);
  state.RegisterObject(first, ObjectLifetime::kRequestImmutable, {128, 768},
                       "request-mask-11");
  state.RegisterObject(second, ObjectLifetime::kRequestImmutable, {128, 768},
                       "request-mask-12");
  EXPECT_NE(state.MaskDigest(first), state.MaskDigest(second));

  const auto reused = RequestKey("request.hidden.1", 13);
  EXPECT_THROW(
      state.RegisterObject(reused, ObjectLifetime::kRequestImmutable,
                           {128, 768}, "request-mask-11"),
      std::exception);
}

TEST(MJVObjectContextTest,
     StaticObjectOpeningSurvivesRequestsUntilModelEpochChanges) {
  MJVObjectContextState state;
  const ObjectKey first{"model.word_embeddings", 3, 0};
  state.RegisterObject(first, ObjectLifetime::kModelStatic, {30522, 768},
                       "model-mask-3");
  state.PrepareEpoch(ObjectLifetime::kModelStatic, 3);
  state.OpenSharedObjectOnce(first);
  const auto first_audit = state.FinishEpoch(ObjectLifetime::kModelStatic, 3);
  EXPECT_EQ(first_audit.shared_side_logical_openings, 1);
  EXPECT_EQ(state.AuditEpoch(ObjectLifetime::kModelStatic, 3)
                .shared_side_logical_openings,
            1);
  EXPECT_THROW(state.OpenSharedObjectOnce(first), std::exception);

  const ObjectKey updated{"model.word_embeddings", 4, 0};
  state.RegisterObject(updated, ObjectLifetime::kModelStatic, {30522, 768},
                       "model-mask-4");
  state.PrepareEpoch(ObjectLifetime::kModelStatic, 4);
  EXPECT_NO_THROW(state.OpenSharedObjectOnce(updated));
  EXPECT_NO_THROW(state.FinishEpoch(ObjectLifetime::kModelStatic, 4));
}

TEST(MJVObjectContextTest, RejectsDuplicateConsumerAndAbortsIncompleteEpoch) {
  MJVObjectContextState state;
  const auto object = RequestKey("request.hidden.0", 31);
  state.RegisterObject(object, ObjectLifetime::kRequestImmutable, {128, 768},
                       "request-mask-31");
  const auto query = Consumer(object, "layer0.q");
  const auto key = Consumer(object, "layer0.k");
  state.RegisterConsumer(query, {768, 768});
  state.RegisterConsumer(key, {768, 768});
  state.PrepareEpoch(ObjectLifetime::kRequestImmutable, 31);
  state.OpenSharedObjectOnce(object);
  state.OpenRightOperandOnce(query);
  state.ConsumeConsumer(query);

  EXPECT_THROW(state.ConsumeConsumer(query), std::exception);
  EXPECT_THROW(state.FinishEpoch(ObjectLifetime::kRequestImmutable, 31),
               std::exception);
  EXPECT_NO_THROW(
      state.AbortEpoch(ObjectLifetime::kRequestImmutable, 31));
  EXPECT_EQ(state.LiveObjectCount(), 0);
  EXPECT_EQ(state.LiveConsumerCount(), 0);
}

TEST(MJVObjectContextTest, AppendOnlyRegistrationFailsClosed) {
  MJVObjectContextState state;
  const ObjectKey cache{"request.kv.layer0", 41, 0};
  EXPECT_THROW(state.RegisterObject(cache, ObjectLifetime::kAppendOnlyUnsupported,
                                    {12, 128, 64}, "cache-mask"),
               std::exception);
}

TEST(MJVObjectContextTest, PhaseBatchMarkersAreUniquePerPlanEpoch) {
  MJVObjectContextState state;
  const std::string plan(64, 'a');
  state.RecordModelSetupBatch(plan, 7);
  state.RecordRequestPreprocessBatch(plan, 7, 19);
  const auto audit = state.AuditPhaseBatches(plan, 7, 19);
  EXPECT_EQ(audit.model_setup_rpc_calls, 1);
  EXPECT_EQ(audit.request_preprocess_rpc_calls, 1);
  EXPECT_THROW(state.RecordModelSetupBatch(plan, 7), std::exception);
  EXPECT_THROW(state.RecordRequestPreprocessBatch(plan, 7, 19),
               std::exception);
  EXPECT_NO_THROW(state.RecordRequestPreprocessBatch(plan, 7, 20));
}

TEST(MJVObjectContextTest, RuntimeAuditIsActualAndFinalizedOnce) {
  MJVObjectContextState state;
  const std::string plan(64, 'b');
  const ObjectKey model_object{"model.word_embeddings", 7, 0};
  state.RegisterObject(model_object, ObjectLifetime::kModelStatic, {5, 3},
                       "model-mask-7");
  state.PrepareEpoch(ObjectLifetime::kModelStatic, 7);
  state.OpenSharedObjectOnce(model_object);
  state.FinishEpoch(ObjectLifetime::kModelStatic, 7);

  const auto request_object = RequestKey("request.hidden.0", 19);
  state.RegisterObject(request_object, ObjectLifetime::kRequestImmutable,
                       {2, 3}, "request-mask-19");
  const auto first = Consumer(request_object, "layer0.q");
  const auto second = Consumer(request_object, "layer0.k");
  state.RegisterConsumer(first, {3, 2});
  state.RegisterConsumer(second, {3, 4});
  state.PrepareEpoch(ObjectLifetime::kRequestImmutable, 19);

  state.RecordModelSetupBatch(plan, 7);
  state.RecordRequestPreprocessBatch(plan, 7, 19);
  state.BeginPlanExecution(plan, 7, 19);
  state.RecordSecureInvoke();
  state.OpenSharedObjectOnce(request_object);
  state.OpenRightFamilyOnce({first, second});
  state.ConsumeFamilyOnce({first, second});
  state.RecordPhysicalFamilyExecution(ObjectLifetime::kRequestImmutable, 19,
                                      "qkv_direct_sum");
  state.RecordProtocolMatrixCall();
  state.RecordHeSubcall( false);
  state.FinishEpoch(ObjectLifetime::kRequestImmutable, 19);

  const auto audit = state.FinalizeRuntimeAudit(plan, 7, 19);
  EXPECT_EQ(audit.schema_version, 1);
  EXPECT_EQ(audit.plan_matches, 1);
  EXPECT_EQ(audit.model_epoch, 7);
  EXPECT_EQ(audit.request_epoch, 19);
  EXPECT_EQ(audit.logical_consumer_count, 2);
  EXPECT_EQ(audit.physical_family_count, 1);
  EXPECT_EQ(audit.model.required_shared_object_contexts, 1);
  EXPECT_EQ(audit.model.shared_side_logical_openings, 1);
  EXPECT_EQ(audit.request.required_shared_object_contexts, 1);
  EXPECT_EQ(audit.request.shared_side_logical_openings, 1);
  EXPECT_EQ(audit.request.logical_right_openings, 2);
  EXPECT_EQ(audit.request.physical_right_collectives, 1);
  EXPECT_EQ(audit.phase_batches.model_setup_rpc_calls, 1);
  EXPECT_EQ(audit.phase_batches.request_preprocess_rpc_calls, 1);
  EXPECT_EQ(audit.consumed_logical_consumers, 2);
  EXPECT_EQ(audit.expected_logical_consumers, 2);
  EXPECT_EQ(audit.fresh_request_masks, 1);
  EXPECT_EQ(audit.static_object_reused, 1);
  EXPECT_EQ(audit.secure_invoke_rpc_calls, 1);
  EXPECT_EQ(audit.protocol_matrix_calls, 1);
  EXPECT_EQ(audit.dot_ole_calls, 1);
  EXPECT_EQ(audit.batch_dot_ole_calls, 0);
  EXPECT_EQ(audit.live_object_contexts, 1);
  EXPECT_EQ(audit.live_consumer_contexts, 0);
  EXPECT_THROW(state.FinalizeRuntimeAudit(plan, 7, 19), std::exception);
  state.EndPlanExecution(plan, 7, 19);
}

TEST(MJVObjectContextTest,
     ReconcilesIndependentHighLevelAndHeSubcallCountsByBackend) {
  const std::string plan(64, 'c');

  MJVObjectContextState baseline;
  baseline.BeginPlanExecution(plan, 7, 19);
  baseline.RecordSecureInvoke();
  baseline.ReconcileLinearAudit( false,
                                75,
                                51,
                                48);
  const auto baseline_audit = baseline.FinalizeRuntimeAudit(plan, 7, 19);
  EXPECT_EQ(baseline_audit.protocol_matrix_calls, 75);
  EXPECT_EQ(baseline_audit.dot_ole_calls, 51);
  EXPECT_EQ(baseline_audit.batch_dot_ole_calls, 48);
  baseline.EndPlanExecution(plan, 7, 19);

  MJVObjectContextState ours;
  ours.BeginPlanExecution(plan, 7, 20);
  ours.RecordSecureInvoke();
  ours.RecordProtocolMatrixCall();
  ours.RecordProtocolMatrixCall();
  ours.ReconcileLinearAudit( true,
                            2,
                            0,
                            0);
  const auto ours_audit = ours.FinalizeRuntimeAudit(plan, 7, 20);
  EXPECT_EQ(ours_audit.protocol_matrix_calls, 2);
  EXPECT_EQ(ours_audit.dot_ole_calls, 0);
  EXPECT_EQ(ours_audit.batch_dot_ole_calls, 0);
  ours.EndPlanExecution(plan, 7, 20);
}

TEST(MJVObjectContextTest, RuntimeFamiliesDrivePhaseSplitOpeningCounts) {
  MJVObjectContextState state;
  const std::string plan(64, 'c');
  state.RegisterModelFamily(plan, 7, "dense.family.0", "request.hidden.0",
                            3,
                            ObjectLifetime::kRequestImmutable,
                            ObjectOperand::kLhs, {2, 3}, {3, 6},
                            "qkv_direct_sum");
  state.RegisterModelFamily(plan, 7, "request.word_embedding_batch",
                            "model.word_embeddings",
                            2,
                            ObjectLifetime::kModelStatic,
                            ObjectOperand::kRhs, {5, 3}, {2, 5},
                            "embedding_batch");
  state.RegisterRequestFamily(plan, 7, 19, "dense.family.0");
  state.RegisterRequestFamily(plan, 7, 19,
                              "request.word_embedding_batch");
  state.RegisterRequestFamily(
      plan, 7, 19, "attention.family.0", "request.attention.0",
      1, ObjectLifetime::kRequestImmutable,
      ObjectOperand::kLhs, {2, 3}, {3, 2}, "attention_matmul");
  state.RecordModelSetupBatch(plan, 7);
  state.RecordRequestPreprocessBatch(plan, 7, 19);
  state.BeginPlanExecution(plan, 7, 19);
  state.RecordSecureInvoke();
  state.RecordRuntimeFamily(plan, 7, 19, "request.word_embedding_batch");
  state.RecordRuntimeFamily(plan, 7, 19, "dense.family.0");
  state.RecordRuntimeFamily(plan, 7, 19, "attention.family.0");

  const auto audit = state.FinalizeRuntimeAudit(plan, 7, 19);
  EXPECT_EQ(audit.logical_consumer_count, 6);
  EXPECT_EQ(audit.physical_family_count, 3);
  EXPECT_EQ(audit.model.required_shared_object_contexts, 1);
  EXPECT_EQ(audit.model.shared_side_logical_openings, 1);
  EXPECT_EQ(audit.model.logical_right_openings, 3);
  EXPECT_EQ(audit.model.physical_right_collectives, 1);
  EXPECT_EQ(audit.request.required_shared_object_contexts, 2);
  EXPECT_EQ(audit.request.shared_side_logical_openings, 2);
  EXPECT_EQ(audit.request.logical_right_openings, 3);
  EXPECT_EQ(audit.request.physical_right_collectives, 2);
  EXPECT_EQ(audit.request.embedding_physical_executions, 1);
  EXPECT_EQ(audit.protocol_matrix_calls, 3);
  EXPECT_THROW(
      state.RecordRuntimeFamily(plan, 7, 19, "attention.family.0"),
      std::exception);
  state.EndPlanExecution(plan, 7, 19);
}

TEST(MJVObjectContextTest,
     FinalRequestReleasesModelContextBeforeNextModelEpoch) {
  MJVObjectContextState state;
  const std::string plan(64, 'e');

  state.RegisterModelFamily(plan, 7, "request.word_embedding_batch",
                            "model.word_embeddings.7",
                            1,
                            ObjectLifetime::kModelStatic,
                            ObjectOperand::kRhs, {5, 3}, {2, 5},
                            "embedding_batch");
  state.RegisterRequestFamily(plan, 7, 19,
                              "request.word_embedding_batch");
  state.BeginPlanExecution(plan, 7, 19);
  state.RecordRuntimeFamily(plan, 7, 19,
                            "request.word_embedding_batch");
  EXPECT_EQ(state.FinalizeRuntimeAudit(plan, 7, 19).live_object_contexts, 1);
  state.EndPlanExecution(plan, 7, 19, true);

  EXPECT_THROW(
      state.RegisterModelFamily(plan, 7, "request.word_embedding_batch",
                                "model.word_embeddings.7",
                                1,
                                ObjectLifetime::kModelStatic,
                                ObjectOperand::kRhs, {5, 3}, {2, 5},
                                "embedding_batch"),
      std::exception);

  state.RegisterModelFamily(plan, 8, "request.word_embedding_batch",
                            "model.word_embeddings.8",
                            1,
                            ObjectLifetime::kModelStatic,
                            ObjectOperand::kRhs, {5, 3}, {2, 5},
                            "embedding_batch");
  state.RegisterRequestFamily(plan, 8, 20,
                              "request.word_embedding_batch");
  state.BeginPlanExecution(plan, 8, 20);
  state.RecordRuntimeFamily(plan, 8, 20,
                            "request.word_embedding_batch");
  EXPECT_EQ(state.FinalizeRuntimeAudit(plan, 8, 20).live_object_contexts, 1);
  state.EndPlanExecution(plan, 8, 20, true);
}

TEST(MJVObjectContextTest, RuntimeAuditRejectsMissingOrCrossEpochFamily) {
  MJVObjectContextState state;
  const std::string plan(64, 'd');
  state.RegisterModelFamily(plan, 7, "dense.family.0", "request.hidden.0",
                            1, ObjectLifetime::kRequestImmutable,
                            ObjectOperand::kLhs, {2, 3}, {3, 2},
                            "other_dense");
  state.RegisterRequestFamily(plan, 7, 19, "dense.family.0");
  state.BeginPlanExecution(plan, 7, 19);
  EXPECT_THROW(state.RecordRuntimeFamily(plan, 7, 20, "dense.family.0"),
               std::exception);
  EXPECT_THROW(state.FinalizeRuntimeAudit(plan, 7, 19), std::exception);
  state.AbortPlanExecution();
}

TEST(MJVObjectContextTest, DirectSumMaskedAlgebraMatchesCombinedDenseProduct) {
  const auto field = FieldType::FM64;
  const auto object = ring_rand(field, {4, 3});
  const auto combined_right = ring_rand(field, {3, 6});
  const auto object_mask = ring_rand(field, object.shape());
  const auto right_mask = ring_rand(field, combined_right.shape());
  const auto gamma = ring_sub(object, object_mask);
  const auto delta = ring_sub(combined_right, right_mask);
  const auto correlation = ring_mmul(object_mask, right_mask);

  const auto output = EvaluateMaskedMatmul(
      correlation, object_mask, right_mask, gamma, delta,
      ObjectOperand::kLhs, true);

  ExpectRingEqual(output, ring_mmul(object, combined_right));
  ExpectRingEqual(output.slice({0, 0}, {4, 2}, {1, 1}),
                  ring_mmul(object, combined_right.slice(
                                        {0, 0}, {3, 2}, {1, 1})));
  ExpectRingEqual(output.slice({0, 2}, {4, 4}, {1, 1}),
                  ring_mmul(object, combined_right.slice(
                                        {0, 2}, {3, 4}, {1, 1})));
  ExpectRingEqual(output.slice({0, 4}, {4, 6}, {1, 1}),
                  ring_mmul(object, combined_right.slice(
                                        {0, 4}, {3, 6}, {1, 1})));
}

TEST(MJVObjectContextTest, ModelObjectOnRhsMatchesOneBatchedEmbeddingProduct) {
  const auto field = FieldType::FM64;
  const auto selectors = ring_rand(field, {8, 17});
  const auto embedding = ring_rand(field, {17, 5});
  const auto selector_mask = ring_rand(field, selectors.shape());
  const auto embedding_mask = ring_rand(field, embedding.shape());
  const auto selector_delta = ring_sub(selectors, selector_mask);
  const auto embedding_gamma = ring_sub(embedding, embedding_mask);
  const auto correlation = ring_mmul(selector_mask, embedding_mask);

  const auto rows = EvaluateMaskedMatmul(
      correlation, embedding_mask, selector_mask, embedding_gamma,
      selector_delta, ObjectOperand::kRhs,
      true);

  ExpectRingEqual(rows, ring_mmul(selectors, embedding));
}

TEST(MJVObjectContextTest,
     EmbeddingLogicalRowsUseOneRightCollectiveAndOnePhysicalExecution) {
  MJVObjectContextState state;
  const ObjectKey embedding{"model.word_embeddings", 8, 0};
  state.RegisterObject(embedding, ObjectLifetime::kModelStatic, {30522, 768},
                       "model-mask-8");
  state.PrepareEpoch(ObjectLifetime::kModelStatic, 8);
  state.OpenSharedObjectOnce(embedding);
  EXPECT_EQ(state.FinishEpoch(ObjectLifetime::kModelStatic, 8)
                .shared_side_logical_openings,
            1);

  std::vector<ConsumerKey> rows;
  for (int64_t row = 0; row < 128; ++row) {
    rows.push_back(Consumer(embedding, "word_embedding.row." +
                                           std::to_string(row)));
    state.RegisterConsumer(rows.back(), {1, 30522},
                           ObjectLifetime::kRequestImmutable, 51);
  }
  state.PrepareEpoch(ObjectLifetime::kRequestImmutable, 51);
  state.OpenRightFamilyOnce(rows);
  state.ConsumeFamilyOnce(rows);
  state.RecordPhysicalFamilyExecution(ObjectLifetime::kRequestImmutable, 51,
                                      "embedding_batch");
  const auto request_audit =
      state.FinishEpoch(ObjectLifetime::kRequestImmutable, 51);

  EXPECT_EQ(request_audit.required_shared_object_contexts, 0);
  EXPECT_EQ(request_audit.shared_side_logical_openings, 0);
  EXPECT_EQ(request_audit.logical_right_openings, 128);
  EXPECT_EQ(request_audit.physical_right_collectives, 1);
  EXPECT_EQ(request_audit.embedding_physical_executions, 1);
  EXPECT_EQ(request_audit.physical_family_executions, 1);
  EXPECT_THROW(state.ConsumeFamilyOnce(rows), std::exception);
}

}  
}
