// Copyright 2026

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/mjv.h"
#include "libspu/mpc/cheetah/mjv_linear.h"
#include "libspu/mpc/cheetah/mjv_object_context.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/semi2k/protocol.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
namespace {

RuntimeConfig MakeConfig(ProtocolKind protocol) {
  RuntimeConfig config;
  config.set_protocol(protocol);
  config.set_field(FieldType::FM64);
  if (protocol == ProtocolKind::CHEETAH) {
    config.mutable_cheetah_2pc_config()->set_ot_kind(
        CheetahOtKind::YACL_Softspoken);
  }
  return config;
}

void ExpectRingEqual(const NdArrayRef& actual, const NdArrayRef& expected) {
  ASSERT_EQ(actual.shape(), expected.shape());
  DISPATCH_ALL_FIELDS(FieldType::FM64, "mjv-test", [&]() {
    auto actual_view = NdArrayView<ring2k_t>(actual);
    auto expected_view = NdArrayView<ring2k_t>(expected);
    for (int64_t index = 0; index < expected.numel(); ++index) {
      EXPECT_EQ(actual_view[index], expected_view[index]);
    }
  });
}

TEST(MJVRegistrationTest, IsIsolatedToCheetah) {
  utils::simulate(2, [](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto cheetah = makeCheetahProtocol(MakeConfig(ProtocolKind::CHEETAH), lctx);
    EXPECT_TRUE(cheetah->hasKernel("mjv_model_setup"));
    EXPECT_TRUE(cheetah->hasKernel("mjv_request_preprocess"));
    EXPECT_TRUE(cheetah->hasKernel("mjv_group_online"));
    EXPECT_TRUE(cheetah->hasKernel("mmul_aa"));
  });

  utils::simulate(2, [](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto semi2k = makeSemi2kProtocol(MakeConfig(ProtocolKind::SEMI2K), lctx);
    EXPECT_FALSE(semi2k->hasKernel("mjv_model_setup"));
    EXPECT_FALSE(semi2k->hasKernel("mjv_request_preprocess"));
    EXPECT_FALSE(semi2k->hasKernel("mjv_group_online"));
  });
}

TEST(MJVStaticContextTest, UsesFreshRequestMasksAndConsumesOnce) {
  constexpr int64_t kModelId = 7;
  constexpr int64_t kRequest1 = 101;
  constexpr int64_t kRequest2 = 102;
  const Sizes widths = {2, 1};
  constexpr int64_t kSumR = 6;
  const auto field = FieldType::FM64;
  const auto weight = ring_rand(field, {3, 3});
  const auto activation1 = ring_rand(field, {2, 3});
  const auto activation2 = ring_rand(field, {2, 3});
  const auto zero_weight = ring_zeros(field, weight.shape());
  const auto zero_activation = ring_zeros(field, activation1.shape());

  std::vector<NdArrayRef> request_mask1(2);
  std::vector<NdArrayRef> request_mask2(2);
  std::vector<std::vector<NdArrayRef>> output1(2);
  std::vector<std::vector<NdArrayRef>> output2(2);

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeConfig(ProtocolKind::CHEETAH), lctx);
    const auto weight_share = Value(
        (rank == 0 ? weight : zero_weight).as(makeType<AShrTy>(field)),
        DT_I64);
    const auto activation_share1 = Value(
        (rank == 0 ? zero_activation : activation1)
            .as(makeType<AShrTy>(field)),
        DT_I64);
    const auto activation_share2 = Value(
        (rank == 0 ? zero_activation : activation2)
            .as(makeType<AShrTy>(field)),
        DT_I64);

    EXPECT_THROW(dynDispatch<Value>(ctx.get(), "mjv_request_preprocess",
                                    weight_share, kModelId, kRequest1,
                                    int64_t{2}),
                 std::exception);
    const auto model_handle = dynDispatch<Value>(
        ctx.get(), "mjv_model_setup", weight_share, kModelId, widths, kSumR);
    EXPECT_THROW(dynDispatch<Value>(ctx.get(), "mjv_model_setup", weight_share,
                                    kModelId, widths, kSumR),
                 std::exception);

    const auto handle1 = dynDispatch<Value>(
        ctx.get(), "mjv_request_preprocess", model_handle, kModelId,
        kRequest1, int64_t{2});
    const auto handle2 = dynDispatch<Value>(
        ctx.get(), "mjv_request_preprocess", model_handle, kModelId,
        kRequest2, int64_t{2});
    auto* mjv_state = ctx->getState<MJVContextState>();
    EXPECT_EQ(mjv_state->LiveRequestCount(), 2);
    request_mask1[rank] = handle1.data();
    request_mask2[rank] = handle2.data();
    EXPECT_THROW(dynDispatch<Value>(ctx.get(), "mjv_request_preprocess",
                                    model_handle, kModelId, kRequest1,
                                    int64_t{2}),
                 std::exception);

    auto result1 = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_group_online", handle1, activation_share1, kModelId,
        kRequest1);
    EXPECT_EQ(mjv_state->LiveRequestCount(), 1);
    auto result2 = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_group_online", handle2, activation_share2, kModelId,
        kRequest2);
    EXPECT_EQ(mjv_state->LiveRequestCount(), 0);
    ASSERT_EQ(result1.size(), widths.size());
    ASSERT_EQ(result2.size(), widths.size());
    for (size_t index = 0; index < widths.size(); ++index) {
      output1[rank].push_back(result1[index].data());
      output2[rank].push_back(result2[index].data());
    }
    EXPECT_THROW(dynDispatch<std::vector<Value>>(
                     ctx.get(), "mjv_group_online", handle1,
                     activation_share1, kModelId, kRequest1),
                 std::exception);
    EXPECT_THROW(dynDispatch<Value>(ctx.get(), "mjv_request_preprocess",
                                    model_handle, kModelId, kRequest1,
                                    int64_t{2}),
                 std::exception);

    const auto stats = mjv_state->GetStats();
    EXPECT_EQ(stats.model_setup_count, 1);
    EXPECT_EQ(stats.request_preprocess_count, 2);
    EXPECT_EQ(stats.online_count, 2);
    EXPECT_EQ(stats.right_operand_opened_elements, 9);
    EXPECT_EQ(stats.joint_view_opened_elements, 12);
    EXPECT_EQ(stats.correlation_elements, 12);
  });

  const auto mask1 = ring_add(request_mask1[0], request_mask1[1]);
  const auto mask2 = ring_add(request_mask2[0], request_mask2[1]);
  DISPATCH_ALL_FIELDS(field, "mjv-mask", [&]() {
    auto first = NdArrayView<ring2k_t>(mask1);
    auto second = NdArrayView<ring2k_t>(mask2);
    bool any_difference = false;
    for (int64_t index = 0; index < mask1.numel(); ++index) {
      any_difference |= first[index] != second[index];
    }
    EXPECT_TRUE(any_difference);
  });

  const auto expected1 = ring_mmul(activation1, weight);
  const auto expected2 = ring_mmul(activation2, weight);
  int64_t offset = 0;
  for (size_t consumer = 0; consumer < widths.size(); ++consumer) {
    const int64_t end = offset + widths[consumer];
    const auto expected_slice1 =
        expected1.slice({0, offset}, {2, end}, {1, 1});
    const auto expected_slice2 =
        expected2.slice({0, offset}, {2, end}, {1, 1});
    ExpectRingEqual(ring_add(output1[0][consumer], output1[1][consumer]),
                    expected_slice1);
    ExpectRingEqual(ring_add(output2[0][consumer], output2[1][consumer]),
                    expected_slice2);
    offset = end;
  }
}

TEST(MJVStaticContextTest, SingleConsumerEmitsNativeReplacementFingerprint) {
  constexpr int64_t kModelId = 17;
  constexpr int64_t kRequestId = 201;
  const auto field = FieldType::FM64;
  const auto weight = ring_rand(field, {3, 2});
  const auto activation = ring_rand(field, {2, 3});
  const auto zero_weight = ring_zeros(field, weight.shape());
  const auto zero_activation = ring_zeros(field, activation.shape());
  std::vector<NdArrayRef> outputs(2);

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeConfig(ProtocolKind::CHEETAH), lctx);
    const auto weight_share = Value(
        (rank == 0 ? weight : zero_weight).as(makeType<AShrTy>(field)),
        DT_I64);
    const auto activation_share = Value(
        (rank == 0 ? activation : zero_activation)
            .as(makeType<AShrTy>(field)),
        DT_I64);
    const Sizes widths = {2};
    const auto model_handle = dynDispatch<Value>(
        ctx.get(), "mjv_model_setup", weight_share, kModelId, widths,
        int64_t{3});
    const auto request_handle = dynDispatch<Value>(
        ctx.get(), "mjv_request_preprocess", model_handle, kModelId,
        kRequestId, int64_t{2});

    auto* audit = ctx->getState<MJVLinearAuditState>();
    audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    const auto result = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_group_online", request_handle, activation_share,
        kModelId, kRequestId);
    const auto snapshot = audit->End();

    ASSERT_EQ(result.size(), 1);
    outputs[rank] = result[0].data();
    ASSERT_EQ(snapshot.high_level_calls.size(), 1);
    EXPECT_EQ(snapshot.high_level_calls[0].family,
              LinearKernelFamily::kMatMulAA);
    EXPECT_EQ(snapshot.high_level_calls[0].lhs_shape, Shape({2, 3}));
    EXPECT_EQ(snapshot.high_level_calls[0].rhs_shape, Shape({3, 2}));
    EXPECT_EQ(snapshot.high_level_calls[0].output_shape, Shape({2, 2}));
    EXPECT_EQ(snapshot.dot_ole_calls, 0);
    EXPECT_EQ(snapshot.batch_dot_ole_calls, 0);
  });

  ExpectRingEqual(ring_add(outputs[0], outputs[1]),
                  ring_mmul(activation, weight));
}

TEST(MJVStaticContextTest,
     EmbeddingBatchKeepsFourLogicalMembersAndOnePhysicalFingerprint) {
  const auto field = FieldType::FM64;
  const auto weight = ring_rand(field, {3, 2});
  const auto activation = ring_rand(field, {4, 3});
  const auto zero_weight = ring_zeros(field, weight.shape());
  const auto zero_activation = ring_zeros(field, activation.shape());
  const std::string plan(64, 'd');

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeConfig(ProtocolKind::CHEETAH), lctx);
    const auto weight_share = Value(
        (rank == 0 ? weight : zero_weight).as(makeType<AShrTy>(field)),
        DT_I64);
    const auto activation_share = Value(
        (rank == 0 ? activation : zero_activation)
            .as(makeType<AShrTy>(field)),
        DT_I64);
    const Sizes widths = {2};
    const auto model_handle = dynDispatch<Value>(
        ctx.get(), "mjv_model_setup", weight_share, int64_t{27}, widths,
        int64_t{3}, static_cast<int64_t>(LinearKernelFamily::kMatMulVVS),
        plan, int64_t{270}, std::string("request.word_embedding_batch"),
        std::string("model.word_embeddings"), int64_t{4},
        static_cast<int64_t>(ObjectLifetime::kModelStatic),
        static_cast<int64_t>(ObjectOperand::kRhs),
        std::string("embedding_batch"), Shape({3, 2}), Shape({4, 3}));
    const auto request_handle = dynDispatch<Value>(
        ctx.get(), "mjv_request_preprocess", model_handle, int64_t{27},
        int64_t{28}, int64_t{4}, plan, int64_t{270}, int64_t{280},
        std::string("request.word_embedding_batch"));

    auto* audit = ctx->getState<MJVLinearAuditState>();
    auto* mjv_state = ctx->getState<MJVContextState>();
    auto* object_state = ctx->getState<MJVObjectContextState>();
    EXPECT_THROW(mjv_state->ReleaseModelEpoch(plan, 270), std::exception);
    object_state->RecordModelSetupBatch(plan, 270);
    object_state->RecordRequestPreprocessBatch(plan, 270, 280);
    object_state->BeginPlanExecution(plan, 270, 280);
    object_state->RecordSecureInvoke();
    audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_group_online", request_handle, activation_share,
        int64_t{27}, int64_t{28}, int64_t{4});
    const auto snapshot = audit->End();
    const auto object_audit =
        object_state->FinalizeRuntimeAudit(plan, 270, 280);

    ASSERT_EQ(snapshot.high_level_calls.size(), 1);
    EXPECT_EQ(snapshot.high_level_calls[0].lhs_shape, Shape({4, 3}));
    EXPECT_EQ(snapshot.high_level_calls[0].rhs_shape, Shape({3, 2}));
    EXPECT_EQ(snapshot.high_level_calls[0].output_shape, Shape({4, 2}));
    EXPECT_EQ(snapshot.dot_ole_calls, 0);
    EXPECT_EQ(snapshot.batch_dot_ole_calls, 0);
    EXPECT_EQ(object_audit.logical_consumer_count, 4);
    EXPECT_EQ(object_audit.consumed_logical_consumers, 4);
    EXPECT_EQ(object_audit.expected_logical_consumers, 4);
    EXPECT_EQ(object_audit.physical_family_count, 1);
    EXPECT_EQ(object_audit.protocol_matrix_calls, 1);
    EXPECT_EQ(object_audit.model.required_shared_object_contexts, 1);
    EXPECT_EQ(object_audit.model.shared_side_logical_openings, 1);
    EXPECT_EQ(object_audit.request.logical_right_openings, 4);
    EXPECT_EQ(object_audit.request.physical_right_collectives, 1);
    EXPECT_EQ(object_audit.request.embedding_physical_executions, 1);
    object_state->EndPlanExecution(plan, 270, 280,
                                   true);
    mjv_state->ReleaseModelEpoch(plan, 270);
    EXPECT_THROW(mjv_state->GetModel(27), std::exception);
    EXPECT_THROW(mjv_state->RequireNewModel(27), std::exception);
  });
}

TEST(MJVStaticContextTest,
     QkvDirectSumKeepsThreeLogicalMembersAndOnePhysicalFingerprint) {
  const auto field = FieldType::FM64;
  const auto weight = ring_rand(field, {3, 6});
  const auto activation = ring_rand(field, {2, 3});
  const auto zero_weight = ring_zeros(field, weight.shape());
  const auto zero_activation = ring_zeros(field, activation.shape());
  const std::string plan(64, 'e');

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeConfig(ProtocolKind::CHEETAH), lctx);
    const auto weight_share = Value(
        (rank == 0 ? weight : zero_weight).as(makeType<AShrTy>(field)),
        DT_I64);
    const auto activation_share = Value(
        (rank == 0 ? zero_activation : activation)
            .as(makeType<AShrTy>(field)),
        DT_I64);
    auto* linear_audit = ctx->getState<MJVLinearAuditState>();
    auto* linear_scope = ctx->getState<MJVLinearScopeState>();
    linear_scope->Begin(17, plan, LinearBackend::kBumbleBee);
    linear_audit->Begin(LinearPhase::kOnline, LinearBackend::kBumbleBee);
    dynDispatch<Value>(ctx.get(), "mmul_aa", activation_share, weight_share);
    const auto baseline = linear_audit->End();
    linear_scope->Finish(17, baseline);
    ASSERT_EQ(baseline.high_level_calls.size(), 1);
    EXPECT_EQ(baseline.high_level_calls[0].lhs_shape, Shape({2, 3}));
    EXPECT_EQ(baseline.high_level_calls[0].rhs_shape, Shape({3, 6}));
    EXPECT_EQ(baseline.high_level_calls[0].output_shape, Shape({2, 6}));

    const Sizes widths = {2, 2, 2};
    const auto model_handle = dynDispatch<Value>(
        ctx.get(), "mjv_model_setup", weight_share, int64_t{7}, widths,
        int64_t{9}, static_cast<int64_t>(LinearKernelFamily::kMatMulAA),
        plan, int64_t{70}, std::string("dense.family.0"),
        std::string("request.hidden.0"), int64_t{3},
        static_cast<int64_t>(ObjectLifetime::kRequestImmutable),
        static_cast<int64_t>(ObjectOperand::kLhs),
        std::string("qkv_direct_sum"), Shape({2, 3}), Shape({3, 6}));
    const auto request_handle = dynDispatch<Value>(
        ctx.get(), "mjv_request_preprocess", model_handle, int64_t{7},
        int64_t{19}, int64_t{2}, plan, int64_t{70}, int64_t{190},
        std::string("dense.family.0"));

    auto* object_state = ctx->getState<MJVObjectContextState>();
    object_state->RecordModelSetupBatch(plan, 70);
    object_state->RecordRequestPreprocessBatch(plan, 70, 190);
    object_state->BeginPlanExecution(plan, 70, 190);
    object_state->RecordSecureInvoke();
    linear_scope->Begin(18, plan, LinearBackend::kOurs);
    linear_audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    const auto outputs = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_group_online", request_handle, activation_share,
        int64_t{7}, int64_t{19});
    const auto replacement = linear_audit->End();
    linear_scope->Finish(18, replacement);
    const auto runtime_audit =
        object_state->FinalizeRuntimeAudit(plan, 70, 190);

    ASSERT_EQ(outputs.size(), 3);
    for (const auto& output : outputs) {
      EXPECT_EQ(output.shape(), Shape({2, 2}));
    }
    ASSERT_EQ(replacement.high_level_calls.size(), 1);
    EXPECT_EQ(replacement.high_level_calls[0], baseline.high_level_calls[0]);
    EXPECT_EQ(replacement.dot_ole_calls, 0);
    EXPECT_EQ(replacement.batch_dot_ole_calls, 0);
    EXPECT_EQ(runtime_audit.logical_consumer_count, 3);
    EXPECT_EQ(runtime_audit.consumed_logical_consumers, 3);
    EXPECT_EQ(runtime_audit.expected_logical_consumers, 3);
    EXPECT_EQ(runtime_audit.physical_family_count, 1);
    EXPECT_EQ(runtime_audit.model.logical_right_openings, 3);
    EXPECT_EQ(runtime_audit.model.physical_right_collectives, 1);
    EXPECT_EQ(runtime_audit.request.required_shared_object_contexts, 1);
    EXPECT_EQ(runtime_audit.request.shared_side_logical_openings, 1);
    EXPECT_EQ(runtime_audit.request.logical_right_openings, 0);
    EXPECT_EQ(runtime_audit.protocol_matrix_calls, 1);
    object_state->EndPlanExecution(plan, 70, 190);
  });
}

TEST(MJVStaticContextTest, EmitsDeclaredBaselineKernelFamily) {
  constexpr int64_t kModelId = 18;
  constexpr int64_t kRequestId = 202;
  const auto field = FieldType::FM64;
  const auto weight = ring_rand(field, {3, 2});
  const auto activation = ring_rand(field, {2, 3});
  const auto zero_weight = ring_zeros(field, weight.shape());
  const auto zero_activation = ring_zeros(field, activation.shape());

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeConfig(ProtocolKind::CHEETAH), lctx);
    const auto weight_share = Value(
        (rank == 0 ? weight : zero_weight).as(makeType<AShrTy>(field)),
        DT_I64);
    const auto activation_share = Value(
        (rank == 0 ? activation : zero_activation)
            .as(makeType<AShrTy>(field)),
        DT_I64);
    const Sizes widths = {2};
    const auto model_handle = dynDispatch<Value>(
        ctx.get(), "mjv_model_setup", weight_share, kModelId, widths,
        int64_t{3}, static_cast<int64_t>(LinearKernelFamily::kMatMulAV));
    const auto request_handle = dynDispatch<Value>(
        ctx.get(), "mjv_request_preprocess", model_handle, kModelId,
        kRequestId, int64_t{2});

    auto* audit = ctx->getState<MJVLinearAuditState>();
    audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    dynDispatch<std::vector<Value>>(ctx.get(), "mjv_group_online",
                                    request_handle, activation_share, kModelId,
                                    kRequestId);
    const auto snapshot = audit->End();

    ASSERT_EQ(snapshot.high_level_calls.size(), 1);
    EXPECT_EQ(snapshot.high_level_calls[0].family,
              LinearKernelFamily::kMatMulAV);
  });
}

}  
}
