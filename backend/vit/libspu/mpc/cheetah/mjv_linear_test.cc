// Copyright 2026

#include "gtest/gtest.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/mjv_linear.h"
#include "libspu/mpc/cheetah/mjv_object_context.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
namespace {

RuntimeConfig MakeAuditConfig() {
  RuntimeConfig config;
  config.set_protocol(ProtocolKind::CHEETAH);
  config.set_field(FieldType::FM64);
  config.mutable_cheetah_2pc_config()->set_ot_kind(
      CheetahOtKind::YACL_Softspoken);
  return config;
}

void ExpectRingEqual(const NdArrayRef& actual, const NdArrayRef& expected) {
  ASSERT_EQ(actual.shape(), expected.shape());
  DISPATCH_ALL_FIELDS(FieldType::FM64, "mjv-linear-test", [&]() {
    auto actual_view = NdArrayView<ring2k_t>(actual);
    auto expected_view = NdArrayView<ring2k_t>(expected);
    for (int64_t index = 0; index < expected.numel(); ++index) {
      EXPECT_EQ(actual_view[index], expected_view[index]);
    }
  });
}

NdArrayRef BatchRingMatMul(const NdArrayRef& lhs, const NdArrayRef& rhs) {
  const auto batch = lhs.shape()[0];
  const auto rows = lhs.shape()[1];
  const auto width = rhs.shape()[2];
  NdArrayRef output(lhs.eltype(), {batch, rows, width});
  for (int64_t index = 0; index < batch; ++index) {
    auto lhs_slice = lhs.slice({index, 0, 0},
                               {index + 1, rows, lhs.shape()[2]}, {1, 1, 1})
                         .reshape({rows, lhs.shape()[2]});
    auto rhs_slice = rhs.slice({index, 0, 0},
                               {index + 1, rhs.shape()[1], width}, {1, 1, 1})
                         .reshape({rhs.shape()[1], width});
    auto output_slice = output.slice({index, 0, 0},
                                     {index + 1, rows, width}, {1, 1, 1})
                            .reshape({rows, width});
    ring_mmul_(output_slice, lhs_slice, rhs_slice);
  }
  return output;
}

TEST(MJVLinearAuditTest, RealMatMulRecordsHighLevelAndHeSubcalls) {
  utils::simulate(2, [](const std::shared_ptr<yacl::link::Context>& lctx) {
    auto ctx = makeCheetahProtocol(MakeAuditConfig(), lctx);
    auto* audit = ctx->getState<MJVLinearAuditState>();
    audit->Begin(LinearPhase::kOnline, LinearBackend::kBumbleBee);

    const auto field = FieldType::FM64;
    const auto lhs = Value(
        ring_zeros(field, {2, 3}).as(makeType<AShrTy>(field)), DT_I64);
    const auto rhs = Value(
        ring_zeros(field, {3, 4}).as(makeType<AShrTy>(field)), DT_I64);
    const auto output =
        dynDispatch<Value>(ctx.get(), "mmul_aa", lhs, rhs);
    EXPECT_EQ(output.shape(), Shape({2, 4}));

    const auto snapshot = audit->End();
    ASSERT_EQ(snapshot.high_level_calls.size(), 1);
    EXPECT_EQ(snapshot.high_level_calls[0].ordinal, 0);
    EXPECT_EQ(snapshot.high_level_calls[0].family,
              LinearKernelFamily::kMatMulAA);
    EXPECT_EQ(snapshot.high_level_calls[0].lhs_shape, Shape({2, 3}));
    EXPECT_EQ(snapshot.high_level_calls[0].rhs_shape, Shape({3, 4}));
    EXPECT_EQ(snapshot.high_level_calls[0].output_shape, Shape({2, 4}));
    EXPECT_EQ(snapshot.dot_ole_calls, 2);
    EXPECT_EQ(snapshot.batch_dot_ole_calls, 0);
  });
}

TEST(MJVLinearAuditTest, OursOnlineRejectsHeSubcalls) {
  MJVLinearAuditState audit;
  audit.Begin(LinearPhase::kOnline, LinearBackend::kOurs);
  EXPECT_THROW(audit.RecordHeSubcall( false), std::exception);
}

TEST(MJVLinearAuditTest, OursAllowsOnlyDeclaredCommonFallbackHe) {
  MJVLinearAuditState audit;
  audit.Begin(LinearPhase::kOnline, LinearBackend::kOurs,
              1,
              0);
  EXPECT_NO_THROW(audit.RecordHeSubcall( false));
  EXPECT_THROW(audit.RecordHeSubcall( false), std::exception);
  const auto snapshot = audit.End();
  EXPECT_EQ(snapshot.dot_ole_calls, 1);
  EXPECT_EQ(snapshot.batch_dot_ole_calls, 0);
}

TEST(MJVLinearAuditTest, VerifiesExactOrderedFingerprintParity) {
  const LinearCallFingerprint first{
      0, LinearKernelFamily::kMatMulAA, Shape({2, 3}), Shape({3, 4}),
      Shape({2, 4})};
  const LinearCallFingerprint second{
      1, LinearKernelFamily::kBatchMatMulAA, Shape({5, 2, 3}),
      Shape({5, 3, 4}), Shape({5, 2, 4})};
  const std::vector<LinearCallFingerprint> baseline = {first, second};
  EXPECT_NO_THROW(VerifyLinearPathParity(baseline, baseline));

  auto reordered = baseline;
  std::swap(reordered[0], reordered[1]);
  EXPECT_THROW(VerifyLinearPathParity(baseline, reordered), std::exception);
}

TEST(MJVLinearAuditTest, OursConsumesFamilyFromCapturedBaseline) {
  const Shape lhs_shape = {2, 3};
  const Shape rhs_shape = {3, 4};
  const Shape output_shape = {2, 4};
  LinearAuditSnapshot baseline;
  baseline.phase = LinearPhase::kOnline;
  baseline.backend = LinearBackend::kBumbleBee;
  baseline.high_level_calls.push_back(
      {0, LinearKernelFamily::kMatMulAV, lhs_shape, rhs_shape, output_shape});

  MJVLinearScopeState scope;
  scope.Begin(7, "manifest", LinearBackend::kBumbleBee);
  scope.Finish(7, baseline);
  scope.Begin(8, "manifest", LinearBackend::kOurs);
  EXPECT_EQ(scope.ConsumeExpectedFamily(lhs_shape, rhs_shape, output_shape),
            LinearKernelFamily::kMatMulAV);

  auto replacement = baseline;
  replacement.backend = LinearBackend::kOurs;
  EXPECT_NO_THROW(scope.Finish(8, replacement));
}

TEST(MJVLinearAuditTest, OursConsumesOneDeclaredCommonFallbackPrefix) {
  LinearAuditSnapshot baseline;
  baseline.phase = LinearPhase::kOnline;
  baseline.backend = LinearBackend::kBumbleBee;
  baseline.high_level_calls = {
      {0, LinearKernelFamily::kMatMulVVS, Shape({196, 768}),
       Shape({768, 768}), Shape({196, 768})},
      {1, LinearKernelFamily::kMatMulAV, Shape({197, 768}),
       Shape({768, 2304}), Shape({197, 2304})},
  };

  MJVLinearScopeState scope;
  scope.Begin(17, "vit-manifest", LinearBackend::kBumbleBee);
  scope.Finish(17, baseline);
  scope.Begin(18, "vit-manifest", LinearBackend::kOurs,
              1,
              1,
              0);
  EXPECT_EQ(scope.ConsumeExpectedFamily(Shape({197, 768}), Shape({768, 2304}),
                                        Shape({197, 2304})),
            LinearKernelFamily::kMatMulAV);

  auto replacement = baseline;
  replacement.backend = LinearBackend::kOurs;
  replacement.dot_ole_calls = 1;
  EXPECT_NO_THROW(scope.Finish(18, replacement));
}

TEST(MJVLinearAuditTest,
     OrdinalZeroMismatchReportsExpectedAndObservedBertQkvShapes) {
  const std::string manifest = "ordinal0-bert-qkv";
  LinearAuditSnapshot baseline;
  baseline.phase = LinearPhase::kOnline;
  baseline.backend = LinearBackend::kBumbleBee;
  baseline.high_level_calls.push_back(
      {0, LinearKernelFamily::kMatMulAV, Shape({128, 768}),
       Shape({768, 2304}), Shape({128, 2304})});

  MJVLinearScopeState scope;
  scope.Begin(70, manifest, LinearBackend::kBumbleBee);
  scope.Finish(70, baseline);
  scope.Begin(71, manifest, LinearBackend::kOurs);

  try {
    scope.ConsumeExpectedFamily(Shape({128, 768}), Shape({768, 768}),
                                Shape({128, 768}));
    FAIL() << "split QKV fingerprint unexpectedly matched fused baseline";
  } catch (const std::exception& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("manifest=ordinal0-bert-qkv"), std::string::npos);
    EXPECT_NE(message.find("ordinal=0"), std::string::npos);
    EXPECT_NE(message.find("expected family=MatMulAV"), std::string::npos);
    EXPECT_NE(message.find("lhs=128x768"), std::string::npos);
    EXPECT_NE(message.find("rhs=768x2304"), std::string::npos);
    EXPECT_NE(message.find("output=128x2304"), std::string::npos);
    EXPECT_NE(message.find("observed lhs=128x768"), std::string::npos);
    EXPECT_NE(message.find("rhs=768x768"), std::string::npos);
    EXPECT_NE(message.find("output=128x768"), std::string::npos);
  }
}

TEST(MJVDynamicLinearTest, RankTwoIsFreshCorrectAndConsumedOnce) {
  constexpr int64_t kRequest1 = 301;
  constexpr int64_t kRequest2 = 302;
  const std::string kPlan(64, 'a');
  const std::string kConsumer = "attention.4";
  const auto field = FieldType::FM64;
  const auto lhs = ring_rand(field, {5, 3});
  const auto rhs = ring_rand(field, {3, 2});
  const auto zero_lhs = ring_zeros(field, lhs.shape());
  const auto zero_rhs = ring_zeros(field, rhs.shape());
  std::vector<NdArrayRef> outputs(2);
  std::vector<NdArrayRef> first_masks(2);
  std::vector<NdArrayRef> second_masks(2);

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeAuditConfig(), lctx);
    const auto lhs_share = Value(
        (rank == 0 ? lhs : zero_lhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto rhs_share = Value(
        (rank == 0 ? rhs : zero_rhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto first = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_consumer_preprocess", kPlan, kRequest1, kConsumer,
        lhs.shape(), rhs.shape(), false);
    const auto second = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_consumer_preprocess", kPlan, kRequest2, kConsumer,
        lhs.shape(), rhs.shape(), false);
    auto* dynamic_state = ctx->getState<MJVDynamicLinearState>();
    EXPECT_EQ(dynamic_state->LiveCorrelationCount(), 2);
    ASSERT_EQ(first.size(), 2);
    ASSERT_EQ(second.size(), 2);
    first_masks[rank] = first[0].data();
    second_masks[rank] = second[0].data();

    auto* audit = ctx->getState<MJVLinearAuditState>();
    audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    const auto result = dynDispatch<Value>(
        ctx.get(), "mjv_dynamic_online", first[0], first[1], lhs_share,
        rhs_share, kPlan, kRequest1, kConsumer);
    EXPECT_EQ(dynamic_state->LiveCorrelationCount(), 1);
    const auto snapshot = audit->End();
    outputs[rank] = result.data();
    ASSERT_EQ(snapshot.high_level_calls.size(), 1);
    EXPECT_EQ(snapshot.high_level_calls[0].family,
              LinearKernelFamily::kMatMulAA);
    EXPECT_EQ(snapshot.dot_ole_calls, 0);
    EXPECT_EQ(snapshot.batch_dot_ole_calls, 0);
    EXPECT_THROW(dynDispatch<Value>(
                     ctx.get(), "mjv_dynamic_online", first[0], first[1],
                     lhs_share, rhs_share, kPlan, kRequest1, kConsumer),
                 std::exception);
    EXPECT_THROW(dynDispatch<std::vector<Value>>(
                     ctx.get(), "mjv_consumer_preprocess", kPlan, kRequest1,
                     kConsumer, lhs.shape(), rhs.shape(), false),
                 std::exception);
    dynDispatch<Value>(ctx.get(), "mjv_dynamic_online", second[0], second[1],
                       lhs_share, rhs_share, kPlan, kRequest2, kConsumer);
    EXPECT_EQ(dynamic_state->LiveCorrelationCount(), 0);
  });

  ExpectRingEqual(ring_add(outputs[0], outputs[1]), ring_mmul(lhs, rhs));
  const auto first_mask = ring_add(first_masks[0], first_masks[1]);
  const auto second_mask = ring_add(second_masks[0], second_masks[1]);
  bool any_difference = false;
  DISPATCH_ALL_FIELDS(field, "mjv-dynamic-freshness", [&]() {
    auto first_view = NdArrayView<ring2k_t>(first_mask);
    auto second_view = NdArrayView<ring2k_t>(second_mask);
    for (int64_t index = 0; index < first_mask.numel(); ++index) {
      any_difference |= first_view[index] != second_view[index];
    }
  });
  EXPECT_TRUE(any_difference);
}

TEST(MJVDynamicLinearTest, RankThreeMatchesBatchRingMatMulWithoutOnlineHe) {
  constexpr int64_t kRequestId = 401;
  const std::string kPlan(64, 'b');
  const std::string kConsumer = "attention.9";
  const auto field = FieldType::FM64;
  const auto lhs = ring_rand(field, {3, 2, 4});
  const auto rhs = ring_rand(field, {3, 4, 2});
  const auto zero_lhs = ring_zeros(field, lhs.shape());
  const auto zero_rhs = ring_zeros(field, rhs.shape());
  std::vector<NdArrayRef> outputs(2);

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeAuditConfig(), lctx);
    const auto lhs_share = Value(
        (rank == 0 ? lhs : zero_lhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto rhs_share = Value(
        (rank == 0 ? rhs : zero_rhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto handles = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_consumer_preprocess", kPlan, kRequestId, kConsumer,
        lhs.shape(), rhs.shape(), true);

    auto* audit = ctx->getState<MJVLinearAuditState>();
    audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    const auto result = dynDispatch<Value>(
        ctx.get(), "mjv_dynamic_online", handles[0], handles[1], lhs_share,
        rhs_share, kPlan, kRequestId, kConsumer);
    const auto snapshot = audit->End();
    outputs[rank] = result.data();
    ASSERT_EQ(snapshot.high_level_calls.size(), 1);
    EXPECT_EQ(snapshot.high_level_calls[0].family,
              LinearKernelFamily::kBatchMatMulAA);
    EXPECT_EQ(snapshot.dot_ole_calls, 0);
    EXPECT_EQ(snapshot.batch_dot_ole_calls, 0);
  });

  ExpectRingEqual(ring_add(outputs[0], outputs[1]),
                  BatchRingMatMul(lhs, rhs));
}

TEST(MJVDynamicLinearTest, ActualAttentionKernelRecordsRuntimeFamily) {
  constexpr int64_t kModelEpoch = 70;
  constexpr int64_t kRequestEpoch = 190;
  const std::string kPlan(64, 'f');
  const std::string kConsumer = "attention.0.qk";
  const auto field = FieldType::FM64;
  const auto lhs = ring_rand(field, {2, 3});
  const auto rhs = ring_rand(field, {3, 2});
  const auto zero_lhs = ring_zeros(field, lhs.shape());
  const auto zero_rhs = ring_zeros(field, rhs.shape());

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeAuditConfig(), lctx);
    const auto lhs_share = Value(
        (rank == 0 ? lhs : zero_lhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto rhs_share = Value(
        (rank == 0 ? rhs : zero_rhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto handles = dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_consumer_preprocess", kPlan, kRequestEpoch,
        kConsumer, lhs.shape(), rhs.shape(), false,
        kModelEpoch, std::string("attention.family.0"),
        std::string("request.attention.0"), int64_t{1},
        static_cast<int64_t>(ObjectLifetime::kRequestImmutable),
        static_cast<int64_t>(ObjectOperand::kLhs),
        std::string("attention_matmul"));

    auto* object_state = ctx->getState<MJVObjectContextState>();
    object_state->RecordModelSetupBatch(kPlan, kModelEpoch);
    object_state->RecordRequestPreprocessBatch(kPlan, kModelEpoch,
                                               kRequestEpoch);
    object_state->BeginPlanExecution(kPlan, kModelEpoch, kRequestEpoch);
    object_state->RecordSecureInvoke();
    dynDispatch<Value>(ctx.get(), "mjv_dynamic_online", handles[0],
                       handles[1], lhs_share, rhs_share, kPlan,
                       kRequestEpoch, kConsumer);
    const auto audit = object_state->FinalizeRuntimeAudit(
        kPlan, kModelEpoch, kRequestEpoch);
    EXPECT_EQ(audit.logical_consumer_count, 1);
    EXPECT_EQ(audit.physical_family_count, 1);
    EXPECT_EQ(audit.request.required_shared_object_contexts, 1);
    EXPECT_EQ(audit.request.shared_side_logical_openings, 1);
    EXPECT_EQ(audit.request.logical_right_openings, 1);
    EXPECT_EQ(audit.request.physical_right_collectives, 1);
    EXPECT_EQ(audit.protocol_matrix_calls, 1);
    object_state->EndPlanExecution(kPlan, kModelEpoch, kRequestEpoch);
  });
}

TEST(MJVDynamicLinearTest, CorrelationsAreOwnedByStableConsumerContext) {
  const std::string plan(64, 'a');
  const DynamicConsumerKey first{plan, 7, "attention.17.dot.17"};
  const DynamicConsumerKey next_epoch{plan, 8, "attention.17.dot.17"};
  MJVDynamicLinearState state;
  state.Store(first, {});
  state.Store(next_epoch, {});

  state.BeginOnline(plan, 7);
  EXPECT_EQ(state.NextOnlineConsumer({}, {}, false), first);
  state.Consume(first);
  EXPECT_THROW(state.Consume(first), std::exception);
  EXPECT_NO_THROW(state.EndOnline(plan, 7));
  EXPECT_EQ(state.LiveCorrelationCount(), 1);
}

TEST(MJVDynamicLinearTest,
     RuntimeLoweredMatMulAVConsumesPreparedCorrelationWithoutOnlineHe) {
  constexpr int64_t kBaselineRequest = 501;
  constexpr int64_t kOursRequest = 502;
  const std::string kPlan = "indexed-manifest";
  const std::string kConsumer = "attention.0";
  const auto field = FieldType::FM64;
  const auto lhs = ring_rand(field, {1, 5});
  const auto rhs = ring_rand(field, {5, 3});
  const auto zero_lhs = ring_zeros(field, lhs.shape());
  const auto zero_rhs = ring_zeros(field, rhs.shape());
  std::vector<NdArrayRef> outputs(2);

  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeAuditConfig(), lctx);
    const auto lhs_share = Value(
        (rank == 0 ? lhs : zero_lhs).as(makeType<AShrTy>(field)), DT_I64);
    const auto rhs_private = Value(
        (rank == 0 ? rhs : zero_rhs).as(makeType<Priv2kTy>(field, 0)),
        DT_I64);
    dynDispatch<std::vector<Value>>(
        ctx.get(), "mjv_consumer_preprocess", kPlan, kOursRequest, kConsumer,
        lhs.shape(), rhs.shape(), false);

    auto* audit = ctx->getState<MJVLinearAuditState>();
    auto* scope = ctx->getState<MJVLinearScopeState>();
    scope->Begin(kBaselineRequest, kPlan,
                 LinearBackend::kBumbleBee);
    audit->Begin(LinearPhase::kOnline, LinearBackend::kBumbleBee);
    dynDispatch<Value>(ctx.get(), "mmul_av", lhs_share, rhs_private);
    const auto baseline = audit->End();
    scope->Finish(kBaselineRequest, baseline);

    scope->Begin(kOursRequest, kPlan, LinearBackend::kOurs);
    ctx->getState<MJVDynamicLinearState>()->BeginOnline(kPlan, kOursRequest);
    audit->Begin(LinearPhase::kOnline, LinearBackend::kOurs);
    const auto result =
        dynDispatch<Value>(ctx.get(), "mmul_av", lhs_share, rhs_private);
    ctx->getState<MJVDynamicLinearState>()->EndOnline(kPlan, kOursRequest);
    const auto replacement = audit->End();
    scope->Finish(kOursRequest, replacement);

    outputs[rank] = result.data();
    ASSERT_EQ(replacement.high_level_calls.size(), 1);
    EXPECT_EQ(replacement.high_level_calls[0].family,
              LinearKernelFamily::kMatMulAV);
    EXPECT_EQ(replacement.dot_ole_calls, 0);
    EXPECT_EQ(replacement.batch_dot_ole_calls, 0);
  });

  ExpectRingEqual(ring_add(outputs[0], outputs[1]), ring_mmul(lhs, rhs));
}

}  
}
