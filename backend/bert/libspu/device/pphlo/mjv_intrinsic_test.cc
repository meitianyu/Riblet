// Copyright 2026

#include <array>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

#include "libspu/device/utils/pphlo_executor_test_runner.h"

namespace spu::device::pphlo::test {
namespace {

TEST(MJVIntrinsicTest, RunsThreeStagesWithMultipleConsumerOutputs) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(xt::xarray<float>({{1.0F, 2.0F, 3.0F},
                                     {4.0F, 5.0F, 6.0F},
                                     {7.0F, 8.0F, 9.0F}}),
                  VIS_SECRET, 0);
  runner.addInput(
      xt::xarray<float>({{1.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 1.0F}}),
      VIS_SECRET, 1);

  runner.run(R"(
func.func @main(%arg0: tensor<3x3x!pphlo.secret<f32>>, %arg1: tensor<2x3x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>, tensor<2x1x!pphlo.secret<f32>>) {
  %0 = pphlo.custom_call @"mjv_model_setup:7:6:2,1"(%arg0) : (tensor<3x3x!pphlo.secret<f32>>) -> tensor<3x3x!pphlo.secret<f32>>
  %1 = pphlo.custom_call @"mjv_request_preprocess:7:101:2"(%0) : (tensor<3x3x!pphlo.secret<f32>>) -> tensor<2x3x!pphlo.secret<f32>>
  %2:2 = pphlo.custom_call @"mjv_group_online:7:101:2,1"(%1, %arg1) : (tensor<2x3x!pphlo.secret<f32>>, tensor<2x3x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>, tensor<2x1x!pphlo.secret<f32>>)
  return %2#0, %2#1 : tensor<2x2x!pphlo.secret<f32>>, tensor<2x1x!pphlo.secret<f32>>
})",
             2);

  const std::array<float, 4> expected_first = {8.0F, 10.0F, 11.0F, 13.0F};
  const std::array<float, 2> expected_second = {12.0F, 15.0F};
  runner.verifyOutput(expected_first.data(), 0);
  runner.verifyOutput(expected_second.data(), 1);
}

TEST(MJVIntrinsicTest, ConvertsBooleanBackedFxpActivationToArithmeticShare) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(xt::xarray<float>({{1.0F, 2.0F, 3.0F},
                                     {4.0F, 5.0F, 6.0F},
                                     {7.0F, 8.0F, 9.0F}}),
                  VIS_SECRET, 0);
  runner.addInput(
      xt::xarray<float>({{2.0F, 0.0F, 3.0F}, {0.0F, 4.0F, 5.0F}}),
      VIS_SECRET, 1);
  runner.addInput(
      xt::xarray<float>({{2.0F, 1.0F, 3.0F}, {1.0F, 4.0F, 5.0F}}),
      VIS_SECRET, 0);

  runner.run(R"(
func.func @main(%arg0: tensor<3x3x!pphlo.secret<f32>>, %arg1: tensor<2x3x!pphlo.secret<f32>>, %arg2: tensor<2x3x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>, tensor<2x1x!pphlo.secret<f32>>) {
  %0 = pphlo.custom_call @"mjv_model_setup:8:6:2,1"(%arg0) : (tensor<3x3x!pphlo.secret<f32>>) -> tensor<3x3x!pphlo.secret<f32>>
  %1 = pphlo.custom_call @"mjv_request_preprocess:8:102:2"(%0) : (tensor<3x3x!pphlo.secret<f32>>) -> tensor<2x3x!pphlo.secret<f32>>
  %cmp = pphlo.equal %arg1, %arg2 : (tensor<2x3x!pphlo.secret<f32>>, tensor<2x3x!pphlo.secret<f32>>) -> tensor<2x3x!pphlo.secret<i1>>
  %activation = pphlo.convert %cmp : (tensor<2x3x!pphlo.secret<i1>>) -> tensor<2x3x!pphlo.secret<f32>>
  %2:2 = pphlo.custom_call @"mjv_group_online:8:102:2,1"(%1, %activation) : (tensor<2x3x!pphlo.secret<f32>>, tensor<2x3x!pphlo.secret<f32>>) -> (tensor<2x2x!pphlo.secret<f32>>, tensor<2x1x!pphlo.secret<f32>>)
  return %2#0, %2#1 : tensor<2x2x!pphlo.secret<f32>>, tensor<2x1x!pphlo.secret<f32>>
})",
             2);

  const std::array<float, 4> expected_first = {8.0F, 10.0F, 11.0F, 13.0F};
  const std::array<float, 2> expected_second = {12.0F, 15.0F};
  runner.verifyOutput(expected_first.data(), 0);
  runner.verifyOutput(expected_second.data(), 1);
}

TEST(MJVIntrinsicTest, RoutesOnlyDotsInsideExplicitOursScope) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(
      xt::xarray<float>({{{1.0F, 2.0F, 3.0F},
                           {4.0F, 5.0F, 6.0F}}}),
      VIS_SECRET, 0);
  runner.addInput(
      xt::xarray<float>({{{1.0F, 0.0F},
                           {0.0F, 1.0F},
                           {1.0F, 1.0F}}}),
      VIS_SECRET, 1);

  runner.run(R"(
func.func @main(%arg0: tensor<1x2x3x!pphlo.secret<f32>>, %arg1: tensor<1x3x2x!pphlo.secret<f32>>) -> (tensor<1x2x2x!pphlo.secret<f32>>, tensor<1x2x2x!pphlo.secret<f32>>, tensor<1x2x2x!pphlo.secret<f32>>, tensor<1x2x2x!pphlo.secret<f32>>) {
  %masks:2 = pphlo.custom_call @"mjv_consumer_preprocess:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:attention.0:b:1x2x3:1x3x2"() : () -> (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>)
  %0 = pphlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>) -> tensor<1x2x2x!pphlo.secret<f32>>
  pphlo.custom_call @"mjv_linear_begin:bumblebee:7:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"() : () -> ()
  %1 = pphlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>) -> tensor<1x2x2x!pphlo.secret<f32>>
  pphlo.custom_call @"mjv_linear_end:7"() : () -> ()
  pphlo.custom_call @"mjv_linear_begin:ours:7:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"() : () -> ()
  %2 = pphlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>) -> tensor<1x2x2x!pphlo.secret<f32>>
  pphlo.custom_call @"mjv_linear_end:7"() : () -> ()
  %3 = pphlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>) -> tensor<1x2x2x!pphlo.secret<f32>>
  return %0, %1, %2, %3 : tensor<1x2x2x!pphlo.secret<f32>>, tensor<1x2x2x!pphlo.secret<f32>>, tensor<1x2x2x!pphlo.secret<f32>>, tensor<1x2x2x!pphlo.secret<f32>>
})",
             4);

  const std::array<float, 4> expected = {4.0F, 5.0F, 10.0F, 11.0F};
  for (size_t output = 0; output < 4; ++output) {
    runner.verifyOutput(expected.data(), output);
  }
}

TEST(MJVIntrinsicTest, RoutesContextEpochByPlanHashWithoutPerOpIdentity) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(xt::xarray<float>({{1.0F, 2.0F}, {3.0F, 4.0F}}),
                  VIS_SECRET, 0);

  runner.run(R"(
func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>> {
  %0 = pphlo.custom_call @"mjv_epoch_begin:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%arg0) : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
  %audit = pphlo.custom_call @"mjv_audit_snapshot:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%0) : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<33xi64>
  %1 = pphlo.custom_call @"mjv_epoch_end:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%0, %audit) : (tensor<2x2x!pphlo.secret<f32>>, tensor<33xi64>) -> tensor<2x2x!pphlo.secret<f32>>
  return %1 : tensor<2x2x!pphlo.secret<f32>>
})",
             1);

  const std::array<float, 4> expected = {1.0F, 2.0F, 3.0F, 4.0F};
  runner.verifyOutput(expected.data(), 0);
}

TEST(MJVIntrinsicTest, RoutesExplicitFinalRequestModelRelease) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(xt::xarray<float>({{1.0F, 2.0F}, {3.0F, 4.0F}}),
                  VIS_SECRET, 0);

  runner.run(R"(
func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>> {
  %0 = pphlo.custom_call @"mjv_epoch_begin:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%arg0) : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
  %audit = pphlo.custom_call @"mjv_audit_snapshot:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%0) : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<33xi64>
  %1 = pphlo.custom_call @"mjv_epoch_end:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19:release_model"(%0, %audit) : (tensor<2x2x!pphlo.secret<f32>>, tensor<33xi64>) -> tensor<2x2x!pphlo.secret<f32>>
  return %1 : tensor<2x2x!pphlo.secret<f32>>
})",
             1);

  const std::array<float, 4> expected = {1.0F, 2.0F, 3.0F, 4.0F};
  runner.verifyOutput(expected.data(), 0);
}

TEST(MJVIntrinsicTest, RoutesOneModelAndRequestBatchMarkerPerPhase) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(xt::xarray<float>({{1.0F, 2.0F}, {3.0F, 4.0F}}),
                  VIS_SECRET, 0);

  runner.run(R"(
func.func @main(%arg0: tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>> {
  %0 = pphlo.custom_call @"mjv_model_setup_batch:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7"(%arg0) : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
  %1 = pphlo.custom_call @"mjv_request_preprocess_batch:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%0) : (tensor<2x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
  return %1 : tensor<2x2x!pphlo.secret<f32>>
})",
             1);

  const std::array<float, 4> expected = {1.0F, 2.0F, 3.0F, 4.0F};
  runner.verifyOutput(expected.data(), 0);
}

TEST(MJVIntrinsicTest, ReturnsActualPhaseBatchAuditFromProtocolState) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);

  runner.run(R"(
func.func @main() -> tensor<33xi64> {
  pphlo.custom_call @"mjv_model_setup_batch:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7"() : () -> ()
  pphlo.custom_call @"mjv_request_preprocess_batch:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"() : () -> ()
  pphlo.custom_call @"mjv_epoch_begin:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"() : () -> ()
  %audit = pphlo.custom_call @"mjv_audit_snapshot:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"() : () -> tensor<33xi64>
  %ended = pphlo.custom_call @"mjv_epoch_end:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%audit, %audit) : (tensor<33xi64>, tensor<33xi64>) -> tensor<33xi64>
  return %ended : tensor<33xi64>
})",
             1);

  const std::array<int64_t, 33> expected = {
      1, 1, 7, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0};
  runner.verifyOutput(expected.data(), 0);
}

TEST(MJVIntrinsicTest, ReconcilesLinearScopeIntoRuntimeAudit) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(
      xt::xarray<float>({{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}}),
      VIS_SECRET, 0);
  runner.addInput(
      xt::xarray<float>({{1.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 1.0F}}),
      VIS_SECRET, 1);

  runner.run(R"(
func.func @main(%arg0: tensor<2x3x!pphlo.secret<f32>>, %arg1: tensor<3x2x!pphlo.secret<f32>>) -> tensor<33xi64> {
  %anchor = pphlo.custom_call @"mjv_epoch_begin:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%arg0) : (tensor<2x3x!pphlo.secret<f32>>) -> tensor<2x3x!pphlo.secret<f32>>
  pphlo.custom_call @"mjv_linear_begin:bumblebee:19:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"() : () -> ()
  %product = pphlo.dot %anchor, %arg1 : (tensor<2x3x!pphlo.secret<f32>>, tensor<3x2x!pphlo.secret<f32>>) -> tensor<2x2x!pphlo.secret<f32>>
  pphlo.custom_call @"mjv_linear_end:19"() : () -> ()
  %audit = pphlo.custom_call @"mjv_audit_snapshot:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"() : () -> tensor<33xi64>
  %ended = pphlo.custom_call @"mjv_epoch_end:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:19"(%product, %audit) : (tensor<2x2x!pphlo.secret<f32>>, tensor<33xi64>) -> tensor<2x2x!pphlo.secret<f32>>
  return %audit : tensor<33xi64>
})",
             1);

  const std::array<int64_t, 33> expected = {
      1, 1, 7, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0};
  runner.verifyOutput(expected.data(), 0);
}

TEST(MJVIntrinsicTest, RoutesConsumerPreprocessByStablePlanEpochIdentity) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);

  runner.run(R"(
func.func @main() -> (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>) {
  %masks:2 = pphlo.custom_call @"mjv_consumer_preprocess:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:7:attention.17.dot.17:b:1x2x3:1x3x2"() : () -> (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>)
  return %masks#0, %masks#1 : tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>
})",
             2);
}

TEST(MJVIntrinsicTest, RoutesCompleteRuntimeFamilyMetadataToProtocolKernels) {
  Runner runner(2, FieldType::FM64, ProtocolKind::CHEETAH);
  runner.addInput(xt::xarray<float>({{1.0F, 2.0F, 3.0F},
                                     {4.0F, 5.0F, 6.0F},
                                     {7.0F, 8.0F, 9.0F}}),
                  VIS_SECRET, 0);

  runner.run(R"(
func.func @main(%arg0: tensor<3x3x!pphlo.secret<f32>>) -> (tensor<2x3x!pphlo.secret<f32>>, tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>) {
  %model = pphlo.custom_call @"mjv_model_setup:7:3:2,1:matmul_aa:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:70:dense.family.0:request.hidden.0:1:r:l:other_dense:2x3:3x3"(%arg0) : (tensor<3x3x!pphlo.secret<f32>>) -> tensor<3x3x!pphlo.secret<f32>>
  %request = pphlo.custom_call @"mjv_request_preprocess:7:19:2:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:70:190:dense.family.0"(%model) : (tensor<3x3x!pphlo.secret<f32>>) -> tensor<2x3x!pphlo.secret<f32>>
  %dynamic:2 = pphlo.custom_call @"mjv_consumer_preprocess:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:190:attention.17.dot.17:b:1x2x3:1x3x2:70:attention.family.17:request.attention.17:1:r:l:attention_matmul"() : () -> (tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>)
  return %request, %dynamic#0, %dynamic#1 : tensor<2x3x!pphlo.secret<f32>>, tensor<1x2x3x!pphlo.secret<f32>>, tensor<1x3x2x!pphlo.secret<f32>>
})",
             3);
}

}  
}
