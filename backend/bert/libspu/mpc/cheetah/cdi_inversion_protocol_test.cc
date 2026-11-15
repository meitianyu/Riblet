// Copyright 2026

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/cdi_inversion.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
namespace {

constexpr int64_t kScale = int64_t{1} << 18;

RuntimeConfig MakeConfig() {
  RuntimeConfig config;
  config.set_protocol(ProtocolKind::CHEETAH);
  config.set_field(FieldType::FM64);
  config.set_fxp_fraction_bits(18);
  config.set_fxp_div_goldschmidt_iters(3);
  config.mutable_cheetah_2pc_config()->set_ot_kind(
      CheetahOtKind::YACL_Softspoken);
  return config;
}

int64_t Signed(uint64_t value) { return static_cast<int64_t>(value); }

int64_t RoundShift(__int128 value, int bits) {
  const auto shifted = value >> bits;
  const auto round = (value >> (bits - 1)) & 1;
  return static_cast<int64_t>(shifted + round);
}

int64_t MulF18(int64_t left, int64_t right) {
  return RoundShift(static_cast<__int128>(left) * right, 18);
}

int64_t RecipReference(int64_t operand, int64_t a, int64_t b, int k) {
  auto y = MulF18(a, operand) + b;
  for (int i = 0; i < k; ++i) {
    y = MulF18(y, 2 * kScale - MulF18(operand, y));
  }
  return y;
}

int64_t RsqrtReference(int64_t operand, int64_t a, int64_t b, int k) {
  auto y = MulF18(a, operand) + b;
  for (int i = 0; i < k; ++i) {
    const auto yy = MulF18(y, y);
    const auto xyy = MulF18(operand, yy);
    y = RoundShift(MulF18(y, 3 * kScale - xyy), 1);
  }
  return y;
}

bool ReciprocalMeetsP12(int64_t operand, int64_t y) {
  constexpr __int128 m = 1 << 12;
  constexpr __int128 scale = kScale;
  const __int128 residual =
      scale * scale - static_cast<__int128>(operand) * y;
  const __int128 magnitude = residual < 0 ? -residual : residual;
  return magnitude * m <= scale * scale;
}

bool RsqrtMeetsP12(int64_t operand, int64_t y) {
  constexpr __int128 m = 1 << 12;
  constexpr __int128 scale = kScale;
  const __int128 scaled_square =
      m * m * static_cast<__int128>(operand) * y * y;
  const __int128 scale_cube = scale * scale * scale;
  return (m - 1) * (m - 1) * scale_cube <= scaled_square &&
         scaled_square <= (m + 1) * (m + 1) * scale_cube;
}

Value SecretInput(const NdArrayRef& clear, FieldType field, size_t rank,
                  DataType dtype) {
  auto zeros = ring_zeros(field, clear.shape());
  return Value((rank == 0 ? clear : zeros).as(makeType<AShrTy>(field)), dtype);
}

Value PublicInput(const NdArrayRef& clear, FieldType field, DataType dtype) {
  return Value(clear.as(makeType<Pub2kTy>(field)), dtype);
}

std::array<NdArrayRef, 2> RunInversion(const std::string& op,
                                       const NdArrayRef& operands,
                                       const NdArrayRef& coefficients,
                                       const NdArrayRef& routes,
                                       const NdArrayRef& public_ks) {
  std::array<NdArrayRef, 2> shares;
  utils::simulate(2, [&](const std::shared_ptr<yacl::link::Context>& lctx) {
    const auto rank = lctx->Rank();
    auto ctx = makeCheetahProtocol(MakeConfig(), lctx);
    EXPECT_TRUE(ctx->hasKernel("cdi_invert"));
    auto outputs = dynDispatch<std::vector<Value>>(
        ctx.get(), "cdi_invert", SecretInput(operands, FieldType::FM64, rank,
                                              DT_F32),
        SecretInput(coefficients, FieldType::FM64, rank, DT_F32),
        PublicInput(routes, FieldType::FM64, DT_I64),
        PublicInput(public_ks, FieldType::FM64, DT_I64), op);
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs[0].shape(), operands.shape());
    EXPECT_EQ(outputs[0].dtype(), DT_F32);
    shares[rank] = outputs[0].data();
  });
  return shares;
}

TEST(CdiInversionProtocolTest, ReciprocalUsesExactRealYGraphAndSpuFallback) {
  const auto field = FieldType::FM64;
  auto operands = ring_zeros(field, {3});
  auto coefficients = ring_zeros(field, {3, 2});
  auto routes = ring_zeros(field, {3});
  auto public_ks = ring_zeros(field, {3});
  auto x = NdArrayView<uint64_t>(operands);
  auto ab = NdArrayView<uint64_t>(coefficients);
  auto z = NdArrayView<uint64_t>(routes);
  auto k = NdArrayView<uint64_t>(public_ks);
  x[0] = 2 * kScale;
  x[1] = 3 * kScale;
  x[2] = kScale;
  ab[0] = static_cast<uint64_t>(-32768);
  ab[1] = 196608;
  ab[2] = static_cast<uint64_t>(-16384);
  ab[3] = 131072;
  z[2] = 1;
  k[0] = 2;
  k[1] = 3;

  const auto shares = RunInversion("reciprocal", operands, coefficients,
                                   routes, public_ks);
  const auto result = ring_add(shares[0], shares[1]);
  const auto view = NdArrayView<const uint64_t>(result);
  EXPECT_EQ(Signed(view[0]), RecipReference(Signed(x[0]), Signed(ab[0]),
                                            Signed(ab[1]), k[0]));
  EXPECT_EQ(Signed(view[1]), RecipReference(Signed(x[1]), Signed(ab[2]),
                                            Signed(ab[3]), k[1]));
  EXPECT_GT(Signed(view[2]), 0);
  EXPECT_TRUE(ReciprocalMeetsP12(Signed(x[2]), Signed(view[2])));
}

TEST(CdiInversionProtocolTest, RsqrtUsesExactFrozenOperationOrder) {
  const auto field = FieldType::FM64;
  auto operands = ring_zeros(field, {2});
  auto coefficients = ring_zeros(field, {2, 2});
  auto routes = ring_zeros(field, {2});
  auto public_ks = ring_zeros(field, {2});
  auto x = NdArrayView<uint64_t>(operands);
  auto ab = NdArrayView<uint64_t>(coefficients);
  auto z = NdArrayView<uint64_t>(routes);
  auto k = NdArrayView<uint64_t>(public_ks);
  x[0] = 4 * kScale;
  x[1] = kScale;
  ab[0] = static_cast<uint64_t>(-8192);
  ab[1] = 163840;
  z[1] = 1;
  k[0] = 2;

  const auto shares =
      RunInversion("rsqrt", operands, coefficients, routes, public_ks);
  const auto result = ring_add(shares[0], shares[1]);
  const auto view = NdArrayView<const uint64_t>(result);
  EXPECT_EQ(Signed(view[0]), RsqrtReference(Signed(x[0]), Signed(ab[0]),
                                            Signed(ab[1]), k[0]));
  EXPECT_GT(Signed(view[1]), 0);
  EXPECT_TRUE(RsqrtMeetsP12(Signed(x[1]), Signed(view[1])));
}

}  
}
