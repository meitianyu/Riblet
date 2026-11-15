#include <array>
#include "gtest/gtest.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"
namespace spu::mpc::cheetah {
TEST(VitRingStock, BothRingsSurviveAlternation) {
 std::array<std::array<NdArrayRef,3>,2> large,small,again;
 utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
  RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
  cfg.mutable_cheetah_2pc_config()->set_ot_kind(YACL_Ferret);
  auto ctx=makeCheetahProtocol(cfg,link);auto* state=ctx->getState<CheetahMulState>();
  state->BenchPrepare(FM64,16);state->BenchPrepare(FM32,16);
  large[link->Rank()]=state->TakeCachedBeaver(FM64,4);
  small[link->Rank()]=state->TakeCachedBeaver(FM32,4);
  again[link->Rank()]=state->TakeCachedBeaver(FM64,4);
  EXPECT_EQ(state->BenchUsed(),12);
  EXPECT_ANY_THROW(state->TakeCachedBeaver(FM32,17));
 });
 for(auto* pair:{&large,&small,&again}){
  auto a=ring_add((*pair)[0][0],(*pair)[1][0]);auto b=ring_add((*pair)[0][1],(*pair)[1][1]);
  EXPECT_TRUE(ring_all_equal(ring_add((*pair)[0][2],(*pair)[1][2]),ring_mul(a,b)));
 }
}
}
