#include <array>
#include <filesystem>
#include "gtest/gtest.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/mjv_linear.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
template<class T,class=void>struct HasValidatedPlan:std::false_type{};
template<class T>struct HasValidatedPlan<T,std::void_t<decltype(std::declval<T>().RegisterValidatedPlan(
    std::declval<std::string>(),std::declval<std::vector<LinearCallFingerprint>>()))>>:std::true_type{};
template<class T>void CheckValidatedPlan(){
  EXPECT_TRUE(HasValidatedPlan<T>::value);
  if constexpr(HasValidatedPlan<T>::value){
    T state;std::string hash(64,'d');
    std::vector<LinearCallFingerprint> calls{{0,LinearKernelFamily::kMatMulAA,{2,3},{3,4},{2,4}}};
    state.RegisterValidatedPlan(hash,calls);
    state.Begin(1,hash,LinearBackend::kOurs);
    EXPECT_EQ(state.ConsumeExpectedFamily({2,3},{3,4},{2,4}),LinearKernelFamily::kMatMulAA);
    LinearAuditSnapshot snapshot;snapshot.high_level_calls=calls;state.Finish(1,snapshot);
    auto bad=calls;bad[0].output_shape={2,5};
    EXPECT_ANY_THROW(state.RegisterValidatedPlan(hash,bad));
    state.Begin(2,hash,LinearBackend::kOurs);
    EXPECT_ANY_THROW(state.ConsumeExpectedFamily({2,3},{3,5},{2,5}));state.Abort();
  }
}
TEST(VitNative, ExplicitPlanChecksActualShapesWithoutFairForward){CheckValidatedPlan<MJVLinearScopeState>();}

}
