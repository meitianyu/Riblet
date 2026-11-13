#include <type_traits>
#include <utility>
#include "gtest/gtest.h"
#include "libspu/mpc/cheetah/mjv_linear.h"

namespace spu::mpc::cheetah {
template<class T,class=void>struct HasPublicRegistration:std::false_type{};
template<class T>struct HasPublicRegistration<T,std::void_t<decltype(std::declval<T>().RegisterPublicPlan(
    std::declval<std::string>(),std::declval<std::vector<LinearCallFingerprint>>()))>>:std::true_type{};
template<class T>void Check(){
  ASSERT_TRUE(HasPublicRegistration<T>::value);
  if constexpr(HasPublicRegistration<T>::value){
    T state;const std::string hash(64,'a');
    std::vector<LinearCallFingerprint> calls{{0,LinearKernelFamily::kMatMulAA,{2,3},{3,4},{2,4}}};
    state.RegisterPublicPlan(hash,calls);
    for(int epoch=1;epoch<=2;++epoch){
      state.Begin(epoch,hash,LinearBackend::kOurs);
      EXPECT_EQ(state.ConsumeExpectedFamily({2,3},{3,4},{2,4}),LinearKernelFamily::kMatMulAA);
      LinearAuditSnapshot audit;audit.high_level_calls=calls;state.Finish(epoch,audit);
    }
    auto bad=calls;bad[0].output_shape={2,9};EXPECT_ANY_THROW(state.RegisterPublicPlan(hash,bad));
    state.Begin(3,hash,LinearBackend::kOurs);
    EXPECT_ANY_THROW(state.ConsumeExpectedFamily({2,3},{3,9},{2,9}));state.Abort();
  }
}
TEST(CdiPublicLinearPlan, InitializeWithoutExtraForwardAndPreserveChecks){Check<MJVLinearScopeState>();}
}
