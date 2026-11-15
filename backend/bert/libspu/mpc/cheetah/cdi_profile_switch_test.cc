#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/ostream_sink.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/cdi_inversion.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {
TEST(CdiProfileSwitch, NoDiagnosticLogsWhenDisabledStillComputesBothInverses) {
  constexpr uint64_t S=uint64_t{1}<<18;
  for (const bool profile : {false,true}) {
    std::ostringstream logs;
    auto old_logger=spdlog::default_logger();
    auto sink=std::make_shared<spdlog::sinks::ostream_sink_mt>(logs);
    spdlog::set_default_logger(std::make_shared<spdlog::logger>("capture",sink));
    std::array<std::array<NdArrayRef,2>,2> results;
    utils::simulate(2,[&](const std::shared_ptr<yacl::link::Context>& link){
      RuntimeConfig cfg;cfg.set_protocol(CHEETAH);cfg.set_field(FM64);cfg.set_fxp_fraction_bits(18);
      cfg.set_enable_hal_profile(profile);
      cfg.mutable_cheetah_2pc_config()->set_ot_kind(YACL_Ferret);
      cfg.mutable_cheetah_2pc_config()->set_enable_mul_lsb_error(true);
      auto ctx=makeCheetahProtocol(cfg,link);
      for(int op=0;op<2;++op){
        auto x=ring_zeros(FM64,{1}),ab=ring_zeros(FM64,{1,2}),r=ring_zeros(FM64,{1}),k=ring_zeros(FM64,{1});
        auto xv=NdArrayView<uint64_t>(x);auto av=NdArrayView<uint64_t>(ab);auto kv=NdArrayView<uint64_t>(k);
        if(link->Rank()==0){
          xv[0]=(op?4:2)*S;
          av[0]=uint64_t(-int64_t(op?S/32:S/8));
          av[1]=op?5*S/8:3*S/4;
        }
        kv[0]=1;
        auto out=dynDispatch<std::vector<Value>>(ctx.get(),"cdi_invert",
          Value(x.as(makeType<AShrTy>(FM64)),DT_F32),Value(ab.as(makeType<AShrTy>(FM64)),DT_F32),
          Value(r.as(makeType<Pub2kTy>(FM64)),DT_I64),Value(k.as(makeType<Pub2kTy>(FM64)),DT_I64),
          std::string(op?"rsqrt":"reciprocal"));
        results[op][link->Rank()]=out[0].data();
      }
    });
    spdlog::set_default_logger(old_logger);
    for(int op=0;op<2;++op){
      auto result=ring_add(results[op][0],results[op][1]);
      EXPECT_LE(std::abs(static_cast<int64_t>(NdArrayView<const uint64_t>(result)[0])-int64_t(S/2)),4);
    }
    EXPECT_EQ(logs.str().find("CDI_PARTS")!=std::string::npos,profile);
    EXPECT_EQ(logs.str().find("CDI_COST")!=std::string::npos,profile);
  }
}
}
