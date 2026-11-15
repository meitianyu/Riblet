#include "libspu/core/cpp_prepare_observer.h"

// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Modified for Riblet; see NOTICE. */

#pragma once

#include "yacl/base/dynamic_bitset.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/kernel/algorithms/base_ot.h"
#include "yacl/kernel/algorithms/ferret_ote.h"
#include "yacl/kernel/algorithms/iknp_ote.h"
#include "yacl/kernel/algorithms/ot_store.h"
#include "yacl/kernel/algorithms/softspoken_ote.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/ot/yacl/bench_budget.h"
#include <array>
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/ot/yacl/yacl_util.h"

namespace spu::mpc::cheetah {

namespace yc = yacl::crypto;
namespace yl = yacl::link;

class YaclOTeAdapter {
 public:
  YaclOTeAdapter() = default;
  virtual ~YaclOTeAdapter() = default;
  virtual void send_rcot(absl::Span<uint128_t> data) = 0;
  virtual void recv_rcot(absl::Span<uint128_t> data,
                         absl::Span<uint8_t> choices) = 0;
  virtual void send_cot(absl::Span<uint128_t> data) = 0;
  virtual void recv_cot(absl::Span<uint128_t> data,
                        absl::Span<const uint8_t> choices) = 0;
  virtual void OneTimeSetup() = 0;

  virtual void BenchReserve(uint64_t) { throw std::runtime_error("unsupported OT stock backend"); }
  virtual void BenchOff() { throw std::runtime_error("unsupported OT stock backend"); }
  virtual std::array<uint64_t,2> BenchRawStats() const { throw std::runtime_error("unsupported OT stock backend"); }
  uint128_t Delta{0};
  virtual uint128_t GetDelta() const { return Delta; }
};

class YaclFerretOTeAdapter : public YaclOTeAdapter {
 public:
  YaclFerretOTeAdapter(const std::shared_ptr<yl::Context>& ctx,
                       bool is_sender) {
    ctx_ = ctx;
    is_sender_ = is_sender;
    reserve_num_ = yc::FerretCotHelper(lpn_param_, 0);

spu::lifecycle::Scope scope_ferret_fixed_lpn_buffer_allocation("Ferret.fixed-LPN-buffer-allocation", "R-S");

    ot_buff_ = yacl::Buffer(lpn_param_.n * sizeof(uint128_t));

scope_ferret_fixed_lpn_buffer_allocation.Stop();

    id_ = yacl_id_;
    ++yacl_id_;
  }

  ~YaclFerretOTeAdapter() {
    SPDLOG_DEBUG(
        "[FerretAdapter {}]({}), comsume OT {}, total time {:.3e} ms, "
        "invoke bootstrap {} ( {:.2e} ms per bootstrap, {:.2e} ms per ot )",
        id_, (is_sender_ ? fmt::format("Sender") : fmt::format("Receiver")),
        consumed_ot_num_, bootstrap_time_, bootstrap_num_,
        bootstrap_time_ / bootstrap_num_, bootstrap_time_ / consumed_ot_num_);
  }

  void OneTimeSetup() override;

  void recv_cot(absl::Span<uint128_t> data,
                absl::Span<const uint8_t> choices) override {
    recv_cot(data, VecU8toBitset(choices));
  }

  void send_rcot(absl::Span<uint128_t> data) override { rcot(data); }

  void recv_rcot(absl::Span<uint128_t> data,
                 absl::Span<uint8_t> choices) override {
    rcot(data);
    std::transform(data.data(), data.data() + data.size(), choices.data(),
                   [](uint128_t v) { return (uint8_t)v & 0x1; });
  }

  // Correlated Cot with Chosen Choices
  void send_cot(absl::Span<uint128_t> data) override;

  void recv_cot(absl::Span<uint128_t> data,
                const yacl::dynamic_bitset<uint128_t>& choices);

  void BenchOff() override { bench_budget_.disarm(); }
  void BenchReserve(uint64_t n) override {

spu::lifecycle::Scope scope_cot_raw_reserve("COT.raw-reserve", "unclassified", [&] { return static_cast<uint64_t>((ctx_)->GetStats()->sent_bytes); }, "yacl-shared-inclusive");
scope_cot_raw_reserve.Outcome("prepared");

    bench_budget_.disarm();OneTimeSetup();
    bench_large_=false;bench_raw_position_=0;
    std::vector<uint128_t>().swap(bench_raw_);
    if(n>lpn_param_.n-reserve_num_) {
      SPU_ENFORCE(n<=uint64_t{1}<<32,"raw stock reservation exceeds public budget ABI");
      bench_raw_.resize(n);

      rcot(absl::MakeSpan(bench_raw_));bench_large_=true;
    } else if(buff_upper_bound_-buff_used_num_<n) { Bootstrap(); }
    bench_budget_.arm(n);
  }
  std::array<uint64_t,2> BenchRawStats() const override {
    SPU_ENFORCE(bench_budget_.active(),"raw OT stock unprepared");
    return {bench_budget_.used(),bench_large_?bench_raw_.size()-bench_raw_position_:buff_upper_bound_-buff_used_num_};
  }
  // YACL FERRET OTE ENTRY
  void rcot(absl::Span<uint128_t> data);

  uint128_t GetDelta() const override { return Delta; }

  uint128_t GetConsumed() const { return consumed_ot_num_; }

  double GetTime() const { return bootstrap_time_; }

 private:
  std::shared_ptr<yl::Context> ctx_{nullptr};

  bool is_sender_{false};

  bool is_setup_{false};

  yc::LpnParam pre_lpn_param_{470016, 32768, 918,
                              yc::LpnNoiseAsm::RegularNoise};

  yc::LpnParam lpn_param_{10485760, 452000, 1280,
                          yc::LpnNoiseAsm::RegularNoise};

  static constexpr uint128_t one =
      yacl::MakeUint128(0xffffffffffffffff, 0xfffffffffffffffe);

  BenchBudget bench_budget_;
  std::vector<uint128_t> bench_raw_;
  uint64_t bench_raw_position_=0;
  bool bench_large_=false;
  uint64_t reserve_num_{0};

  uint64_t buff_used_num_{0};

  uint64_t buff_upper_bound_{0};

  // We choose `yacl::Buffer` instead of `yacl::UninitAlignedVector`. Because
  // `yacl::UninitAlignedVector` or `std::vector` would fill the
  // vector with initializing data. When `size` is a big number, it would
  // take lots of time to set the memory (ten millison for thiry milliseconds).
  // Thus, we use `yacl::Buffer` to avoid meaningless initialization.
  yacl::Buffer ot_buff_;  // ot buffer

  // Yacl Ferret OTe
  void Bootstrap();
  // Yacl Ferret OTe
  void BootstrapInplace(absl::Span<uint128_t> ot, absl::Span<uint128_t> data);

  // runtime record
  uint128_t consumed_ot_num_{0};
  uint128_t bootstrap_num_{0};  // number of invoke bootstrap
  double bootstrap_time_{0.0};  // ms
  uint128_t id_{0};
  static uint128_t yacl_id_;
};

class YaclIknpOTeAdapter : public YaclOTeAdapter {
 public:
  YaclIknpOTeAdapter(const std::shared_ptr<yl::Context>& ctx, bool is_sender) {
    ctx_ = ctx;
    is_sender_ = is_sender;
    id_ = yacl_id_;
    ++yacl_id_;
  }

  ~YaclIknpOTeAdapter() {
    SPDLOG_DEBUG(
        "[IknpAdapter {}]({}), comsume OT {}, total time {:.3e} ms,"
        "invoke IKNP-OTe {} ( {:.2e} ms per iknp , {:.2e} ms per ot )",
        id_, (is_sender_ ? fmt::format("Sender") : fmt::format("Receiver")),
        consumed_ot_num_, ote_time_, ote_num_, ote_time_ / ote_num_,
        ote_time_ / consumed_ot_num_);
  }

  void OneTimeSetup() override;

  void recv_cot(absl::Span<uint128_t> data,
                absl::Span<const uint8_t> choices) override {
    recv_cot(data, VecU8toBitset(choices));
  }

  inline void send_rcot(absl::Span<uint128_t> data) override { send_cot(data); }

  inline void recv_rcot(absl::Span<uint128_t> data,
                        absl::Span<uint8_t> choices) override {
    auto _choices =
        yc::RandBits<yacl::dynamic_bitset<uint128_t>>(data.size(), true);
    BitsettoVecU8(_choices, choices);

    recv_cot(data, _choices);
  }

  // IKNP ENTRY
  // Correlated Cot with Chosen Choices
  void send_cot(absl::Span<uint128_t> data) override;

  void recv_cot(absl::Span<uint128_t> data,
                const yacl::dynamic_bitset<uint128_t>& choices);

  uint128_t GetDelta() const override { return Delta; }

  uint128_t GetConsumed() const { return consumed_ot_num_; }

  double GetTime() const { return ote_time_; }

 private:
  std::shared_ptr<yl::Context> ctx_{nullptr};

  bool is_sender_{false};

  bool is_setup_{false};

  std::unique_ptr<yc::OtSendStore> send_ot_ptr_{nullptr};

  std::unique_ptr<yc::OtRecvStore> recv_ot_ptr_{nullptr};

  // runtime record
  uint128_t consumed_ot_num_{0};
  uint128_t ote_num_{0};  // number of invoke ote protocol
  double ote_time_{0.0};  // ms
  uint128_t id_{0};
  static uint128_t yacl_id_;
};

class YaclSsOTeAdapter : public YaclOTeAdapter {
 public:
  // LocalHost or 10000Mbps, set k = 2
  // 1000Mbps, set k = 4
  // 500Mbps, set k = 5
  // 200Mbps, set k = 7
  // 100Mbps or lower, set k = 8
  YaclSsOTeAdapter(const std::shared_ptr<yl::Context>& ctx, bool is_sender,
                   uint64_t k = 2) {
    ctx_ = ctx;
    is_sender_ = is_sender;

    if (is_sender_) {
      ss_sender_ = std::make_unique<yc::SoftspokenOtExtSender>(k);
    } else {
      ss_receiver_ = std::make_unique<yc::SoftspokenOtExtReceiver>(k);
    }

    id_ = yacl_id_;
    ++yacl_id_;
  }

  ~YaclSsOTeAdapter() {
    SPDLOG_DEBUG(
        "[Destructor] SoftspokenAdapter work as {}, total comsume OT {}, "
        "invoke softspoken {}, softspoken time {} ms, {} ms per softspoken , "
        "{} ms per ot ",
        (is_sender_ ? fmt::format("Sender") : fmt::format("Receiver")),
        consumed_ot_num_, ote_num_, ote_time_, ote_time_ / ote_num_,
        ote_time_ / consumed_ot_num_);
  }

  void OneTimeSetup() override;

  void recv_cot(absl::Span<uint128_t> data,
                absl::Span<const uint8_t> choices) override {
    recv_cot(data, VecU8toBitset(choices));
  }

  inline void send_rcot(absl::Span<uint128_t> data) override { send_cot(data); }

  inline void recv_rcot(absl::Span<uint128_t> data,
                        absl::Span<uint8_t> choices) override {
    auto _choices =
        yc::RandBits<yacl::dynamic_bitset<uint128_t>>(data.size(), true);
    BitsettoVecU8(_choices, choices);

    recv_cot(data, _choices);
  }

  // Softspoken ENTRY
  // Correlated Cot with Chosen Choices
  void send_cot(absl::Span<uint128_t> data) override;

  void recv_cot(absl::Span<uint128_t> data,
                const yacl::dynamic_bitset<uint128_t>& choices);

  uint128_t GetDelta() const override { return Delta; }

  uint128_t GetConsumed() const { return consumed_ot_num_; }

  double GetTime() const { return ote_time_; }

 private:
  std::shared_ptr<yl::Context> ctx_{nullptr};

  std::unique_ptr<yc::SoftspokenOtExtSender> ss_sender_{nullptr};

  std::unique_ptr<yc::SoftspokenOtExtReceiver> ss_receiver_{nullptr};

  bool is_sender_{false};

  bool is_setup_{false};

  // Just for test
  uint128_t consumed_ot_num_{0};
  uint128_t ote_num_{0};  // number of invoke ote protocol
  double ote_time_{0.0};  // ms
  // Debug only
  uint128_t id_{0};
  static uint128_t yacl_id_;
};
}  // namespace spu::mpc::cheetah
