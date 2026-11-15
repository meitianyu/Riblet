// Copyright 2026

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libspu/core/value.h"
#include "libspu/mpc/cheetah/mjv_linear.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::cheetah {

struct MJVStats {
  int64_t model_setup_count = 0;
  int64_t request_preprocess_count = 0;
  int64_t online_count = 0;
  int64_t right_operand_opened_elements = 0;
  int64_t joint_view_opened_elements = 0;
  int64_t correlation_elements = 0;
};

class MJVContextState final : public State {
 public:
  static constexpr char kBindName[] = "MJVContext";

  struct RuntimeFamilyBinding {
    std::string plan_sha256;
    int64_t model_epoch = 0;
    int64_t request_epoch = 0;
    std::string physical_family_id;

    bool isBound() const { return !plan_sha256.empty(); }
  };

  struct ModelContext {
    NdArrayRef mask;
    NdArrayRef opened_delta;
    Sizes output_widths;
    int64_t sum_r = 0;
    int64_t rho = 0;
    DataType dtype = DT_INVALID;
    LinearKernelFamily linear_family = LinearKernelFamily::kMatMulAA;
    RuntimeFamilyBinding runtime_family;
  };

  struct RequestContext {
    int64_t model_id = 0;
    NdArrayRef correlation;
    int64_t rows = 0;
    RuntimeFamilyBinding runtime_family;
  };

  void RequireNewModel(int64_t model_id) const;
  void StoreModel(int64_t model_id, ModelContext model);
  ModelContext GetModel(int64_t model_id) const;
  void ReleaseModelEpoch(const std::string& plan_sha256,
                         int64_t model_epoch);

  void RequireNewRequest(int64_t model_id, int64_t request_id) const;
  void StoreRequest(int64_t request_id, RequestContext request);
  RequestContext ConsumeRequest(int64_t model_id, int64_t request_id,
                                int64_t opened_elements);

  int64_t LiveRequestCount() const;
  MJVStats GetStats() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<int64_t, ModelContext> models_;
  std::unordered_set<int64_t> used_model_ids_;
  std::unordered_map<int64_t, RequestContext> requests_;
  std::unordered_set<int64_t> used_request_ids_;
  MJVStats stats_;
};

class MJVModelSetup final : public Kernel {
 public:
  static constexpr char kBindName[] = "mjv_model_setup";

  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

class MJVRequestPreprocess final : public Kernel {
 public:
  static constexpr char kBindName[] = "mjv_request_preprocess";

  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

class MJVGroupOnline final : public Kernel {
 public:
  static constexpr char kBindName[] = "mjv_group_online";

  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

}
