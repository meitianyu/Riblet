// Copyright 2026

#pragma once

#include "libspu/core/value.h"
#include "libspu/mpc/kernel.h"

namespace spu::mpc::cheetah {

class CdiInvert final : public Kernel {
 public:
  static constexpr char kBindName[] = "cdi_invert";
  Kind kind() const override { return Kind::Dynamic; }
  void evaluate(KernelEvalContext* ctx) const override;
};

}
