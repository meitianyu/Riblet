# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for Riblet; see NOTICE.

from .example_binary_impl import example_binary
from .example_impl import example
from .cdi_impl import (
    cdi_scope,
    cdi_material_prepare,
    cdi_invert,
)
from .mjv_impl import (
    mjv_group_online,
    mjv_linear_group_online,
    mjv_linear_group_setup,
    mjv_model_setup,
    mjv_request_preprocess,
)
from .mjv_linear_impl import (
    mjv_consumer_preprocess,
    mjv_linear_begin,
    mjv_linear_end,
)
from .mjv_context_impl import (
    RUNTIME_OBJECT_AUDIT_V1_FIELDS,
    mjv_audit_snapshot,
    mjv_epoch_begin,
    mjv_epoch_end,
    mjv_model_setup_batch,
    mjv_plan_register,
    mjv_linear_manifest,
    mjv_request_preprocess_batch,
)

# DO-NOT-EDIT:ADD_IMPORT
from .spu_gelu_impl import spu_gelu
from .spu_vit_gelu_impl import spu_vit_gelu
from .spu_nexp_impl import spu_neg_exp
from .spu_silu_impl import spu_silu

__all__ = [
    "spu_gelu",
    "spu_vit_gelu",
    "spu_silu",
    "spu_neg_exp",
    "cdi_scope",
    "cdi_material_prepare",
    "cdi_invert",
    "mjv_model_setup",
    "mjv_request_preprocess",
    "mjv_group_online",
    "mjv_linear_group_setup",
    "mjv_linear_group_online",
    "mjv_consumer_preprocess",
    "mjv_linear_begin",
    "mjv_linear_end",
    "mjv_plan_register",
    "mjv_model_setup_batch",
    "mjv_request_preprocess_batch",
    "mjv_epoch_begin",
    "mjv_epoch_end",
    "mjv_audit_snapshot",
    "RUNTIME_OBJECT_AUDIT_V1_FIELDS",
    # "example",
    # "example_binary",
    # DO-NOT-EDIT:EOL
]
