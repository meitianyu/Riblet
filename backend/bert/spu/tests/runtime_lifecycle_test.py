# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for Riblet; see NOTICE.

import copy
import gc
import os
import time
import unittest

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import spu.utils.distributed as ppd
import spu.utils.distributed_impl as ppd_impl
from spu.tests.utils import get_free_port
from spu.utils.polyfill import Process


def make_cluster_defs():
    nodes_def = {
        "node:0": f"127.0.0.1:{get_free_port()}",
        "node:1": f"127.0.0.1:{get_free_port()}",
    }
    devices_def = {
        "SPU": {
            "kind": "SPU",
            "config": {
                "node_ids": ["node:0", "node:1"],
                "spu_internal_addrs": [
                    f"127.0.0.1:{get_free_port()}",
                    f"127.0.0.1:{get_free_port()}",
                ],
                "runtime_config": {"protocol": "CHEETAH", "field": "FM64"},
            },
        }
    }
    return nodes_def, devices_def


def runtime_snapshot(server, device_name):
    return {
        "process_id": os.getpid(),
        "generation": server._locals.get(f"{device_name}-generation"),
        "last_transition": server._locals.get(f"{device_name}-last-transition"),
        "runtime_present": f"{device_name}-rt" in server._locals,
        "io_present": f"{device_name}-io" in server._locals,
    }


def remove_runtime_component(server, device_name, component):
    server._locals.pop(f"{device_name}-{component}", None)


def set_runtime_generation(server, device_name, generation):
    server._locals[f"{device_name}-generation"] = generation


class RuntimeLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.nodes_def, self.devices_def = make_cluster_defs()
        self.workers = []
        for node_id in self.nodes_def:
            worker = Process(target=ppd.RPC.serve, args=(node_id, self.nodes_def))
            worker.start()
            self.workers.append(worker)
        time.sleep(3)
        ppd.init(self.nodes_def, self.devices_def)

    def tearDown(self):
        for worker in self.workers:
            worker.kill()

    def transition(self, transition_id):
        next_devices = copy.deepcopy(self.devices_def)
        next_devices["SPU"]["config"][
            "runtime_retirement_transition"
        ] = transition_id
        return ppd.init(self.nodes_def, next_devices)

    def snapshots(self, device=None):
        device = device or ppd.device("SPU")
        return [
            client.run_return(runtime_snapshot, "SPU")
            for client in device.node_clients
        ]

    def complete_one_computation(self, device=None):
        device = device or ppd.device("SPU")
        result = device(jnp.add)(np.array([1, 2]), np.array([3, 4]))
        npt.assert_equal(ppd.get(result), np.array([4, 6]))
        del result
        gc.collect()

    def test_explicit_mode_boundary_retires_runtime_without_restarting_workers(self):
        first_device = ppd.device("SPU")
        self.complete_one_computation(first_device)
        first_snapshots = self.snapshots(first_device)

        self.transition("cell-0-to-cell-1")
        second_snapshots = self.snapshots()
        transition_audit = ppd.device("SPU").runtime_transition_audit

        self.assertIs(transition_audit["native_heap_trimmed"], True)

        self.assertEqual(
            [snapshot["process_id"] for snapshot in second_snapshots],
            [snapshot["process_id"] for snapshot in first_snapshots],
        )
        self.assertEqual(
            [snapshot["generation"] for snapshot in second_snapshots],
            [2, 2],
        )
        self.assertEqual(
            [snapshot["last_transition"] for snapshot in second_snapshots],
            ["cell-0-to-cell-1", "cell-0-to-cell-1"],
        )
        self.complete_one_computation()

    def test_old_spu_handle_fails_after_generation_transition(self):
        first_device = ppd.device("SPU")
        old_function = first_device(jnp.add)
        self.complete_one_computation(first_device)

        self.transition("cell-0-to-cell-1")

        with self.assertRaisesRegex(RuntimeError, "retired SPU handle"):
            old_function(np.array([1]), np.array([2]))
        with self.assertRaisesRegex(RuntimeError, "retired SPU handle"):
            first_device.get(None)

    def test_retirement_rejects_partial_runtime_state_missing_io(self):
        client = ppd.device("SPU").node_clients[0]
        client.run_return(
            remove_runtime_component, "SPU", "io"
        )
        with self.assertRaisesRegex(Exception, "incomplete state"):
            client.run_return(
                ppd_impl.builtin_spu_retire, "SPU", "missing-io", 0
            )

    def test_retirement_rejects_partial_runtime_state_missing_runtime(self):
        client = ppd.device("SPU").node_clients[0]
        client.run_return(
            remove_runtime_component, "SPU", "rt"
        )
        with self.assertRaisesRegex(Exception, "incomplete state"):
            client.run_return(
                ppd_impl.builtin_spu_retire, "SPU", "missing-runtime", 0
            )

    def test_retirement_rejects_fully_missing_runtime_state(self):
        client = ppd.device("SPU").node_clients[0]
        client.run_return(remove_runtime_component, "SPU", "rt")
        client.run_return(remove_runtime_component, "SPU", "io")
        with self.assertRaisesRegex(Exception, "incomplete state"):
            client.run_return(
                ppd_impl.builtin_spu_retire, "SPU", "missing-both", 0
            )

    def test_duplicate_transition_identity_fails_closed(self):
        self.transition("cell-0-to-cell-1")
        with self.assertRaisesRegex(RuntimeError, "duplicate transition"):
            self.transition("cell-0-to-cell-1")

    def test_retirement_before_reveal_or_live_object_completion_fails(self):
        live_result = ppd.device("SPU")(jnp.add)(
            np.array([1, 2]), np.array([3, 4])
        )
        with self.assertRaisesRegex(RuntimeError, "live SPU object"):
            self.transition("cell-0-to-cell-1")
        npt.assert_equal(ppd.get(live_result), np.array([4, 6]))

    def test_all_parties_must_acknowledge_same_generation(self):
        ppd.device("SPU").node_clients[1].run_return(
            set_runtime_generation, "SPU", 9
        )
        with self.assertRaisesRegex(RuntimeError, "acknowledgement mismatch"):
            self.transition("cell-0-to-cell-1")

    def test_retirement_requires_all_parties_to_acknowledge_native_heap_trim(self):
        device = ppd.device("SPU")
        acknowledgements = [
            {
                "device_name": "SPU",
                "party_index": 0,
                "transition_id": "cell-0-to-cell-1",
                "generation": 1,
                "native_heap_trimmed": True,
            },
            {
                "device_name": "SPU",
                "party_index": 1,
                "transition_id": "cell-0-to-cell-1",
                "generation": 1,
                "native_heap_trimmed": False,
            },
        ]

        with self.assertRaisesRegex(
            RuntimeError, "native heap trim acknowledgement mismatch"
        ):
            device._validate_runtime_acknowledgements(
                acknowledgements,
                transition_id="cell-0-to-cell-1",
            )


if __name__ == "__main__":
    unittest.main()
