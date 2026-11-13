import importlib
import unittest


class CdiIntrinsicTest(unittest.TestCase):
    def _api(self):
        try:
            api = importlib.import_module("cdi_impl")
        except ModuleNotFoundError as exc:
            if exc.name != "cdi_impl":
                raise
            api = None
        self.assertIsNotNone(api, "cdi_impl production API is missing")
        return api

    def test_inversion_flattens_only_the_custom_call_boundary(self):
        try:
            import jax
            import jax.numpy as jnp
        except ModuleNotFoundError:
            self.skipTest("JAX is validated by the Bazel/Linux target")
        jax.config.update("jax_enable_x64", True)
        api = self._api()

        def invert(operand, coefficients, routes, public_ks):
            return api.cdi_invert(
                operand,
                coefficients,
                routes,
                public_ks,
                operator="reciprocal",
            )

        traced = jax.make_jaxpr(invert)(
            jnp.ones((5, 2), dtype=jnp.float32),
            jnp.ones((5, 2, 2), dtype=jnp.float32),
            jnp.zeros((5, 2), dtype=jnp.int64),
            jnp.full((5, 2), 3, dtype=jnp.int64),
        )
        custom = [
            equation
            for equation in traced.jaxpr.eqns
            if equation.primitive.name == "cdi_invert"
        ]
        self.assertEqual(len(custom), 1)
        self.assertEqual(custom[0].invars[0].aval.shape, (10,))
        self.assertEqual(custom[0].invars[1].aval.shape, (10, 2))
        self.assertEqual(custom[0].invars[2].aval.shape, (10,))
        self.assertEqual(custom[0].invars[3].aval.shape, (10,))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
