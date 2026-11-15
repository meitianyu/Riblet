def extend_selection(routes, coefficients, physical_length, *, jnp):
    n, sites = routes.shape
    if coefficients.shape != (n, sites, 2) or not n <= physical_length <= 128:
        raise ValueError("invalid selected parameter layout")
    if n == physical_length:
        return routes, coefficients
    return (
        jnp.concatenate(
            [routes, jnp.ones((physical_length - n, sites), dtype=routes.dtype)], axis=0
        ),
        jnp.concatenate(
            [coefficients, jnp.zeros((physical_length - n, sites, 2), dtype=coefficients.dtype)],
            axis=0,
        ),
    )


def pad_ks(ks, effective_length, *, token_axis, jnp):
    dims = [1] * ks.ndim
    dims[token_axis] = ks.shape[token_axis]
    valid = (jnp.arange(ks.shape[token_axis]) < effective_length).reshape(dims)
    return jnp.where(valid, ks, jnp.zeros_like(ks))
