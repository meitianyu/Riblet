import sys
import numpy as np

model, output = sys.argv[1:]
if model == 'bert':
    import plaintext_numeric as p
    from native_mixed import native_exp, native_gelu
else:
    from vit_cdi import reference_numeric as p
    from model import neg_exp as native_exp
    from native_mixed import native_gelu

xp = p.xp


def exercise(values):
    ctx = p.Arithmetic(xp.uint64(1949))
    x = ctx.input(values)
    y = ctx.mul(x, x)
    z = ctx.trunc(ctx.local_product(x, 131071))
    e = native_exp(ctx, x)
    g = native_gelu(ctx, x)
    return dict(x=x.value, x_state=x.state, product=y.value, product_state=y.state,
                trunc=z.value, trunc_state=z.state, exp=e.value, exp_state=e.state,
                gelu=g.value, gelu_state=g.state, flags=ctx.flags,
                counter=xp.asarray(ctx.counter), truncations=xp.asarray(ctx.truncation_count))


values = xp.asarray([-5,-1,0,1,3], xp.int64) * (1 << 18)
if xp is np:
    with np.errstate(over='ignore'):
        result = exercise(values)
else:
    import jax
    if jax.default_backend() != 'cpu' or not jax.config.jax_enable_x64:
        raise RuntimeError('the release reference must use CPU with x64')
    result = jax.device_get(jax.jit(exercise)(values))
if model == 'vit':
    import selected_kernel
    x = xp.asarray([262144,262145,393216,524288], xp.int64)
    seeds = xp.arange(4, dtype=xp.uint64)
    for op, a, b, ks in [('reciprocal',-100000,360000,[0,1,3,10]),
                          ('rsqrt',-131072,393216,[0,1,3,5])]:
        fn = selected_kernel.selected
        if xp is not np:
            fn = jax.jit(fn, static_argnums=(0,))
        with np.errstate(over='ignore'):
            y, valid = fn(op, x, xp.full(4,a,xp.int64), xp.full(4,b,xp.int64),
                          xp.asarray(ks,xp.int64), seeds)
        result[op] = np.asarray(y.value)
        result[op+'_state'] = np.asarray(y.state)
        result[op+'_valid'] = np.asarray(valid)
np.savez(output, **result)
