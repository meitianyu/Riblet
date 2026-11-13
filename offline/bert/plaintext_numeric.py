from dataclasses import dataclass
import os
import numpy as np

if os.environ.get('CDI_ARRAY_BACKEND') == 'jax':
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as xp
else:
    xp = np

S = 1 << 18
MASK = (1 << 64) - 1
REFERENCE_MODEL = {
    'id': 'G9_PLAINTEXT_STATEFUL_EXACT_RING_V1',
    'cryptographic_execution': False,
    'native_joint_randomness_equivalence': False,
    'raw_product': 'exact_modular_product_no_RLWE_error_model',
    'truncation': 'source_signed_heuristic_and_balancing_bit_with_live_local_state',
    'state_refresh': 'ideal_secret_product_only;linear_views_and_public_products_preserve_state',
}


def validate_execution(config):
    if config.get('execution_mode') != 'plaintext_numeric' or config.get('reference_model') != REFERENCE_MODEL['id']:
        raise ValueError('only the explicitly named plaintext numeric reference is allowed')


@dataclass(frozen=True)
class Word:
    value: object
    state: object

    @property
    def shape(self): return self.value.shape

    def reshape(self, *shape): return Word(self.value.reshape(*shape), self.state.reshape(*shape))
    def transpose(self, *axes): return Word(self.value.transpose(*axes), self.state.transpose(*axes))
    def __getitem__(self, index): return Word(self.value[index], self.state[index])
    def __neg__(self): return Word(-self.value, xp.uint64(0) - self.state)


if xp is not np:
    jax.tree_util.register_pytree_node(Word, lambda x: ((x.value, x.state), None), lambda _, xs: Word(*xs))


def public(x):
    v = xp.asarray(x, xp.int64)
    return Word(v, v.astype(xp.uint64))


def as_word(x): return x if isinstance(x, Word) else public(x)


def words(seed, index, shape, domain=0):
    increment=(xp.asarray(index,xp.uint64)+xp.uint64(1))*xp.uint64(0x9E3779B97F4A7C15)+xp.uint64(domain)
    z = xp.asarray(seed, xp.uint64) + increment
    if z.ndim == 0:
        z = z + xp.arange(int(np.prod(shape)), dtype=xp.uint64).reshape(shape)*xp.uint64(0xD1B54A32D192ED03)
    z = (z ^ (z >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)
    return z ^ (z >> xp.uint64(31))


def signed_mul_overflow(a, b):

    a = xp.asarray(a, xp.int64); b = xp.asarray(b, xp.int64)
    aa = xp.where(a < 0, xp.uint64(0)-a.astype(xp.uint64), a.astype(xp.uint64))
    bb = xp.where(b < 0, xp.uint64(0)-b.astype(xp.uint64), b.astype(xp.uint64))
    limit = xp.where((a < 0) != (b < 0), xp.uint64(1 << 63), xp.uint64((1 << 63)-1))
    return (bb != 0) & (aa > limit // xp.maximum(bb, xp.uint64(1)))


class Arithmetic:
    def __init__(self, seed, per_event=False):
        self.seed = seed; self.counter = 0; self.truncation_count = 0
        self.per_event = per_event
        self.flags = xp.zeros_like(seed,dtype=xp.int64) if per_event else xp.int64(0)
        self.warning_flags = xp.int64(0)
        self.hint_warning_count = xp.int64(0); self.active_mask = None

    def mark(self, bad, code):
        if self.active_mask is not None:
            self.warning_flags |= xp.where(xp.any(bad & ~self.active_mask), xp.int64(code), xp.int64(0))
            bad = bad & self.active_mask
        self.flags |= xp.where(bad if self.per_event else xp.any(bad), xp.int64(code), xp.int64(0))

    def input(self, value):
        v = xp.asarray(value, xp.int64)
        if self.per_event: v=xp.broadcast_to(v,np.shape(self.seed))
        a = words(self.seed, self.counter, v.shape); self.counter += 1
        return Word(v, xp.broadcast_to(a, v.shape))

    def trunc(self, x, bits=18, sign='unknown'):
        if not 1 <= bits <= 62 or sign not in ('unknown', 'positive', 'negative'):
            raise ValueError('invalid truncation specification')
        x = as_word(x); a = x.state; b = x.value.astype(xp.uint64)-a
        heuristic = sign == 'unknown'
        h = xp.uint64(1 << 62) if heuristic else xp.uint64(0)
        if heuristic:
            self.mark((x.value < -(1 << 62)) | (x.value >= (1 << 62)), 2)
        else:
            self.hint_warning_count += xp.sum(x.value < 0 if sign == 'positive' else x.value >= 0, dtype=xp.int64)
        a = a + h + xp.uint64(1 << 63)
        ma = a >> xp.uint64(63); mb = b >> xp.uint64(63)
        carry = ma | mb if sign == 'negative' else ma & mb
        balance = (a >> xp.uint64(bits)) & xp.uint64(1)
        offset = xp.uint64(1 << (63-bits))
        heuristic_offset = xp.uint64(1 << (62-bits)) if heuristic else xp.uint64(0)

        w0 = words(self.seed, self.truncation_count, x.shape, 0xA0761D6478BD642F)
        self.truncation_count += 1
        s0 = (a >> xp.uint64(bits)) - (w0 << xp.uint64(64-bits)) - offset + balance - heuristic_offset
        value = (a >> xp.uint64(bits)) + (b >> xp.uint64(bits)) - (carry << xp.uint64(64-bits)) - offset + balance - heuristic_offset
        return Word(value.astype(xp.int64), s0)

    def add(self, a, b):
        a = as_word(a); b = as_word(b); v = a.value+b.value
        self.mark(((a.value ^ v) & (b.value ^ v)) < 0, 1)
        return Word(v, a.state+b.state)

    def local_product(self, x, c):
        x = as_word(x); c = xp.asarray(c, xp.int64)
        self.mark(signed_mul_overflow(x.value, c), 1)
        return Word(x.value*c, x.state*c.astype(xp.uint64))

    def mul_public(self, x, c): return self.trunc(self.local_product(x, c))

    def mul(self, a, b, bits=18, sign='unknown'):
        a = as_word(a); b = as_word(b)
        self.mark(signed_mul_overflow(a.value, b.value), 1)
        return self.trunc(self.input(a.value*b.value), bits, sign)

    def square(self, x): return self.mul(x, x, sign='positive')

    def dot(self, a, b):
        a = as_word(a); b = as_word(b)
        bound = xp.max(xp.abs(a.value.astype(xp.float64)))*xp.max(xp.abs(b.value.astype(xp.float64)))*a.shape[-1]

        self.warning_flags |= xp.where(bound >= float(1 << 62), xp.int64(32), xp.int64(0))
        return self.trunc(self.input(xp.matmul(a.value, b.value)))

    def sum(self, x, axis=-1, keepdims=True):
        return Word(xp.sum(x.value, axis=axis, keepdims=keepdims, dtype=xp.int64),
                    xp.sum(x.state, axis=axis, keepdims=keepdims, dtype=xp.uint64))

    def select(self, condition, a, b):
        a = as_word(a); b = as_word(b)

        return self.input(xp.where(condition, a.value, b.value))

    def less_public(self, x, constants):
        x = as_word(x); a = x.state; b = x.value.astype(xp.uint64)-a
        return [((((a-xp.uint64(int(c) & MASK)) >> xp.uint64(14)) + (b >> xp.uint64(14))) >> xp.uint64(49)) & xp.uint64(1) != 0 for c in constants]


def bit_length(x):
    x = xp.asarray(x, xp.uint64)
    if xp is np:
        out = np.zeros(x.shape, np.int64)
        for shift in range(64): out += (x >> np.uint64(shift)) != 0
        return out
    return 64-jax.lax.clz(x).astype(xp.int64)


def spu_reciprocal(ctx, x, positive=False):
    x = as_word(x); negative = x.value < 0
    absolute = x if positive else ctx.select(negative, -x, x)
    z = bit_length(absolute.value)
    ctx.mark((absolute.value <= 0) | (z > 36), 8)
    factor = ctx.input(xp.left_shift(xp.int64(1), xp.clip(36-z, 0, 63)))
    c = ctx.mul(absolute, factor, sign='positive')
    w = ctx.add(763940, -ctx.local_product(c, 2))
    e = ctx.add(S, -ctx.mul(c, w, sign='positive'))
    r = ctx.mul(w, ctx.add(e, S), sign='positive')
    e = ctx.square(e); r = ctx.mul(r, ctx.add(e, S), sign='positive')
    y = ctx.mul(r, factor, sign='positive'); ctx.mark(y.value <= 0, 16)
    return y if positive else ctx.select(negative, -y, y)


def spu_rsqrt(ctx, x):
    x = as_word(x); z = bit_length(x.value)
    ctx.mark((x.value <= 0) | (x.value >= (1 << 35)), 8)
    factor = ctx.input(xp.left_shift(xp.int64(1), xp.clip(35-z, 0, 63)))
    u = ctx.mul(x, factor); power = u; total = public(xp.zeros_like(x.value))
    for degree, c in enumerate((-4057974, 10085068, -13072088, 6823457), 1):
        if degree > 1: power = ctx.mul(power, u, sign='positive')
        total = ctx.add(total, ctx.local_product(power, c))
    poly = ctx.add(ctx.trunc(total, sign='negative'), 1086023)
    comp = xp.where(z % 2 == 0, xp.int64(370727), xp.int64(262144))*xp.left_shift(xp.int64(1), xp.clip(17-z//2, 0, 63))
    y = ctx.mul(poly, ctx.input(comp), bits=27); ctx.mark(y.value <= 0, 16)
    return y, dict(z_index=z, normalized=u.value, polynomial=poly.value,
                   compensation_word=comp, final_shift=27)
