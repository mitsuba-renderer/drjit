import drjit as dr
import sys
import pytest

@pytest.test_arrays('shape=(*), float32', 'shape=(*), float64')
def test01_pcg32_basic(t):
    m = sys.modules[t.__module__]
    f32 = m.PCG32().next_float32()
    f64 = m.PCG32().next_float64()
    assert f32 == 0.10837864875793457
    assert f64 == 0.10837870510295033

    f32_n = m.PCG32().next_float32_normal()
    f64_n = m.PCG32().next_float64_normal()
    assert f32_n == -1.2351967096328735
    assert f64_n == -1.235196513088357

    if dr.is_jit_v(t):
        f = m.PCG32().next_float(t)
        f_n = m.PCG32().next_float_normal(t)

        if dr.type_v(t) == dr.VarType.Float32:
            assert f == f32
            assert f_n == f32_n
        else:
            assert f == f64
            assert f_n == f64_n
    else:
        f = m.PCG32().next_float(float)
        f_n = m.PCG32().next_float_normal(float)
        assert f == f64
        assert f_n == f64_n


@pytest.test_arrays('shape=(*), float32')
def test02_pcg32_prev_sample(t):
    m = sys.modules[t.__module__]
    pcg = m.PCG32()

    uint32 = pcg.next_uint32()
    uint32_prev = pcg.prev_uint32()
    assert uint32 == uint32_prev

    uint64 = pcg.next_uint64()
    uint64_prev = pcg.prev_uint64()
    assert uint64 == uint64_prev

    f32 = pcg.next_float32()
    f32_prev = pcg.prev_float32()
    assert f32 == f32_prev

    f64 = pcg.next_float64()
    f64_prev = pcg.prev_float64()
    assert f64 == f64_prev

    f16 = pcg.next_float16()
    f16_prev = pcg.prev_float16()
    assert f16 == f16_prev

    f16_n = pcg.next_float16_normal()
    f16_n_prev = pcg.prev_float16_normal()
    assert f16_n == f16_n_prev

    f32_n = pcg.next_float32_normal()
    f32_n_prev = pcg.prev_float32_normal()
    assert f32_n == f32_n_prev

    f64_n = pcg.next_float64_normal()
    f64_n_prev = pcg.prev_float64_normal()
    assert f64_n == f64_n_prev

    mask = m.Bool([True, False])
    uint32_masked = pcg.next_uint32(mask)
    uint32_masked_prev = pcg.prev_uint32(mask)
    assert dr.all(uint32_masked == uint32_masked_prev)

    uint64_masked = pcg.next_uint64(mask)
    uint64_masked_prev = pcg.prev_uint64(mask)
    assert dr.all(uint64_masked == uint64_masked_prev)

    f16_masked = pcg.next_float16(mask)
    f16_masked_prev = pcg.prev_float16(mask)
    assert dr.all(f16_masked == f16_masked_prev)

    f32_masked = pcg.next_float32(mask)
    f32_masked_prev = pcg.prev_float32(mask)
    assert dr.all(f32_masked == f32_masked_prev)

    f64_masked = pcg.next_float64(mask)
    f64_masked_prev = pcg.prev_float64(mask)
    assert dr.all(f64_masked == f64_masked_prev)

    f16_n_masked = pcg.next_float16_normal(mask)
    f16_n_masked_prev = pcg.prev_float16_normal(mask)
    assert dr.all(f16_n_masked == f16_n_masked_prev)

    f32_n_masked = pcg.next_float32_normal(mask)
    f32_n_masked_prev = pcg.prev_float32_normal(mask)
    assert dr.all(f32_n_masked == f32_n_masked_prev)

    f64_n_masked = pcg.next_float64_normal(mask)
    f64_n_masked_prev = pcg.prev_float64_normal(mask)
    assert dr.all(f64_n_masked == f64_n_masked_prev)

    if dr.is_jit_v(t):
        f = pcg.next_float(t)
        f_prev = pcg.prev_float(t)
        assert dr.all(f == f_prev)

        f_n = pcg.next_float_normal(t)
        f_n_prev = pcg.prev_float_normal(t)
        assert dr.all(f_n == f_n_prev)
    else:
        f = pcg.next_float(float)
        f_prev = pcg.prev_float(float)
        assert f == f_prev

        f_n = pcg.next_float_normal(float)
        f_n_prev = pcg.prev_float_normal(float)
        assert dr.all(f_n == f_n_prev)


@pytest.test_arrays('shape=(4, *), uint32', 'shape=(4), uint32')
def test03_philox(t):
    # Test data from OpenRAND

    ref = \
        ((0x5d5ce31a, 0x44c9c0bc, 0x4e3d18c6, 0x5d1fb025),
         (0x35a82af1, 0x9bf3750e, 0xde264205, 0xff8aa6c9),
         (0xad537add, 0x459b3aaf, 0x3f5c4716, 0x23ac1969),
         (0x4bfca5c6, 0xcf907aaa, 0xafcebe2d, 0xdd7dc9de),
         (0x4a32c886, 0xd9762338, 0xc86f6072, 0x9067c5db),
         (0xe494fe2c, 0x9674b214, 0x65493727, 0x6385a822),
         (0x7dc9de38, 0xb1cb2a5c, 0x3037f445, 0x7d2a1a),
         (0x91375fbf, 0x582efbd0, 0x2adef277, 0x8aae3eef),
         (0xdbadd0bd, 0x74b615dd, 0x216a0f2c, 0x7761df5b),
         (0x5e709dea, 0x45f0edeb, 0x7c7b296b, 0x2d2fd83c),
         (0x4a1ae0aa, 0x102ec9ca, 0x1e247d26, 0x514a3826),
         (0x3d196f14, 0x10169eeb, 0x272f0272, 0xbb259fc8),
         (0xaf9c9024, 0x5ca87d4, 0x33868095, 0x124b6a37),
         (0x6f70a947, 0xb5e70e52, 0x79112edb, 0x26a73257),
         (0xedfaa927, 0x53a0af72, 0x6c866deb, 0xad913511),
         (0x4fbe687b, 0x75380bcf, 0x8a3178ee, 0xc3533b8),
         (0x75bdc2d8, 0x1a3097f, 0x156cb857, 0xefea48b3),
         (0xa1060344, 0x6543d5a5, 0xaae13a43, 0x2cad2bc6))

    m = sys.modules[t.__module__]

    for i in range(3):
        for j in range(3):
            rng = m.Philox4x32(seed=i, counter_0=j, counter_1=0x12345, counter_2=0xAAAAAAAA, iterations=10)
            offset = (i*3+j)*2
            r0 = t(ref[offset+0])
            r1 = t(ref[offset+1])
            v0 = rng.next_uint32x4()
            v1 = rng.next_uint32x4()
            assert dr.all(v0 == r0)
            assert dr.all(v1 == r1)


@pytest.test_arrays('shape=(*), float32',
                    'shape=(*, *), float32',
                    'tensor, float32',
                    'shape=(3), float32')
def test04_rng_shape(t):
    rng = dr.rng()
    incompat = 'could not construct output: the provided "shape" and "dtype" parameters are incompatible.'

    if not dr.is_dynamic_v(t):
        x = rng.random(t, 3)
        assert type(x) is t
        assert x.shape[-1] == 3
        with pytest.raises(RuntimeError, match=incompat):
            rng.random(t, 4)
    elif dr.depth_v(t) > 1 and dr.size_v(t) == dr.Dynamic:
        x = rng.random(t, (2, 15))
        assert type(x) is t
        assert x.shape == (2, 15)
        with pytest.raises(RuntimeError, match=incompat):
            rng.random(t, (2, 3, 4))
    else:
        x = rng.random(t, 15)
        assert type(x) is t
        assert x.shape[-1] == 15

        if not dr.is_tensor_v(t):
            with pytest.raises(RuntimeError, match=incompat):
                rng.random(t, (2, 3))

    if dr.is_tensor_v(t):
        x = rng.random(t, (15, 20))
        assert type(x) is t
        assert x.shape == (15, 20)

@pytest.test_arrays('shape=(*), float32')
def test06_rng_seed(t):
    # Test that same seed produces same sequence
    rng1 = dr.rng(seed=0)
    v0 = rng1.random(t, 10)
    rng1_c = rng1.clone()
    v1 = rng1.random(t, 10)
    v1_2 = rng1_c.random(t, 10)

    rng2 = dr.rng(seed=0)
    v2 = rng2.random(t, 10)
    v3 = rng2.random(t, 10)

    assert dr.all(v0 == v2) and dr.all(v1 == v3) and dr.all(v1 == v1_2)
    assert dr.all(v0 != v1)

    rng3 = dr.rng(seed=5)
    v4 = rng3.random(t, 10)

    assert dr.all(v4 != v0)


@pytest.test_arrays('shape=(*), float')
@pytest.mark.parametrize('symbolic', (True, False))
def test06_rng_distr(t, symbolic):
    m = sys.modules[t.__module__]
    rng = dr.rng(symbolic=symbolic, seed = m.UInt32(0))
    x = rng.random(t, 100)
    assert type(x) is t
    assert dr.all((x >= 0) & (x < 1))

    x = rng.uniform(t, 100, low=10, high=20)
    assert dr.all((x >= 10) & (x < 20))

    x = rng.normal(t, 10000)
    assert dr.allclose(dr.mean(x), 0, atol = 5e-2)
    assert dr.allclose(dr.mean(dr.square(x)), 1, atol = 5e-2)

    x = rng.normal(t, 10000, loc=2, scale=3)
    assert dr.allclose(dr.mean(x), 2, atol = 5e-2)
    if dr.type_v(x) != dr.VarType.Float16:
        assert dr.allclose(dr.mean(dr.square(x-2)), 9, atol = 5e-1)

@pytest.test_arrays('shape=(*), float, jit')
@pytest.mark.parametrize('seed_mode', (0, 1))
@dr.syntax
def test06_rng_symbolic(t, seed_mode):
    """Test symbolic recording of dr.rng() operations"""
    Index = dr.uint32_array_t(t)
    if seed_mode == 0:
        seed = Index(0)
    else:
        seed = 0

    rng = dr.rng(seed=seed)
    i = Index(0)
    while i < 10:
        i += 1
        if seed_mode == 1:
            with pytest.raises(RuntimeError) as e:
                rng.uniform(t, 1)
            expected_error = 'To generate random numbers within a symbolic loop, you must initialize the Generator with a seed of the underlying JIT backend, e.g.: rng = dr.rng(seed=dr.cuda.UInt32(0))'
            assert expected_error in str(e)
        else:
            rng.uniform(t, 1)

    if seed_mode == 0:
        x = rng.uniform(t, 1)
        assert dr.allclose(x, 0.467728)
