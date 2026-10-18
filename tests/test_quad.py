import drjit as dr
import pytest
import math


@pytest.test_arrays('shape=(*), float, -float16')
def test01_gauss_legendre(t):
    gl = dr.quad.gauss_legendre
    assert dr.allclose(gl(t, 1), [[0], [2]])
    assert dr.allclose(gl(t, 2), [[-dr.sqrt(1.0/3.0), dr.sqrt(1.0/3.0)], [1, 1]])
    assert dr.allclose(gl(t, 3), [[-dr.sqrt(3.0/5.0), 0, dr.sqrt(3.0/5.0)], [5.0/9.0, 8.0/9.0, 5.0/9.0]])
    assert dr.allclose(gl(t, 4), [[-0.861136, -0.339981, 0.339981, 0.861136], [0.347855, 0.652145, 0.652145, 0.347855]], atol=1e-5)

    nodes, weights = gl(t, 4)
    assert type(nodes) is t and type(weights) is t


@pytest.test_arrays('shape=(*), float, -float16')
def test02_gauss_lobatto(t):
    gl = dr.quad.gauss_lobatto
    assert dr.allclose(gl(t, 2), [[-1, 1], [1.0, 1.0]])
    assert dr.allclose(gl(t, 3), [[-1, 0, 1], [1.0/3.0, 4.0/3.0, 1.0/3.0]])
    assert dr.allclose(gl(t, 4), [[-1, -dr.sqrt(1.0/5.0), dr.sqrt(1.0/5.0), 1], [1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0]])
    assert dr.allclose(gl(t, 5), [[-1, -dr.sqrt(3.0/7.0), 0, dr.sqrt(3.0/7.0), 1], [1.0/10.0, 49.0/90.0, 32.0/45.0, 49.0/90.0, 1.0/10.0]])


@pytest.test_arrays('shape=(*), float, -float16')
def test03_composite_simpson(t):
    cs = dr.quad.composite_simpson
    assert dr.allclose(cs(t, 3), [dr.linspace(t, -1, 1, 3), [1.0/3.0, 4.0/3.0, 1.0/3.0]])
    assert dr.allclose(cs(t, 5), [dr.linspace(t, -1, 1, 5), [.5/3.0, 2/3.0, 1/3.0, 2/3.0, .5/3.0]])


@pytest.test_arrays('shape=(*), float, -float16')
def test04_composite_simpson_38(t):
    cs = dr.quad.composite_simpson_38
    assert dr.allclose(cs(t, 4), [dr.linspace(t, -1, 1, 4), [0.25, 0.75, 0.75, 0.25]])
    assert dr.allclose(cs(t, 7), [dr.linspace(t, -1, 1, 7), [0.125, 0.375, 0.375, 0.25, 0.375, 0.375, 0.125]], atol=1e-6)


@pytest.test_arrays('shape=(*), float, -float16')
def test05_chebyshev(t):
    nodes = dr.quad.chebyshev(t, 4)
    ref = [-math.cos((2*i + 1) / 8 * math.pi) for i in range(4)]
    assert dr.allclose(nodes, ref)


@pytest.test_arrays('shape=(*), float64')
def test06_polynomial_exactness(t):
    # Integrate monomials over [-1, 1] and compare against the analytic result
    def check(nodes, weights, max_degree):
        for k in range(max_degree + 1):
            ref = 0.0 if k % 2 == 1 else 2.0 / (k + 1)
            assert dr.allclose(dr.sum(weights * nodes**k), ref, atol=1e-12)

    for n in [1, 2, 5, 8, 13]:
        check(*dr.quad.gauss_legendre(t, n), 2*n - 1)

    for n in [2, 3, 6, 9, 14]:
        check(*dr.quad.gauss_lobatto(t, n), 2*n - 3)

    check(*dr.quad.composite_simpson(t, 9), 3)
    check(*dr.quad.composite_simpson_38(t, 10), 3)


@pytest.test_arrays('shape=(*), float64')
def test07_large_n(t):
    for rule in [dr.quad.gauss_legendre, dr.quad.gauss_lobatto]:
        nodes, weights = rule(t, 200)
        assert len(nodes) == 200 and len(weights) == 200
        assert dr.allclose(dr.sum(weights), 2, atol=1e-12)
        assert dr.all(weights > 0)
        assert dr.all(nodes[1:] > nodes[:-1])


@pytest.test_arrays('shape=(*), float16')
def test08_float16(t):
    nodes, weights = dr.quad.gauss_legendre(t, 3)
    assert type(nodes) is t
    assert dr.allclose(nodes, [-dr.sqrt(3.0/5.0), 0, dr.sqrt(3.0/5.0)], atol=1e-3)
    assert dr.allclose(weights, [5.0/9.0, 8.0/9.0, 5.0/9.0], atol=1e-3)


@pytest.test_arrays('tensor, float32')
def test09_tensor(t):
    nodes, weights = dr.quad.gauss_legendre(t, 4)
    assert type(nodes) is t and nodes.shape == (4,)
    assert dr.allclose(dr.sum(weights), 2)


def test10_errors():
    from drjit.scalar import ArrayXf, ArrayXi, Array3f
    with pytest.raises(RuntimeError, match='n must be >= 1'):
        dr.quad.gauss_legendre(ArrayXf, 0)
    with pytest.raises(RuntimeError, match='n must be >= 2'):
        dr.quad.gauss_lobatto(ArrayXf, 1)
    with pytest.raises(RuntimeError, match='odd'):
        dr.quad.composite_simpson(ArrayXf, 4)
    with pytest.raises(RuntimeError, match='divisible by 3'):
        dr.quad.composite_simpson_38(ArrayXf, 5)
    with pytest.raises(TypeError, match='floating point'):
        dr.quad.gauss_legendre(ArrayXi, 3)
    with pytest.raises(TypeError, match='dynamically sized'):
        dr.quad.gauss_legendre(Array3f, 3)
