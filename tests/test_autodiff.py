import drjit as dr
import pytest
import sys
import math

def make_mystruct(t):
    class MyStruct:
        def __init__(self) -> None:
            self.x = t(1)
            self.y = t(2)
        DRJIT_STRUCT = { 'x': t, 'y': t }
    return MyStruct

@pytest.test_arrays('is_diff,-mask,-shape=()')
def test01_enable_grad(t):
    a = t(1)

    assert not dr.grad_enabled(a)
    dr.enable_grad(a)

    if not dr.is_float_v(t):
        assert not dr.grad_enabled(a)
        return

    assert dr.grad_enabled(a)

    b = (t(1), t(2), t(3))
    assert not dr.grad_enabled(b)
    dr.enable_grad(b)
    assert dr.grad_enabled(b) and \
           dr.grad_enabled(b[0]) and \
           dr.grad_enabled(b[1]) and \
           dr.grad_enabled(b[2])

    b = {'a': t(1), 'b': t(2), 'c': t(3)}
    assert not dr.grad_enabled(b)
    dr.enable_grad(b)
    assert dr.grad_enabled(b) and \
           dr.grad_enabled(b['a']) and \
           dr.grad_enabled(b['b']) and \
           dr.grad_enabled(b['c'])

    a, b, c = t(1), t(2), t(3)
    assert not dr.grad_enabled(a, b, c)
    dr.enable_grad(a, b, c)
    assert dr.grad_enabled(a, b, c) and \
           dr.grad_enabled(a) and \
           dr.grad_enabled(b) and \
           dr.grad_enabled(c)

    if a.ndim > 1:
        assert dr.grad_enabled(a[0])


    MyStruct = make_mystruct(t)
    a = MyStruct()
    assert not dr.grad_enabled(a) and \
           not dr.grad_enabled(a.x) and \
           not dr.grad_enabled(a.y)

    dr.enable_grad(a)

    assert dr.grad_enabled(a) and \
           dr.grad_enabled(a.x) and \
           dr.grad_enabled(a.y)

@pytest.test_arrays('is_diff,float,-shape=()')
def test02_detach(t):
    a = t(1)
    dr.enable_grad(a)
    b = dr.detach(a, preserve_type=False)
    c = dr.detach(a, preserve_type=True)
    assert type(a) is not type(b)
    assert dr.detached_t(type(a)) is type(b)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(b)
    assert not dr.grad_enabled(c)

    MyStruct = make_mystruct(t)
    a = MyStruct()
    dr.enable_grad(a)
    c = dr.detach(a)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(c)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test03_set_grad(t):
    a = t([1, 2, 3])
    dr.set_grad(a, 2.0) # AD tracking not yet enabled
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 0.0)
    dr.enable_grad(a)
    g = dr.grad(a)
    assert len(g) == 3 and dr.allclose(g, 0.0)
    dr.set_grad(a, 2.0)
    g = dr.grad(a)
    assert len(g) == 3 and dr.allclose(g, 2.0)

    dr.set_grad(a, t(3, 4, 5))
    g = dr.grad(a)
    assert len(g) == 3 and dr.allclose(g, [3, 4, 5])

    with pytest.raises(RuntimeError, match='attempted to store a gradient of size 2 into AD variable'):
        dr.set_grad(a, t(1, 2))

    a = t(1)
    dr.enable_grad(a)
    dr.set_grad(a, t(3, 4, 5))
    g = dr.grad(a)
    assert len(g) == 1 and dr.allclose(g, 3+4+5)

    assert type(dr.grad(a)) is type(a)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    Array3f = getattr(sys.modules[t.__module__], 'Array3f')
    a = Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5])
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.set_grad(a, 1.0)
    assert dr.allclose(dr.grad(a), [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    v = Array3f([2, 3, 4], [5, 6, 7], [8, 9, 10])
    dr.set_grad(a, v)
    assert dr.allclose(dr.grad(a), v)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    MyStruct = make_mystruct(t)
    a, b = MyStruct(), MyStruct()
    a.x = t(0, 0, 0)
    a.y = t(0, 0, 0)
    dr.enable_grad(a)
    b.x = t(1, 2, 3)
    b.y = t(4, 5, 6)
    dr.set_grad(a, b)

    c = dr.grad(a)
    assert dr.all(b.x == c.x)
    assert dr.all(b.y == c.y)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test04_accum_grad(t):
    a = t([1, 2, 3])
    dr.accum_grad(a, 2) # AD tracking not yet enabled
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 0)
    dr.enable_grad(a)
    dr.accum_grad(a, 2)
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 2)
    dr.accum_grad(a, 2)
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 4)

    a = t([1])
    dr.enable_grad(a)
    dr.accum_grad(a, [1, 2, 3])
    g = dr.grad(a)
    assert len(a) == 1 and dr.allclose(g, 6)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test05_set_label(t):
    a = t(1.0)
    b = [t(1.0), t(2.0)]
    Array3f = getattr(sys.modules[t.__module__], 'Array3f')
    c = Array3f(1.0, 2.0, 3.0)
    d = make_mystruct(t)()

    assert dr.label(a) is None
    dr.enable_grad(a, b, c, d)
    assert dr.label(a) is None

    dr.set_label(a, 'aa')
    assert dr.label(a) == 'aa'

    dr.set_label(a=a, b=b, c=c, d=d)
    assert dr.label(a) == 'a'
    assert dr.label(b[0]) == 'b_0'
    assert dr.label(b[1]) == 'b_1'
    assert dr.label(c.x) == 'c_0'
    assert dr.label(c.y) == 'c_1'
    assert dr.label(c.z) == 'c_2'
    assert dr.label(d.x) == 'd_x'
    assert dr.label(d.y) == 'd_y'

    with pytest.raises(TypeError, match='incompatible function arguments'):
        dr.set_label(a, 'aa', b=b)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test06_forward_to(t):
  a = t(1.0)
  dr.enable_grad(a)
  b = a * a * 2
  c = a * 2
  dr.set_grad(a, 1.0)
  d = t(4.0) # some detached variable
  grad_b, grad_c, grad_d = dr.forward_to(b, c, d)
  assert dr.allclose(dr.grad(a), 0.0)
  assert dr.allclose(grad_b, 4.0)
  assert dr.allclose(grad_c, 2.0)
  assert dr.allclose(grad_d, 0.0)
  dr.forward_to(b, c, d, dr.ADFlag.Default)

  # Error because the input isn't attached to the AD graph
  with pytest.raises(RuntimeError, match='argument does not depend on the input'):
      dr.forward_to(t(1.0))

  dr.forward_to(t(1.0), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

  # Error because the input isn't a diff array
  with pytest.raises(RuntimeError, match='argument does not depend on the input'):
      dr.forward_to(dr.detached_t(t)(1.0))

  # Trying to call with a different flag
  dr.set_grad(a, 1.0)
  b = a * a * 2
  grad_b = dr.forward_to(b, flags=dr.ADFlag.ClearInterior)
  assert dr.allclose(dr.grad(a), 1.0)
  assert dr.allclose(grad_b, 4.0)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test07_forward_from(t):
    a = t(1.0)

    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.forward_from(a)

    dr.enable_grad(a)
    b = a * a * 2
    dr.forward_from(a)
    assert dr.allclose(dr.grad(a), 0.0)
    assert dr.allclose(dr.grad(b), 4.0)

    b = a * a * 2
    dr.forward_from(a, flags=dr.ADFlag.ClearInterior)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(dr.grad(b), 4.0)

    # Interior gradients are cleared, forwarding again will accumulate gradients
    dr.forward_from(a, flags=dr.ADFlag.ClearEdges)
    assert dr.allclose(dr.grad(b), 8.0)

    # Edges are cleared, forwarding again will do nothing
    dr.forward_from(a, flags=dr.ADFlag.ClearEdges)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(dr.grad(b), 8.0)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test08_backward_to(t):
    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.backward_to(1.0)

    a = t(1.0)

    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.backward_to(a)

    b = t(3.0)
    dr.enable_grad(a, b)
    c = a * b * 2

    dr.set_grad(c, 1.0)
    dr.backward_to(a, flags=dr.ADFlag.ClearVertices)
    assert dr.allclose(dr.grad(a), 6.0)
    assert dr.allclose(dr.grad(b), 0.0)
    assert dr.allclose(dr.grad(c), 0.0)

    dr.set_grad(c, 1.0)
    dr.backward_to(a, b, flags=dr.ADFlag.ClearVertices)
    assert dr.allclose(dr.grad(a), 12.0) # accumulates
    assert dr.allclose(dr.grad(b), 2.0)
    assert dr.allclose(dr.grad(c), 0.0)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test09_backward_from(t):
    a = t(1.0)

    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.backward_from(a)

    dr.enable_grad(a)
    b = a * a * 2
    dr.backward_from(b)
    assert dr.allclose(dr.grad(a), 4.0)
    assert dr.allclose(dr.grad(b), 0.0)

    a = t(1.0)
    dr.enable_grad(a)
    if '64' in t.__name__:
        Array3f = getattr(sys.modules[t.__module__], 'Array3f64')
    else:
        Array3f = getattr(sys.modules[t.__module__], 'Array3f')

    c = Array3f(a)
    dr.backward_from(c)
    print(dr.grad(a))
    assert dr.allclose(dr.grad(a), 3.0)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test10_forward_to_reuse(t):
    a, b, c = t(1), t(2), t(3)
    dr.enable_grad(a, b, c)
    dr.set_grad(a, 10)
    dr.set_grad(b, 100)
    dr.set_grad(c, 1000)

    d, e, f = a + b, a + c, b + c
    g, h, i = d*d, e*e, f*f

    for k in range(2):
        for j, v in enumerate([g, h, i]):
            dr.set_grad(v, 0)
            dr.forward_to(v, flags=dr.ADFlag.ClearInterior)
            assert v == [9, 16, 25][j]
            assert dr.grad(v) == [660, 8080, 11000][j]
            assert dr.grad([g, h, i][(j + 1)%3]) == 0
            assert dr.grad([g, h, i][(j + 2)%3]) == 0
            dr.set_grad(v, t())

@pytest.test_arrays('is_diff,float,shape=(*)')
def test11_backward_to_reuse(t):
    a, b, c = t(1), t(2), t(3)
    dr.enable_grad(a, b, c)

    d, e, f = a + b, a + c, b + c
    g, h, i = d*d, e*e, f*f

    dr.set_grad(g, 10)
    dr.set_grad(h, 100)
    dr.set_grad(i, 1000)

    for k in range(2):
        for j, v in enumerate([a, b, c]):
            dr.backward_to(v, flags=dr.ADFlag.ClearInterior)
            assert dr.grad(v) == [860, 10060, 10800][j]
            assert dr.grad([a, b, c][(j + 1)%3]) == 0
            assert dr.grad([a, b, c][(j + 2)%3]) == 0
            dr.set_grad(v, t())

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test12_mixed_precision_bwd(t):
    t2 = dr.float64_array_t(t)

    a = t(1)
    dr.enable_grad(a)
    b = dr.sin(a)
    c = dr.sin(t2(a))
    d = t2(t(t2(a)))
    e = b + c + d + a
    dr.backward_from(e)
    assert dr.allclose(dr.grad(a), dr.cos(1) * 2 + 2)


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test12_mixed_precision_fwd(t):
    t2 = dr.float64_array_t(t)

    a = t(1)
    dr.enable_grad(a)
    b = dr.sin(a)
    c = dr.sin(t2(a))
    d = t2(t(t2(a)))
    e = b + c + d + a
    dr.forward_from(a)
    assert dr.allclose(dr.grad(e), dr.cos(1) * 2 + 2)


counter = 12

def std_test(name, func, f_in, f_out, grad_out):
    global counter

    if not isinstance(f_in, tuple):
        f_in = (f_in, )

    def test_func(t):
        args = tuple(t(v) for v in f_in)
        dr.enable_grad(args)
        rv = func(*args)
        assert dr.allclose(rv, f_out)
        dr.backward_from(rv, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        assert dr.allclose(dr.grad(args), grad_out)

    query = pytest.test_arrays('is_diff,float,shape=(*)')
    test_func_param = query(test_func)

    globals()[f'test{counter:02}_op_{name}'] = test_func_param
    counter += 1

# Spot-check various builtin operations
std_test('neg', lambda a: -a, 3, -3, (-1))
std_test('add', lambda a, b: a + b, (2, 3), 5, (1, 1))
std_test('sub', lambda a, b: a - b, (2, 3), -1, (1, -1))
std_test('mul', lambda a, b: a * b, (2, 3), 6, (3, 2))
std_test('div', lambda a, b: a / b, (2, 3), 2 / 3, (1 / 3, -2 / 9))
std_test('fma', lambda a, b, c: dr.fma(a, b, c), (3, 7, 11), 32, (7, 3, 1))
std_test('abs_pos', lambda a: abs(a), 3, 3, 1)
std_test('abs_neg', lambda a: abs(a), -3, 3, -1)
std_test('rcp', lambda a: dr.rcp(a), 3, 1/3, -1/9)
std_test('rsqrt', lambda a: dr.rsqrt(a), 3, 1/dr.sqrt(3), -1/(6*dr.sqrt(3)))
std_test('cbrt', lambda a: dr.cbrt(a), 3, 3**(1/3), 1/(3* 3**(2/3)))
std_test('erf', lambda a: dr.erf(a), .2, math.erf(.2), 2*math.exp(-.2**2)/math.sqrt(math.pi))
std_test('log', lambda a: dr.log(a), 2, math.log(2), .5)
std_test('log2', lambda a: dr.log2(a), 2, math.log2(2), 1/(2*math.log(2)))
std_test('exp', lambda a: dr.exp(a), 2, math.exp(2), math.exp(2))
std_test('exp2', lambda a: dr.exp2(a), 2, math.exp2(2), 4*math.log(2))

std_test('sin', lambda a: dr.sin(a), 1, math.sin(1), math.cos(1))
std_test('cos', lambda a: dr.cos(a), 1, math.cos(1), -math.sin(1))
std_test('tan', lambda a: dr.tan(a), 1, math.tan(1), 1/math.cos(1)**2)
std_test('asin', lambda a: dr.asin(a), .5, math.asin(.5), 1/math.sqrt(1 - .5**2))
std_test('acos', lambda a: dr.acos(a), .5, math.acos(.5), -1/math.sqrt(1 - .5**2))
std_test('atan', lambda a: dr.atan(a), .5, math.atan(.5), 1/(1 + .5**2))

std_test('sinh', lambda a: dr.sinh(a), 1, math.sinh(1), math.cosh(1))
std_test('cosh', lambda a: dr.cosh(a), 1, math.cosh(1), math.sinh(1))
std_test('tanh', lambda a: dr.tanh(a), 1, math.tanh(1), 1/math.cosh(1)**2)
std_test('asinh', lambda a: dr.asinh(a), .5, math.asinh(.5), 1/math.sqrt(1 + .5**2))
std_test('acosh', lambda a: dr.acosh(a), 1.5, math.acosh(1.5), 1/math.sqrt(1.5**2 - 1))
std_test('atanh', lambda a: dr.atanh(a), .5, math.atanh(.5), 1/(1 - .5**2))

std_test('sincos_s', lambda a: dr.sincos(a)[0], 1, math.sin(1), math.cos(1))
std_test('sincos_c', lambda a: dr.sincos(a)[1], 1, math.cos(1), -math.sin(1))
std_test('sincosh_s', lambda a: dr.sincosh(a)[0], 1, math.sinh(1), math.cosh(1))
std_test('sincosh_c', lambda a: dr.sincosh(a)[1], 1, math.cosh(1), math.sinh(1))

std_test('atan2_1', lambda a, b: dr.atan2(a, b), (1, 2), math.atan2(1, 2), (2/5, 1/5))
std_test('atan2_2', lambda a, b: dr.atan2(a, b), (-1, 2), math.atan2(-1, 2), (2/5, -1/5))

std_test('round', lambda a: dr.round(a), 1.6, 2.0, 0.0)
std_test('trunc', lambda a: dr.trunc(a), 1.6, 1.0, 0.0)
std_test('ceil', lambda a: dr.ceil(a), 1.6, 2.0, 0.0)
std_test('floor', lambda a: dr.floor(a), 1.6, 1.0, 0.0)
