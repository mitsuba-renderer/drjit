import drjit as dr
import pytest
import importlib

@pytest.fixture(scope="module", params=['drjit.cuda.ad', 'drjit.llvm.ad'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test01_leaf_array_t(m):
    assert m.Float == dr.leaf_array_t(m.Float)
    assert m.Float == dr.leaf_array_t(m.Float(1.0))
    assert m.Float == dr.leaf_array_t([m.Float(1)])
    assert m.Float == dr.leaf_array_t({'a': m.Float(1)})
    assert m.Float == dr.leaf_array_t(m.ArrayXf)

    # # Preference for AD types
    assert m.Float == dr.leaf_array_t([m.Float(1), dr.detached_t(m.Float)(1)])

    # # Preference for floating point types
    assert m.Float == dr.leaf_array_t([m.Float(1), m.UInt(1)])

    class MyStruct:
        def __init__(self) -> None:
            self.a = m.Float(1.0)
            self.b = m.Float(2.0)
            self.c = m.Array3f(0.0)
        DRJIT_STRUCT = { 'a': m.Float, 'b': m.Float, 'c': m.Array3f }

    assert m.Float == dr.leaf_array_t(MyStruct)
    assert m.Float == dr.leaf_array_t(MyStruct())

    assert dr.leaf_array_t(float) is None
    assert dr.leaf_array_t([1.0, 2.0]) is None


def test02_expr_t(m):
    for t in [m.Float, m.UInt, m.Array3f, dr.detached_t(m.Float)]:
        assert t == dr.expr_t(t)

    assert dr.expr_t(1.0, m.Float(4.0)) == m.Float
    assert dr.expr_t(m.Array3f, m.Float(4.0)) == m.Array3f
    assert dr.expr_t(m.Array3f, m.ArrayXf, m.Float) == m.ArrayXf
    assert dr.expr_t(m.Array3f, [1, 2, 3]) == m.Array3f
    assert dr.expr_t(m.Array3f, list) == m.Array3f
    assert dr.expr_t(m.Array3f, m.ArrayXf, m.Float) == m.ArrayXf
    assert dr.expr_t(int, float) == float
    assert dr.expr_t(int, int) == int
    assert dr.expr_t(m.Bool, m.Float) == m.Float

    with pytest.raises(TypeError) as ei:
        dr.expr_t(m.Array3f, m.ArrayXf, {})
    assert "expr_t(): incompatible types" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.expr_t(m.Array3f, m.ArrayXf, None)
    assert "expr_t(): incompatible types" in str(ei.value)

    class MyStruct:
        def __init__(self) -> None:
            self.a = m.Float(1.0)
            self.b = m.Float(2.0)
        DRJIT_STRUCT = { 'a': m.Float, 'b': m.Float }

    with pytest.raises(TypeError) as ei:
        dr.expr_t(MyStruct, m.Float)
    assert "expr_t(): incompatible types" in str(ei.value)
