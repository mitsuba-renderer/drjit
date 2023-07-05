import drjit as dr
import pytest
import sys

@pytest.test_arrays('uint32, shape=(*)')
def test01_slice_index(t):
    with pytest.raises(TypeError):
        dr.slice_index(dtype=int, shape=(1,), indices=(0,))

    with pytest.raises(TypeError):
        dr.slice_index(dtype=dr.scalar.ArrayXi, shape=(1,), indices=(0,))

    t = dr.scalar.ArrayXu

    with pytest.raises(RuntimeError, match='index 20 is out of bounds for axis 0 with size 10'):
        shape, index = dr.slice_index(dtype=t, shape=(10,), indices=(20,))

    with pytest.raises(RuntimeError, match='too many indices'):
        shape, index = dr.slice_index(dtype=t, shape=(10,), indices=(2,3))

    with pytest.raises(RuntimeError, match='unsupported type "str" in slice'):
        shape, index = dr.slice_index(dtype=t, shape=(10,), indices=("foo",))

    def check(shape, indices, shape_out, index_out):
        shape, index = dr.slice_index(dtype=t, shape=shape, indices=indices)
        assert shape == shape_out and dr.all(index == index_out)

    # 1D arrays, simple slice, integer-based, and array-based indexing
    check(shape=(10,), indices=(5,), shape_out=(), index_out=t(5))
    check(shape=(10,), indices=(-2,), shape_out=(), index_out=t(8))
    check(shape=(10,), indices=(slice(0, 10, 2),),
          shape_out=(5,), index_out=t(0, 2, 4, 6, 8))
    check(shape=(10,), indices=(slice(-100, -2, 2),),
          shape_out=(4,), index_out=t(0, 2, 4, 6))
    check(shape=(10,), indices=(slice(100, 0, -2),),
          shape_out=(5,), index_out=t(9, 7, 5, 3, 1))
    check(shape=(10,), indices=(slice(None, None, None),),
          shape_out=(10,), index_out=t(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    check(shape=(10,), indices=(t(0, 2, 4),),
          shape_out=(3,), index_out=t(0, 2, 4))
    check(shape=(10,), indices=(None, t(0, 2, 4), None),
          shape_out=(1, 3, 1), index_out=t(0, 2, 4))
    check(shape=(10,), indices=(),
          shape_out=(10,), index_out=dr.arange(t, 10))
    check(shape=(10,), indices=(Ellipsis,),
          shape_out=(10,), index_out=dr.arange(t, 10))
    check(shape=(10,), indices=(None, Ellipsis),
          shape_out=(1, 10,), index_out=dr.arange(t, 10))
    check(shape=(10,), indices=(Ellipsis, None),
          shape_out=(10, 1), index_out=dr.arange(t, 10))

    # 2D arrays, simple slice and integer-based, and array-based indexing
    check(shape=(3, 7), indices=(2, 5), shape_out=(), index_out=t(7*2 + 5))
    check(shape=(3, 7), indices=(-2, -5), shape_out=(), index_out=t(7*1 + 2))
    check(shape=(3, 7), indices=(slice(None, None, None), 1),
          shape_out=(3,), index_out=t(1, 8, 15))
    check(shape=(3, 7), indices=(slice(None, None, None), 1),
          shape_out=(3,), index_out=t(1, 8, 15))
    check(shape=(3, 7), indices=(1, slice(None, None, None)),
          shape_out=(7,), index_out=t(7, 8, 9, 10, 11, 12, 13))
    check(shape=(3, 7), indices=(slice(0, 3, 3), slice(0, 7, 3)),
          shape_out=(1, 3), index_out=t(0, 3, 6))
    check(shape=(3, 7), indices=(t(0), slice(0, 7, 3)),
          shape_out=(1, 3), index_out=t(0, 3, 6))
    check(shape=(3, 7), indices=(t(0), t(0, 3, 6)),
          shape_out=(1, 3), index_out=t(0, 3, 6))
    check(shape=(3, 7), indices=(2, slice(None, None, None)),
          shape_out=(7,), index_out=t(14, 15, 16, 17, 18, 19, 20))
    check(shape=(3, 7), indices=(slice(None, None, None), 2),
          shape_out=(3,), index_out=t(2, 9, 16))
    check(shape=(3, 7), indices=(slice(None, None, None), t(2)),
          shape_out=(3, 1), index_out=t(2, 9, 16))
    check(shape=(3, 7), indices=(slice(0, 0, 1), t(2)),
          shape_out=(0, 1), index_out=t())
    check(shape=(3, 7), indices=(),
          shape_out=(3, 7), index_out=dr.arange(t, 7*3))
    check(shape=(3, 7), indices=(1,),
          shape_out=(7,), index_out=dr.arange(t, 7)+7)
    check(shape=(3, 7), indices=(1, ...),
          shape_out=(7,), index_out=dr.arange(t, 7)+7)
    check(shape=(3, 7), indices=(...,),
          shape_out=(3, 7), index_out=dr.arange(t, 7*3))
    check(shape=(3, 7), indices=(None, ..., None, 1, None),
          shape_out=(1, 3, 1, 1), index_out=t(1, 8, 15))

@pytest.test_arrays('is_tensor, -bool')
def test02_construct(t):
    v = t()
    ta = dr.array_t(t)

    assert len(v) == 0 and v.ndim == 1 and v.shape == (0,)
    assert type(v.array) is dr.array_t(t) and len(v.array) == 0
    assert str(v) == "[]"

    v = t([1, 2, 3, 4])
    assert len(v) == 4 and v.ndim == 1 and v.shape == (4,)
    assert dr.all(v.array == [1, 2, 3, 4])
    assert str(v) == "[1, 2, 3, 4]"

    v = t(ta(1, 2, 3, 4))
    assert len(v) == 4 and v.ndim == 1 and v.shape == (4,)
    assert dr.all(v.array == [1, 2, 3, 4])
    assert str(v) == "[1, 2, 3, 4]"

    v = t(ta(1, 2, 3, 4), shape=(2, 2))
    assert len(v) == 2 and v.ndim == 2 and v.shape == (2, 2)
    assert dr.all(v.array == [1, 2, 3, 4])
    assert str(v) == "[[1, 2],\n [3, 4]]"

    v = t(ta(1, 2, 3, 4), shape=(1, 4))
    assert len(v) == 1 and v.ndim == 2 and v.shape == (1, 4)
    assert dr.all(v.array == [1, 2, 3, 4])
    assert str(v) == "[[1, 2, 3, 4]]"

    v = t(ta(1, 2, 3, 4), shape=(4, 1))
    assert len(v) == 4 and v.ndim == 2 and v.shape == (4, 1)
    assert dr.all(v.array == [1, 2, 3, 4])
    assert str(v) == "[[1],\n [2],\n [3],\n [4]]"

    v = t([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert len(v) == 2 and v.ndim == 2 and v.shape == (2, 4)
    assert dr.all(v.array == [1, 2, 3, 4, 5, 6, 7, 8])
    assert str(v) == "[[1, 2, 3, 4],\n [5, 6, 7, 8]]"

    if not dr.is_jit_v(t):
        return

    mod = sys.modules[t.__module__]
    v = mod.Array3f([1, 2], [3, 4], [5, 6])
    assert str(v) == "[[1, 3, 5],\n [2, 4, 6]]"

    v = t(v)
    assert len(v) == 3 and v.ndim == 2 and v.shape == (3, 2)
    assert dr.all(v.array == [1, 2, 3, 4, 5, 6])
    assert str(v) == "[[1, 2],\n [3, 4],\n [5, 6]]"

@pytest.test_arrays('-bool, is_tensor')
def test04_binop(t):
    mod = sys.modules[t.__module__]

    with pytest.raises(RuntimeError, match='Incompatible arguments'):
        v = t(1) + mod.Array3f(1, 2, 3)

    v = t(1) + 4
    assert type(v) is t
    assert str(v) == '5'

    v = t(1) + t(4)
    assert type(v) is t
    assert str(v) == '5'
    assert str(t([1, 2, 3]) + t(4)) == '[5, 6, 7]'
    assert str(t([1, 2, 3]) + t([4, 5, 6])) == '[5, 7, 9]'

@pytest.test_arrays('is_tensor, float32, is_jit')
def test05_cast(t, drjit_verbose, capsys):
    ti = dr.int32_array_t(t)
    tu = dr.uint32_array_t(t)
    td = dr.float64_array_t(t)

    v = t(dr.array_t(t)(1, 2, 3))
    assert dr.all(ti(v) == ti(dr.array_t(ti)(1, 2, 3)))
    assert dr.all(tu(v) == tu(dr.array_t(tu)(1, 2, 3)))
    assert dr.all(td(v) == tu(dr.array_t(td)(1, 2, 3)))

    assert capsys.readouterr().out.count("jit_var_cast") == 3
