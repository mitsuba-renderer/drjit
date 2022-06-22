import drjit as dr
import pytest

def test01_tensor_traits():
    assert dr.array_t(dr.scalar.TensorXf) is dr.scalar.ArrayXf
    assert dr.array_t(dr.scalar.TensorXu) is dr.scalar.ArrayXu
    assert dr.array_t(dr.scalar.TensorXb) is dr.scalar.ArrayXb
    assert dr.value_t(dr.scalar.TensorXf) is float
    assert dr.value_t(dr.scalar.TensorXu) is int
    assert dr.value_t(dr.scalar.TensorXb) is bool
    assert dr.scalar_t(dr.scalar.TensorXf) is float
    assert dr.scalar_t(dr.scalar.TensorXu) is int
    assert dr.scalar_t(dr.scalar.TensorXb) is bool
    assert dr.mask_t(dr.scalar.TensorXf) is dr.scalar.TensorXb
    assert dr.mask_t(dr.scalar.TensorXu) is dr.scalar.TensorXb
    assert dr.mask_t(dr.scalar.TensorXb) is dr.scalar.TensorXb
    assert dr.is_tensor_v(dr.scalar.TensorXf) and not dr.is_tensor_v(dr.scalar.Array3f)
    assert dr.is_float_v(dr.scalar.TensorXf) and not dr.is_float_v(dr.scalar.TensorXu)
    assert not dr.is_integral_v(dr.scalar.TensorXf) and dr.is_integral_v(dr.scalar.TensorXu)

    assert dr.array_t(dr.llvm.TensorXf) is dr.llvm.Float
    assert dr.array_t(dr.llvm.TensorXu) is dr.llvm.UInt
    assert dr.array_t(dr.llvm.TensorXb) is dr.llvm.Bool
    assert dr.value_t(dr.llvm.TensorXf) is float
    assert dr.value_t(dr.llvm.TensorXu) is int
    assert dr.value_t(dr.llvm.TensorXb) is bool
    assert dr.scalar_t(dr.llvm.TensorXf) is float
    assert dr.scalar_t(dr.llvm.TensorXu) is int
    assert dr.scalar_t(dr.llvm.TensorXb) is bool
    assert dr.mask_t(dr.llvm.TensorXf) is dr.llvm.TensorXb
    assert dr.mask_t(dr.llvm.TensorXu) is dr.llvm.TensorXb
    assert dr.mask_t(dr.llvm.TensorXb) is dr.llvm.TensorXb
    assert dr.is_tensor_v(dr.llvm.TensorXf) and not dr.is_tensor_v(dr.llvm.Array3f)
    assert dr.is_float_v(dr.llvm.TensorXf) and not dr.is_float_v(dr.llvm.TensorXu)
    assert not dr.is_integral_v(dr.llvm.TensorXf) and dr.is_integral_v(dr.llvm.TensorXu)


def test02_slice_index():
    with pytest.raises(TypeError):
        dr.slice_index(dtype=int, shape=(1,), indices=(0,))

    with pytest.raises(TypeError):
        dr.slice_index(dtype=dr.scalar.ArrayXi, shape=(1,), indices=(0,))

    tp = dr.scalar.ArrayXu

    with pytest.raises(RuntimeError) as ei:
        shape, index = dr.slice_index(dtype=tp, shape=(10,), indices=(20,))
    assert "index 20 is out of bounds for axis 0 with size 10" in str(ei.value)

    def check(shape, indices, shape_out, index_out):
        shape, index = dr.slice_index(dtype=tp, shape=shape, indices=indices)
        assert shape == shape_out and dr.all(index == index_out)

    # 1D arrays, simple slice, integer-based, and array-based indexing
    check(shape=(10,), indices=(5,), shape_out=(), index_out=tp(5))
    check(shape=(10,), indices=(-2,), shape_out=(), index_out=tp(8))
    check(shape=(10,), indices=(slice(0, 10, 2),),
          shape_out=(5,), index_out=tp(0, 2, 4, 6, 8))
    check(shape=(10,), indices=(slice(-100, -2, 2),),
          shape_out=(4,), index_out=tp(0, 2, 4, 6))
    check(shape=(10,), indices=(slice(100, 0, -2),),
          shape_out=(5,), index_out=tp(9, 7, 5, 3, 1))
    check(shape=(10,), indices=(slice(None, None, None),),
          shape_out=(10,), index_out=tp(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    check(shape=(10,), indices=(tp(0, 2, 4),),
          shape_out=(3,), index_out=tp(0, 2, 4))
    check(shape=(10,), indices=(None, tp(0, 2, 4), None),
          shape_out=(1, 3, 1), index_out=tp(0, 2, 4))
    check(shape=(10,), indices=(),
          shape_out=(10,), index_out=dr.arange(tp, 10))
    check(shape=(10,), indices=(Ellipsis,),
          shape_out=(10,), index_out=dr.arange(tp, 10))
    check(shape=(10,), indices=(None, Ellipsis),
          shape_out=(1, 10,), index_out=dr.arange(tp, 10))
    check(shape=(10,), indices=(Ellipsis, None),
          shape_out=(10, 1), index_out=dr.arange(tp, 10))

    # 2D arrays, simple slice and integer-based, and array-based indexing
    check(shape=(3, 7), indices=(2, 5), shape_out=(), index_out=tp(7*2 + 5))
    check(shape=(3, 7), indices=(-2, -5), shape_out=(), index_out=tp(7*1 + 2))
    check(shape=(3, 7), indices=(slice(None, None, None), 1),
          shape_out=(3,), index_out=tp(1, 8, 15))
    check(shape=(3, 7), indices=(slice(None, None, None), 1),
          shape_out=(3,), index_out=tp(1, 8, 15))
    check(shape=(3, 7), indices=(1, slice(None, None, None)),
          shape_out=(7,), index_out=tp(7, 8, 9, 10, 11, 12, 13))
    check(shape=(3, 7), indices=(slice(0, 3, 3), slice(0, 7, 3)),
          shape_out=(1, 3), index_out=tp(0, 3, 6))
    check(shape=(3, 7), indices=(tp(0), slice(0, 7, 3)),
          shape_out=(1, 3), index_out=tp(0, 3, 6))
    check(shape=(3, 7), indices=(tp(0), tp(0, 3, 6)),
          shape_out=(1, 3), index_out=tp(0, 3, 6))
    check(shape=(3, 7), indices=(2, slice(None, None, None)),
          shape_out=(7,), index_out=tp(14, 15, 16, 17, 18, 19, 20))
    check(shape=(3, 7), indices=(slice(None, None, None), 2),
          shape_out=(3,), index_out=tp(2, 9, 16))
    check(shape=(3, 7), indices=(slice(None, None, None), tp(2)),
          shape_out=(3, 1), index_out=tp(2, 9, 16))
    check(shape=(3, 7), indices=(slice(0, 0, 1), tp(2)),
          shape_out=(0, 1), index_out=tp())
    check(shape=(3, 7), indices=(),
          shape_out=(3, 7), index_out=dr.arange(tp, 7*3))
    check(shape=(3, 7), indices=(1,),
          shape_out=(7,), index_out=dr.arange(tp, 7)+7)
    check(shape=(3, 7), indices=(1, ...),
          shape_out=(7,), index_out=dr.arange(tp, 7)+7)
    check(shape=(3, 7), indices=(...,),
          shape_out=(3, 7), index_out=dr.arange(tp, 7*3))
    check(shape=(3, 7), indices=(None, ..., None, 1, None),
          shape_out=(1, 3, 1, 1), index_out=tp(1, 8, 15))


def test02_tensor_init_basic():
    t = dr.scalar.TensorXf()
    assert t.shape == (0,) and dr.all(t.array == dr.scalar.ArrayXf())
    t = dr.scalar.TensorXf(123)
    assert t.shape == (1,) and dr.all(t.array == dr.scalar.ArrayXf([123]))
    t = dr.scalar.TensorXf([123, 456])
    assert t.shape == (2,) and dr.all(t.array == dr.scalar.ArrayXf([123, 456]))

    t = dr.llvm.TensorXf()
    assert t.shape == (0,) and dr.all(t.array == dr.llvm.Float())
    t = dr.llvm.TensorXf(123)
    assert t.shape == (1,) and dr.all(t.array == dr.llvm.Float([123]))
    t = dr.llvm.TensorXf([123, 456])
    assert t.shape == (2,) and dr.all(t.array == dr.llvm.Float([123, 456]))
