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
