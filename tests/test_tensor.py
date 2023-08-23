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
    t3f = dr.reinterpret_array_t(mod.Array3f, dr.var_type_v(v))
    v = t3f([1, 2], [3, 4], [5, 6])
    assert str(v) == "[[1, 3, 5],\n [2, 4, 6]]"

    v = t(v)
    assert len(v) == 3 and v.ndim == 2 and v.shape == (3, 2)
    assert dr.all(v.array == [1, 2, 3, 4, 5, 6])
    assert str(v) == "[[1, 2],\n [3, 4],\n [5, 6]]"

    with pytest.raises(TypeError, match='ragged input'):
        v = t([[1, 2, 3, 4], [5, 6, 7, 8], 5])

    v = t(ta(1, 2, 3, 4, 5, 6, 7, 8))
    assert len(v) == 8 and v.ndim == 1 and v.shape == (8,)

    v = t(array=ta(1, 2, 3, 4, 5, 6, 7, 8), shape=(2, 4))
    assert len(v) == 2 and v.ndim == 2 and v.shape == (2, 4)

    with pytest.raises(TypeError, match=r'Input array has the wrong number of entries \(got 4, expected 8\)'):
        v = t(ta(1, 2, 3, 4), (2, 4))

    with pytest.raises(TypeError, match='Input array must be specified'):
        v = t(shape=(3,4))

@pytest.test_arrays('is_tensor, -bool')
def test03_construct_2(t):
    assert dr.all(dr.arange(t, 3) == [0, 1, 2])
    assert dr.all(dr.zeros(t, 3) == [0, 0, 0])
    assert dr.all(dr.full(t, 1, 3) == [1, 1, 1])
    assert dr.all(dr.empty(t, 3).shape == (3,))

    if dr.is_float_v(t):
        assert dr.all(dr.linspace(t, 0, 1, 3) == [0, 0.5, 1])

    v = dr.zeros(t, shape=(1, 2, 3))
    assert v.shape == (1, 2, 3) and dr.all(v.array == 0)
    v = dr.full(t, 1, shape=(1, 2, 3))
    assert v.shape == (1, 2, 3) and dr.all(v.array == 1)

@pytest.test_arrays('-bool, is_tensor')
def test05_binop(t):
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

@pytest.test_arrays('is_tensor, bool, is_jit')
def test06_reduce(t, drjit_verbose, capsys):
    v = t([[True, False], [False, False]])
    v_any = dr.any_nested(v)
    v_all = dr.all_nested(v)
    assert type(v_any) is t and type(v_all) is t
    assert v_any.shape == () and v_all.shape == ()
    assert bool(v_any)
    assert not bool(v_all)
    msg = capsys.readouterr().out
    assert msg.count('jit_var_mem_copy') == 1
    assert msg.count('jit_any') == 1
    assert msg.count('jit_all') == 1

    v_any = dr.any(v)
    msg = capsys.readouterr().out
    assert msg.count('jit_var_or') == 1

    assert dr.all(v_any == t([True, False]))


@pytest.test_arrays('is_tensor, float32, is_jit')
def test07_cast(t, drjit_verbose, capsys):
    ti = dr.int32_array_t(t)
    tu = dr.uint32_array_t(t)
    td = dr.float64_array_t(t)

    v = t(dr.array_t(t)(1, 2, 3))
    assert dr.all(ti(v) == ti(dr.array_t(ti)(1, 2, 3)))
    assert dr.all(tu(v) == tu(dr.array_t(tu)(1, 2, 3)))
    assert dr.all(td(v) == td(dr.array_t(td)(1, 2, 3)))

    msg = capsys.readouterr().out
    assert msg.count("jit_var_cast") == 3
    assert msg.count("jit_all") == 3
    assert msg.count("jit_var_mem_copy") == 4


@pytest.test_arrays('is_tensor, uint32')
def test08_slice(t):
    class Checker:
        """
        Compares a Tensor indexing operation against a NumPy reference
        and asserts if there is a mismatch.
        """
        def __init__(self, shape, tensor_type):
            np = pytest.importorskip("numpy")

            self.shape = shape
            size = np.prod(shape)
            self.array_n = np.arange(size, dtype=np.uint32).reshape(shape)
            self.array_e = tensor_type(dr.arange(dr.array_t(tensor_type), size), shape)

        def __getitem__(self, args):
            np = pytest.importorskip("numpy")

            #  print(type(self.array_e))
            #  print(self.array_e)
            #  print(args)
            ref_n = self.array_n[args]
            ref_e = self.array_e[args]
            assert ref_n.shape == ref_e.shape
            assert np.all(ref_n.ravel() == np.array(ref_e.array))

    c = Checker((10,), t)
    c[:]
    c[3]
    c[1:5]
    c[-5]

    c = Checker((10, 20), t)
    c[:]
    c[0:0]
    c[5, 0]
    c[5, 0:2]
    c[:, 5]
    c[5, :]
    c[:, :]
    c[1:3, 2:7:2]
    c[8:2:-1, 7:0:-1]
    c[0:0, 0:0]

    c = Checker((8, 9, 10, 11), t)
    c[...]
    c[1, ...]
    c[..., 1]
    c[4, ..., 3]
    c[0, 1:3, ..., 3]

    c[None]
    c[..., None]
    c[1, None, ...]
    c[..., None, 1, None]
    c[None, 4, ..., 3, None]

@pytest.test_arrays('is_tensor, -bool')
def test09_broadcast(t):
    np = pytest.importorskip("numpy")

    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                shape = [i, j, k]
                for l in range(len(shape)):
                    shape_2 = list(shape)
                    shape_2[l] = 1
                    array_n1 = np.arange(np.prod(shape),   dtype=np.uint32).reshape(shape)
                    array_n2 = np.arange(np.prod(shape_2), dtype=np.uint32).reshape(shape_2)

                    index_t = dr.uint32_array_t(dr.array_t(t))
                    array_e1 = t(dr.arange(index_t, np.prod(shape)),   tuple(shape))
                    array_e2 = t(dr.arange(index_t, np.prod(shape_2)), tuple(shape_2))

                    out_n = array_n1 + array_n2
                    out_e = array_e1 + array_e2

                    assert out_n.shape == out_e.shape
                    assert np.all(out_n.ravel() == np.array(out_e.array))

    with pytest.raises(RuntimeError, match=r'Operands have incompatible shapes: \(2,\) and \(2, 1\).'):
        dr.zeros(t, 2) + dr.zeros(t, (2, 1))

    with pytest.raises(RuntimeError, match=r'Operands have incompatible shapes: \(2,\) and \(3,\).'):
        dr.zeros(t, 2) + dr.zeros(t, 3)

    with pytest.raises(RuntimeError, match=r'Operands have incompatible shapes: \(3, 2\) and \(2, 3\).'):
        dr.zeros(t, (3, 2)) + dr.zeros(t, (2, 3))

@pytest.test_arrays('is_tensor, -bool')
def test10_inplace(t):
    v1 = t([[1, 2], [4, 5]])
    v2 = t(v1)
    v3 = v1

    v1 += 1
    assert v3 is v1 and v3 is not v2
    assert str(v1) == "[[2, 3],\n [5, 6]]"

    v1 += v2
    assert str(v1) == "[[3, 5],\n [9, 11]]"
    assert v3 is v1 and v3 is not v2

    if dr.is_float_v(t):
        ti = dr.int_array_t(t)

        v1 = ti([[1, 2], [4, 5]])
        v2 = t([[1, 2], [4, 5]])
        v3 = v1
        v1 += v2

        assert v1 is not v2
        assert type(v1) is t
        assert str(v1) == "[[2, 4],\n [8, 10]]"

@pytest.test_arrays('is_tensor, -bool')
def test11_masked_assignment(t):
    v1 = t([[1, 2], [4, 5]])
    v1[v1>4] = 10
    assert str(v1) == "[[1, 2],\n [4, 10]]"

@pytest.test_arrays('matrix, shape=(3, 3), float32')
def test12_convert_tensor_scalar(t):
    mod = sys.modules[t.__module__]
    m = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert str(m) == "[[1, 2, 3],\n [4, 5, 6],\n [7, 8, 9]]"

    tx = mod.TensorXf(m)
    assert str(tx) == "[[1, 2, 3],\n [4, 5, 6],\n [7, 8, 9]]"

@pytest.test_arrays('matrix, shape=(3, 3, *), float32')
def test13_convert_tensor_vectorized(t, drjit_verbose, capsys):
    mod = sys.modules[t.__module__]
    m = t([[[1,8], 2, 3], [4, 5, 6], [7, 8, 9]])

    assert str(m) == "[[[1, 2, 3],\n  [4, 5, 6],\n" \
        "  [7, 8, 9]],\n [[8, 2, 3],\n" \
        "  [4, 5, 6],\n  [7, 8, 9]]]"

    tx = mod.TensorXf(m)
    assert str(tx) == \
        "[[[1, 8],\n  [2, 2],\n" \
        "  [3, 3]],\n [[4, 4],\n" \
        "  [5, 5],\n  [6, 6]],\n" \
        " [[7, 7],\n  [8, 8],\n" \
        "  [9, 9]]]"

    transcript = capsys.readouterr().out
    assert transcript.count("jit_var_scatter") == 9
