import drjit as dr
import pytest

dr.set_log_level(dr.LogLevel.Info)

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("custom_type_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test01_custom_type(t):
    pkg = get_pkg(t)

    x = pkg.Color3f([1, 5], 2, 3)
    x.r *= 2
    x.g = 4
    assert str(x) == "[[2, 4, 3],\n [10, 4, 3]]"
    assert type(x) is pkg.Color3f

    y = x + x
    assert str(y) == "[[4, 8, 6],\n [20, 8, 6]]"
    assert type(y) is pkg.Color3f

    z = x * dr.value_t(x)(2)
    assert str(z) == "[[4, 8, 6],\n [20, 8, 6]]"
    assert type(z) is pkg.Color3f

    w = x * 2
    assert str(w) == "[[4, 8, 6],\n [20, 8, 6]]"
    assert type(w) is pkg.Color3f


def test02_struct_to_string():
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("custom_type_ext")
    s = m.struct_to_string()

    assert (
        s
        == """Ray[
  time=[0, 0, 0, 0],
  o=[[0, 0, 0],
     [0, 0, 0],
     [0, 3, 0],
     [0, 0, 0]],
  d=[[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
  has_ray_differentials=1
]"""
    )

@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test03_cpp_make_opaque(t):
    pkg = get_pkg(t)
    Float = t

    v = dr.zeros(Float, 7)
    assert v.state == dr.VarState.Literal

    holder = pkg.CustomFloatHolder(v)
    assert holder.value().state == dr.VarState.Literal

    pkg.cpp_make_opaque(holder)
    assert holder.value().state == dr.VarState.Evaluated


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test04_traverse_opaque(t):
    # Tests that it is possible to traverse an opaque C++ object
    pkg = get_pkg(t)
    Float = t

    v = dr.arange(Float, 10)

    a = pkg.CustomA(v)
    assert dr.detail.collect_indices(a) == [v.index]


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test05_traverse_py(t):
    # Tests the implementation of `traverse_py_cb_ro`,
    # used for traversal of PyTrees inside of C++ objects
    Float = t

    v = dr.arange(Float, 10)

    class PyClass:
        def __init__(self, v) -> None:
            self.v = v

    c = PyClass(v)

    result = []

    def callback(index, domain, variant):
        result.append(index)

    dr.detail.traverse_py_cb_ro(c, callback)

    assert result == [v.index]


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test06_trampoline_traversal(t):
    """
    Tests that classes inheriting from trampoline classes are traversed
    automatically.
    """
    pkg = get_pkg(t)
    Float = t

    v = dr.opaque(Float, 0, 3)

    class B(pkg.CustomBase):
        def __init__(self, v) -> None:
            super().__init__()
            self.v = v

        def value(self):
            return self.v

    b = B(v)

    assert dr.detail.collect_indices(b) == [v.index]
