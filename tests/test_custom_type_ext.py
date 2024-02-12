import drjit as dr
import pytest


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
