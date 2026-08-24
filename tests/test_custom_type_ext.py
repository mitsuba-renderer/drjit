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
    elif backend == dr.JitBackend.Metal:
        return m.metal


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
    """
    Tests that it is possible to traverse an opaque C++ object.
    """
    pkg = get_pkg(t)
    Float = t

    value = dr.arange(Float, 10)
    base_value = dr.arange(Float, 10)

    a = pkg.CustomA(value, base_value)
    assert dr.detail.collect_indices(a) == [base_value.index, value.index]


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test05_shared_object_traversal(t):
    """
    Tests that an object reachable through several references is traversed
    once.
    """
    pkg = get_pkg(t)
    Float = t

    a = pkg.CustomA(dr.arange(Float, 10), dr.arange(Float, 10) + 1)
    nested = pkg.Nested(a, a)

    assert dr.detail.collect_indices(nested) == dr.detail.collect_indices(a)


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test06_trampoline_traversal(t):
    """
    Tests that classes inheriting from trampoline classes are traversed
    automatically.
    """
    pkg = get_pkg(t)
    Float = t

    value = dr.opaque(Float, 0, 3)
    base_value = dr.opaque(Float, 1, 3)

    class B(pkg.CustomBase):
        def __init__(self, value, base_value) -> None:
            super().__init__(base_value)
            self._value = value

        def value(self):
            return self._value

    b = B(value, base_value)

    assert dr.detail.collect_indices(b) == [base_value.index, value.index]

@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test07_nested_traversal(t):
    """
    Test traversal of nested objects, and more specifically the traversal of
    ``std::vector<std::pair<nb::ref<Object>, size_t>>`` members.
    """
    pkg = get_pkg(t)
    Float = t

    value = dr.arange(Float, 10) + 0
    base_value = dr.arange(Float, 10) + 1

    a = pkg.CustomA(value, base_value)

    value = dr.arange(Float, 10) + 2
    base_value = dr.arange(Float, 10) + 3

    b = pkg.CustomA(value, base_value)

    nested = pkg.Nested(a, b)

    indices_a = dr.detail.collect_indices(a)
    indices_b = dr.detail.collect_indices(b)
    indices_nested = dr.detail.collect_indices(nested)

    assert indices_nested == indices_a + indices_b

@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test08_custom_type_refcycle(t):
    """
    Tests that it is possible to collect indices from PyTrees with refcycles
    through C++ objects. Each object is visited once.
    """
    pkg = get_pkg(t)
    Float = t

    value = dr.opaque(Float, 0, 3)
    base_value = dr.opaque(Float, 1, 3)

    class B(pkg.CustomBase):
        def __init__(self, value, base_value) -> None:
            super().__init__(base_value)
            self._value = value

        def value(self):
            return self._value

    class C(pkg.CustomBase):
        def __init__(self, value, base_value, ref) -> None:
            super().__init__(base_value)
            self._value = value
            self._ref = ref

        def value(self):
            return self._value

    # Construct a reference cycle
    b = B(value, base_value)
    c = C(value, base_value, b)
    b.child = c

    assert dr.detail.collect_indices(b) == [
        base_value.index, value.index, base_value.index, value.index
    ]


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test09_python_subclass_loop_state(t):
    """
    Tests that the Dr.Jit attributes of a Python subclass of a C++ class are
    tracked as state of a symbolic loop.
    """
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)

    class Holder(pkg.CustomBase):
        def __init__(self, value, base_value):
            super().__init__(base_value)
            self.v = value

    def body(i, h):
        h.v = h.v + 2
        return i + 1, h

    results = []
    for symbolic in [False, True]:
        h = Holder(dr.arange(t, 4), dr.full(t, 2, 4))
        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, symbolic):
            _, h = dr.while_loop((UInt32(0), h), lambda i, h: i < 3, body)
        results.append(h.v)

    assert dr.all(results[0] == dr.arange(t, 4) + 6)
    assert dr.all(results[1] == results[0])


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test10_python_subclass_traversal(t):
    """
    Tests that a Python subclass of a C++ class is traversed on both sides:
    the members of the C++ base and the Python attributes.
    """
    pkg = get_pkg(t)

    class Holder(pkg.CustomBase):
        def __init__(self, value, base_value):
            super().__init__(base_value)
            self.v = value

    value, base_value = dr.arange(t, 4), dr.arange(t, 4) * 2
    h = Holder(value, base_value)
    assert dr.detail.collect_indices(h) == [base_value.index, value.index]


@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test11_python_subclass_as_child_loop_state(t):
    """
    Tests that the Python attributes of a Python subclass instance that is
    reached through another C++ object are tracked as symbolic loop state.
    """
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)

    class Holder(pkg.CustomBase):
        def __init__(self, value, base_value):
            super().__init__(base_value)
            self.v = value

    def body(i, n):
        h = n.child(0)
        h.v = h.v + 2
        return i + 1, n

    results = []
    for symbolic in [False, True]:
        h = Holder(dr.arange(t, 4), dr.full(t, 2, 4))
        n = pkg.Nested(h, h)
        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, symbolic):
            _, n = dr.while_loop((UInt32(0), n), lambda i, n: i < 3, body)
        results.append(n.child(0).v)

    assert dr.all(results[0] == dr.arange(t, 4) + 6)
    assert dr.all(results[1] == results[0])



@pytest.test_arrays("float32,-diff,shape=(*),jit")
def test12_python_subclass_transform_in_place(t):
    """
    Tests that transformations (``dr.detach()``, ``dr.grad()``) treat a
    Python subclass of a C++ class like the C++ object itself: the object is
    returned as is, and its attributes are left alone.
    """
    pkg = get_pkg(t)

    class Holder(pkg.CustomBase):
        def __init__(self, value, base_value):
            super().__init__(base_value)
            self.v = value

    v = dr.arange(t, 4)
    h = Holder(v, dr.arange(t, 4) + 1)

    assert dr.detach(h) is h and h.v is v
    assert dr.grad(h) is h and h.v is v
