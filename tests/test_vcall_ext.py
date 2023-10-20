import drjit as dr
import vcall_ext as m
import pytest
import re

def get_pkg(t):
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda

def cleanup(s):
    """Remove memory addresses and backend names from a string """
    s = re.sub(r' at 0x[a-fA-F0-9]*',r'', s)
    s = re.sub(r'\.llvm\.',r'.', s)
    s = re.sub(r'\.cuda\.',r'.', s)
    return s

@pytest.test_arrays('float32,is_diff,is_diff,shape=(*)')
def test01_array_operations(t):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    dr.set_log_level(5)
    # Creating objects
    c = dr.zeros(BasePtr, 2)
    assert(str(c) == '[None, None]')
    assert c[0] is None

    c = dr.full(BasePtr, a, 2)
    assert cleanup(str(c)) == '[<vcall_ext.A object>, <vcall_ext.A object>]'
    c = dr.full(BasePtr, b, 2)
    assert cleanup(str(c)) == '[<vcall_ext.B object>, <vcall_ext.B object>]'
    assert c[0] is b and c[1] is b
    c[0] = a
    assert cleanup(str(c)) == '[<vcall_ext.A object>, <vcall_ext.B object>]'

    c = BasePtr(a)
    assert cleanup(str(c)) == '[<vcall_ext.A object>]'
    assert c[0] is a

    c = BasePtr(a, b)
    print(c)
    assert cleanup(str(c)) == '[<vcall_ext.A object>, <vcall_ext.B object>]'
    assert c[0] is a and c[1] is b
    print(c)
    c[0] = b
    c[1] = a
    print(c)
    assert cleanup(str(c)) == '[<vcall_ext.B object>, <vcall_ext.A object>]'
    assert c[0] is b and c[1] is a

    with pytest.raises(TypeError, match=re.escape("unsupported operand type(s) for +: 'BasePtr' and 'BasePtr'")):
        c+c
    assert dr.all(c == c)
