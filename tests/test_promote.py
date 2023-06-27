import drjit as dr
import pytest

@pytest.test_arrays('float32,shape=(3)', "float32,shape=(*)", "float32,shape=(3, *)")
def test13_binop_promote_broadcast(t):
    print('a')
    x = t(10, 100, 1000) + 1
    print('b')
    assert type(x) is t and dr.all(x == t(11, 101, 1001))
    print('c')

    x = 1 + t(10, 100, 1000)
    assert type(x) is t and dr.all(x == t(11, 101, 1001))

    x = t(10, 100, 1000) + (1, 2, 3)
    assert type(x) is t and dr.all(x == t(11, 102, 1003))

    x = t(10, 100, 1000) + [1, 2, 3]
    assert type(x) is t and dr.all(x == t(11, 102, 1003))

    x = (1, 2, 3) + t(10, 100, 1000)
    assert type(x) is t and dr.all(x == t(11, 102, 1003))
#
#
#
#    x = [1, 2, 3] + t(10, 100, 1000)
#    assert type(x) is t and dr.all(x == t(11, 102, 1003))
#    x = s.Array3f(10, 100, 1000) + 1
#    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 101, 1001))
#    x = 1 + s.Array3f(10, 100, 1000)
#    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 101, 1001))
#    x = s.Array3f(10, 100, 1000) + (1, 2, 3)
#    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
#    x = (1, 2, 3) + s.Array3f(10, 100, 1000)
#    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
#    x = [1, 2, 3] + s.Array3f(10, 100, 1000)
#    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
#    x = s.Array3f(10, 100, 1000) + [1, 2, 3]
#    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
#
#    x = l.Float(10, 100, 1000) + 1
#    assert type(x) is l.Float and dr.all(x == l.Float(11, 101, 1001))
#    x = 1 + l.Float(10, 100, 1000)
#    assert type(x) is l.Float and dr.all(x == l.Float(11, 101, 1001))
#    x = l.Float(10, 100, 1000) + (1, 2, 3)
#    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
#    x = (1, 2, 3) + l.Float(10, 100, 1000)
#    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
#    x = [1, 2, 3] + l.Float(10, 100, 1000)
#    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
#    x = l.Float(10, 100, 1000) + [1, 2, 3]
#    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
#
#    x = l.Array3f(10, 100, 1000) + 1
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
#    x = 1 + l.Array3f(10, 100, 1000)
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
#    x = l.Array3f(10, 100, 1000) + (1, 2, 3)
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
#    x = (1, 2, 3) + l.Array3f(10, 100, 1000)
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
#    x = [1, 2, 3] + l.Array3f(10, 100, 1000)
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
#    x = l.Array3f(10, 100, 1000) + [1, 2, 3]
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
#
#    x = s.Array3i(10, 100, 1000) + l.Float(1)
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
#    x = s.Array3f(10, 100, 1000) + l.Float(1)
#    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
