import drjit as dr
import pytest
import io
import re

class Buffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None

    def write(self, value):
        if self.value is not None:
            raise Exception("A string was already set")
        self.value = value


def test01_format_basics():
    assert dr.format('Hello') == 'Hello'
    assert dr.format('Hello {}', 'world') == 'Hello world'
    assert dr.format('Hello {{}}') == 'Hello {}'
    assert dr.format('{1} {0}', 'world', 'Hello') == 'Hello world'
    assert dr.format('{} {}', 'Hello', 'world') == 'Hello world'
    assert dr.format('{hello} {world}', world='world', hello='Hello') == 'Hello world'
    assert dr.format('{} {world}', 'Hello', world='world') == 'Hello world'
    assert dr.format('{hello} {}', 'world', hello='Hello') == 'Hello world'
    assert dr.format('{hello} {}', 123, hello=0.25) == '0.25 123'

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): unmatched brace in format string (at position 3).')):
        assert dr.format('123{456')

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): missing keyword argument "hello" referenced by format string (at position 0).')):
        assert dr.format('{hello}')

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): missing positional argument 0 referenced by format string (at position 0).')):
        assert dr.format('{0}')

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): cannot switch from implicit to explicit field numbering.')):
        assert dr.format('{}{0}', 'a', 'b')

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): cannot switch from explicit to implicit field numbering.')):
        assert dr.format('{0}{}', 'a', 'b')

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): missing positional argument 1 referenced by format string (at position 0).')):
        assert dr.format('{1}', 'a')

    with pytest.raises(RuntimeError, match=re.escape('drjit.format(): missing positional argument 1 referenced by format string (at position 2).')):
        assert dr.format('{}{}', 'a')


def test02_format_scalar_pytrees():
    class Foo:
        DRJIT_STRUCT = { 'a': int, 'b': tuple, 'c': dict}
        def __init__(self):
            self.a = 5
            self.b = (float(0.25), 123)
            self.c = {
                'e' : [1, 2],
                'f': 'foo',
                'g': {}
            }

    ref = '''
    Foo[
      a=5,
      b=(
        0.25,
        123
      ),
      c={
        'e': [
          1,
          2
        ],
        'f': 'foo',
        'g': {}
      }
    ]
    '''

    import textwrap
    assert dr.format('{}', Foo()) == textwrap.dedent(ref).strip()


def test03_format_scalar_array():
    a = dr.scalar.ArrayXf(1,2,3)
    assert dr.format('{}', a) == '[1, 2, 3]'
    assert dr.format('{}', (a,)) == '(\n  [1, 2, 3]\n)'
    a = dr.arange(dr.scalar.ArrayXf, 100)
    assert dr.format('{}', a) == '[0, 1, 2, .. 94 skipped .., 97, 98, 99]'
    assert dr.format('{}', dr.scalar.Matrix2f(1,2,3,4)) == "[[1, 2],\n [3, 4]]"
    assert dr.format('{}', (dr.scalar.Matrix3f(1,2,3,4,5,6,7,8,9),)) == \
        "(\n  [[1, 2, 3],\n   [4, 5, 6],\n   [7, 8, 9]]\n)"


def test04_simple_print():
    b = Buffer()
    dr.print("a{}", 1, file=b)
    assert b.value == 'a1\n'
    b.reset()
    dr.print("a{}", 1, file=b, end='')
    assert b.value == 'a1'


@pytest.test_arrays('shape=(2, *), uint32')
def test05_evaluated_print(t):
    b = Buffer()
    dr.print(t([1, 2, 3], [4, 5, 6]), file=b, end='')
    assert b.value == '[[1, 4],\n [2, 5],\n [3, 6]]'


@pytest.test_arrays('shape=(*), uint32')
def test06_evaluated_format_big(t):
    assert dr.format("{}", dr.arange(t, 100)) == "[0, 1, 2, .. 94 skipped .., 97, 98, 99]"


@pytest.test_arrays('shape=(2, *), uint32')
def test07_evaluated_format_big_2d(t):
    assert dr.format("{}", t(dr.arange(dr.value_t(t), 100), 0)) == \
        '[[0, 0],\n [1, 0],\n [2, 0],\n .. 94 skipped ..,\n [97, 0],\n [98, 0],\n [99, 0]]'


@pytest.test_arrays('shape=(*), uint32, jit')
def test08_symbolic_print(t):
    b = Buffer()
    dr.print(t([1, 2, 3]), file=b, end='', mode='symbolic')
    assert b.value is None
    dr.eval()
    assert b.value == '[1, 2, 3]'


@pytest.test_arrays('shape=(*), uint32, jit')
def test09_symbolic_print_large(t):
    b = Buffer()
    dr.print(dr.arange(t, 1000)*2, file=b, mode='symbolic', limit=25)
    with pytest.warns(RuntimeWarning) as record:
        assert b.value is None
        dr.eval()
        r = list(map(int, b.value[1:-2].split(',')))
        assert sorted(r) == r
    assert len(record) == 1
    assert "symbolic print statement only captured 25 of 1000 available outputs" in record[0].message.args[0]


@pytest.test_arrays('shape=(*), uint32, jit')
def test10_print_from_subroutine(t):
    b1 = Buffer()
    b2 = Buffer()

    def f1(x):
        dr.print("in f1: x={x}", x=x, file=b1)

    def f2(x):
        dr.print("in f2: x={x}", x=x, file=b2)

    dr.switch(
        index=t(0, 0, 0, 1, 1, 1),
        targets=[f1, f2],
        x=t(1, 2, 3, 4, 5, 6)
    )

    dr.eval()
    assert b1.value == 'in f1: x=[1, 2, 3]\n'
    assert b2.value == 'in f2: x=[4, 5, 6]\n'


@pytest.test_arrays('shape=(*), uint32, jit')
def test11_print_from_subroutine_complex(t):
    b1 = Buffer()
    b2 = Buffer()

    def f1(x):
        i = t(0)

        def body(i):
            dr.print("in f1: tid={thread_id}, x={x}", x=x+i*10, file=b1)
            return (i+1,)

        dr.while_loop(
            (i,),
            lambda i: i<2,
            body
        )

    def f2(x):
        dr.print("in f2: tid={thread_id}, x={x}", x=x, file=b2)

    dr.switch(
        index=t(0, 0, 0, 1, 1, 1),
        targets=[f1, f2],
        x=t(1, 2, 3, 4, 5, 6)
    )

    dr.eval()
    assert b1.value == 'in f1: tid=[0, 0, 1, 1, 2, 2], x=[1, 11, 2, 12, 3, 13]\n'
    assert b2.value == 'in f2: tid=[3, 4, 5], x=[4, 5, 6]\n'
