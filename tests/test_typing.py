# This file doesn't perform any useful or interesting computation. It only
# exists to be type-checked, and to ensuere that the various type inference
# rules work as expected.

# pyright: strict, reportUnusedVariable=false

import drjit as dr
from drjit.llvm import Array3f, Array3u, Array3b, Float, UInt, Bool
from drjit.scalar import Array3f as ScalarArray3f
from typing import Callable, Tuple

# TBD: matmul, matrix ops in general, complex ops
# slicing, mask assignment
# Tensor.array


def test01_element_access() -> None:
    """Type-check array creation, element access, element assignment"""
    a: Array3f = Array3f(1, 2, 3)
    b: Float = a[0]
    c: Float = a.x
    d: float = c[0]
    a[1] = a[0]
    a.y = a.x
    a[2] = 0
    #a.z = 0  # Doesn't type-check on MyPy (https://github.com/python/mypy/issues/3004)

def test02_unary_op_dunder() -> None:
    """Type-check a builtin (dunder) unary op"""
    a = Array3f(1, 2, 3)
    b: Array3f = -a
    c: Array3f = +a

def test03_unary_op_dr() -> None:
    """Type-check a dr.* unary op"""
    a = Array3f(1, 2, 3)
    b: Array3f = dr.rcp(dr.square(dr.abs(abs(a))))
    c: int = dr.square(1)
    d: float = dr.rsqrt(1)

def test04_binary_op_dunder() -> None:
    """Type-check a builtin (dunder) binary op"""
    a = Array3f(1, 2, 3)
    b = Array3u(1, 2, 3)
    c = ScalarArray3f(1, 2, 3)

    t0: Array3f = a + a
    t1: Array3f = a + a.x
    t2: Array3f = a.x + a
    t3: Array3f = a + 0.0
    t4: Array3f = 0.0 + a
    t5: Array3f = a + b
    t6: Array3f = b + a
    t7: Array3f = c + a
    t8: Array3f = a + c

def test05_binary_op_dr() -> None:
    """Type-check a dr.* binary op"""
    a: Array3f = Array3f(1, 2, 3)
    b = Array3u(1, 2, 3)
    c = ScalarArray3f(1, 2, 3)

    t0: Array3f = dr.minimum(a, a)
    t1: Array3f = dr.minimum(a, a.x)
    t2: Array3f = dr.minimum(a.x, a)
    t3: Array3f = dr.minimum(a, 0.0)
    t4: Array3f = dr.minimum(0.0, a)
    t5: Array3f = dr.minimum(a, b)
    t6: Array3f = dr.minimum(b, a)
    t7: Array3f = dr.minimum(c, a)
    t8: Array3f = dr.minimum(a, c)

def test06_ternary_fma() -> None:
    """Type-check the FMA operation"""
    a: Array3f = Array3f(1, 2, 3)
    b: Float = Float(4)
    c: float = 1.0
    d0: Array3f = dr.fma(a, a, a)
    d1: Array3f = dr.fma(a, b, c)
    d2: Array3f = dr.fma(b, c, a)
    d3: Array3f = dr.fma(c, a, b)

def test07_ternary_select() -> None:
    """Type-check the select operation"""
    a: Array3f = Array3f(1, 2, 3)
    b: Array3f = dr.select(a < 0, a, a)
    c: Array3f = dr.select(False, a, 0)
    d: Array3f = dr.select(a.x < 0, a.x, a)

def test08_reduce_zeros() -> None:
    """Type-check a horizontal reduction"""
    x1: Array3f = dr.zeros(Array3f, 100)
    y1: Float = dr.sum(x1)
    z1: Float = dr.sum(y1)

    x2: ScalarArray3f = dr.zeros(ScalarArray3f)
    y2: float = dr.sum(x2)


def test09_reduce_scalars() -> None:
    x: bool = dr.any([True, False])
    y: float = dr.sum([1, 2, 3])


def test10_reduce_comparison() -> None:
    x: Array3f = Array3f(1, 2, 3)
    y: Array3b = x == 2
    z: Array3b = ~(x == x.x)
    w: Bool = dr.all(z)
    q: Bool = dr.all(w)

def test11_full_opaque() -> None:
    x: Array3f = dr.full(Array3f, 0, (3, 10))
    y: Array3f = dr.opaque(Array3f, 0, (3, 10))


def test11_decorator() -> None:
    """"@dr.syntax and @dr.wrap preserve the signature of the input function"""

    def f1(x: int) -> int:
        return x

    @dr.syntax
    def f2(x: int) -> int:
        return x

    @dr.wrap("drjit", "drjit")
    def f3(x: int) -> int:
        return x

    a: Callable[[int], int] = f1
    b: Callable[[int], int] = f2
    c: Callable[[int], int] = f3


def test12_while_loop() -> None:
    state = (Array3f(1, 2, 3), 4)

    def cond(a: Array3f, b: int) -> Bool:
        return a.x < 10

    def body(a: Array3f, b: int) -> Tuple[Array3f, int]:
        return a + 10, b

    result : Tuple[Array3f, int] = dr.while_loop(
        state,
        cond,
        body
    )

def test13_if_stmt() -> None:
    state = (Array3f(1, 2, 3), 4)

    def true_fn(a: Array3f, b: int) -> Tuple[Array3f, Float]:
        return a - b, a.x

    def false_fn(a: Array3f, b: int) -> Tuple[Array3f, Float]:
        return a + b, a.y

    result: Tuple[Array3f, Float] = dr.if_stmt(
        state,
        Bool(True),
        true_fn,
        false_fn
    )

def test14_test_dot() -> None:
    a: int = dr.dot([1, 2], [3, 4])
    b0: Float = dr.dot(ScalarArray3f(1, 2, 3), Array3f(1, 2, 3))
    b1: Float = dr.abs_dot(ScalarArray3f(1, 2, 3), Array3f(1, 2, 3))
    c0: Float = dr.dot(Array3f(1, 2, 3), ScalarArray3f(1, 2, 3))
    c1: Float = dr.abs_dot(Array3f(1, 2, 3), ScalarArray3f(1, 2, 3))
    d0: Float = dr.dot(Array3f(1, 2, 3), Array3f(1, 2, 3))
    d1: Float = dr.abs_dot(Array3f(1, 2, 3), Array3f(1, 2, 3))
    e0: float = dr.dot(ScalarArray3f(1, 2, 3), ScalarArray3f(1, 2, 3))
    e1: float = dr.abs_dot(ScalarArray3f(1, 2, 3), ScalarArray3f(1, 2, 3))

def test15_test_norm() -> None:
    a0: float = dr.norm([1, 2])
    a1: float = dr.squared_norm([1, 2])
    b0: Float = dr.norm(Float(1, 2, 3))
    b1: Float = dr.squared_norm(Float(1, 2, 3))
    c0: Float = dr.norm(Array3f(1, 2, 3))
    c1: Float = dr.squared_norm(Array3f(1, 2, 3))

def test16_gather() -> None:
    x: Float = Float(0, 1, 2)
    y: Array3f = dr.gather(Array3f, x, index=[0, 0])


def test17_switch() -> None:
    x: UInt = UInt(1, 2, 3)
    y: int = 4

    def f1(x: UInt, y: int):
        return 3, Float(2*x+y)

    def f2(x: UInt, y: int):
        return 3, Float(2*x+y)

    a: int
    b: Float
    a, b = dr.switch(
        UInt(0, 1, 0),
        (f1, f2),
        x, y
    )
