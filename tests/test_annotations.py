# Test selection of Dr.Jit functionality using postponed evaluation of
# annotations. The following import has to be included first which is why we
# have a dedicated test file
from __future__ import annotations

import drjit as dr
import pytest
from dataclasses import dataclass
from typing import ClassVar

@pytest.test_arrays('float32,shape=(*),is_diff')
def test01_gather_pytree(t):
    x = t([1, 2, 3, 4])
    y = t([5, 6, 7, 8])
    i = dr.uint32_array_t(t)([1, 0])

    if dr.backend_v(t) == dr.JitBackend.CUDA:
        @dataclass
        class MyDataclass:
            a : dr.cuda.ad.Float
    else:
        @dataclass
        class MyDataclass:
            a : dr.llvm.ad.Float

    s = MyDataclass(x)

    r = dr.gather(MyDataclass, s, i)
    assert type(r) is MyDataclass
    assert dr.all(r.a == t([2, 1]))


@pytest.test_arrays('float32,shape=(*), is_diff')
def test02_scatter_pytree(t):
    x = dr.zeros(t, 4)
    y = dr.zeros(t, 4)

    if dr.backend_v(t) == dr.JitBackend.CUDA:
        @dataclass
        class MyDataclass:
            a : dr.cuda.ad.Float
    else:
        @dataclass
        class MyDataclass:
            a : dr.llvm.ad.Float

    s = MyDataclass(x)
    s.a = dr.zeros(t, 4)
    dr.scatter(
        s,
        MyDataclass(t(1, 2)),
        (1, 0),
        (True, False)
    )
    assert dr.all(s.a == [0, 1, 0, 0])
