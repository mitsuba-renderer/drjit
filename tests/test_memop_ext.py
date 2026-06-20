import drjit as dr
import pytest
import sys


def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("memop_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda
    elif backend == dr.JitBackend.Metal:
        return m.metal


@pytest.test_arrays('float32,is_diff,shape=(*)')
@pytest.mark.parametrize('fn,packet', [('gather', False),
                                       ('packet_gather', True),
                                       ('packet_gather_dynamic', True)])
def test01_gather(t, fn, packet):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    index = UInt32(0, 2)
    source, source_r = dr.arange(t, 8) + 0.5, dr.arange(t, 8) + 0.5
    dr.enable_grad(source, source_r)

    if packet:
        a, b = getattr(pkg, fn)(source, index)
        r, r_ref = a + b, dr.gather(t, source_r, index * 2) + \
                          dr.gather(t, source_r, index * 2 + 1)
    else:
        r, r_ref = pkg.gather(source, index), dr.gather(t, source_r, index)

    dr.backward(dr.sum(r))
    dr.backward(dr.sum(r_ref))
    assert dr.allclose(r, r_ref)
    assert dr.allclose(dr.grad(source), dr.grad(source_r))


@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
@pytest.test_arrays('float32,is_diff,shape=(*)')
@pytest.mark.parametrize('fn,packet', [('scatter', False),
                                       ('packet_scatter', True),
                                       ('packet_scatter_dynamic', True)])
def test02_scatter(t, fn, packet):
    pkg = get_pkg(t)
    UInt32 = sys.modules[t.__module__].UInt32
    target, target_r = dr.full(t, 1, 6), dr.full(t, 1, 6)
    acc = dr.zeros(t, 6)

    if packet:
        index = UInt32(0, 1)
        vals = [t(10, 20), t(30, 40)]
        vals_r = [t(10, 20), t(30, 40)]
        dr.enable_grad(target, target_r, *vals, *vals_r)
        r = getattr(pkg, fn)(target, vals[0], vals[1], index)
        for j, v_r in enumerate(vals_r):
            dr.scatter_reduce(dr.ReduceOp.Add, acc, v_r, index * 2 + j)
    else:
        index = UInt32(0, 5)
        vals, vals_r = [t(10, 20)], [t(10, 20)]
        dr.enable_grad(target, target_r, *vals, *vals_r)
        r = pkg.scatter(target, vals[0], index)
        dr.scatter_reduce(dr.ReduceOp.Add, acc, vals_r[0], index)

    r_ref = target_r + acc
    dr.backward(dr.sum(r))
    dr.backward(dr.sum(r_ref))
    assert dr.allclose(r, r_ref)
    assert dr.allclose(dr.grad(target), dr.grad(target_r))
    for v, v_r in zip(vals, vals_r):
        assert dr.allclose(dr.grad(v), dr.grad(v_r))
