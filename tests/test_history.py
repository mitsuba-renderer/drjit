import drjit as dr
import pytest

@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test01_kernel_history(t):
    for i in range(4):
        dr.eval(dr.arange(t, i + 4))

    # Kernel history should be disabled by default
    assert len(dr.kernel_history()) == 0
    assert not dr.flag(dr.JitFlag.KernelHistory)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        assert dr.flag(dr.JitFlag.KernelHistory)
        for i in range(4):
            dr.eval(dr.arange(t, i + 4))

        history = dr.kernel_history()
        assert len(history) == 4
        for i in range(4):
            assert history[i]['size'] == i + 4

    assert not dr.flag(dr.JitFlag.KernelHistory)

    # Kernel history should be erased after queried
    assert len(dr.kernel_history()) == 0
