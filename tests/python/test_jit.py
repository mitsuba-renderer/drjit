import drjit as dr
import pytest
import importlib


@pytest.fixture(scope="module", params=['drjit.cuda.ad', 'drjit.llvm.ad'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test01_kernel_history(m):
    for i in range(4):
        dr.eval(dr.arange(m.Float, i + 4))

    # Kernel history should be disabled by default
    assert len(dr.kernel_history()) == 0

    assert not dr.flag(dr.JitFlag.KernelHistory)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        assert dr.flag(dr.JitFlag.KernelHistory)
        for i in range(4):
            dr.eval(dr.arange(m.Float, i + 4))

        history = dr.kernel_history()
        assert len(history) == 4
        for i in range(4):
            assert history[i]['size'] == i + 4

    assert not dr.flag(dr.JitFlag.KernelHistory)

    # Kernel history should be erased after queried
    assert len(dr.kernel_history()) == 0



# TODO:
# - Check number of kernel launched when scheduling variables to make sure it create a single kernel
# - Check that number of output only contains the ones required (optimization)
# - ...
