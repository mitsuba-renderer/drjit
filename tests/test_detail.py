import drjit as dr


def test01_jit_scope():
    backend = dr.JitBackend.LLVM
    scope = dr.detail.scope(backend)
    dr.detail.new_scope(backend)
    assert dr.detail.scope(backend) > scope

    dr.detail.set_scope(backend, scope)
    assert dr.detail.scope(backend) == scope
