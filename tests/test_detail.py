import drjit as dr


def test01_jit_scope():
    backend = dr.JitBackend.LLVM
    scope = dr.detail.scope(backend)
    dr.detail.new_scope(backend)
    assert dr.detail.scope(backend) > scope

    dr.detail.set_scope(backend, scope)
    assert dr.detail.scope(backend) == scope


def test02_scope_compaction():
    # Drive the scope counter to the brink of overflow. A node 'a' then receives
    # a near-max scope, and crossing the boundary must compact (renumber live
    # scopes) rather than wrap, which would schedule 'a' after its consumer.
    from drjit.llvm import UInt32
    backend = dr.JitBackend.LLVM

    dr.detail.advance_scope(backend, 0xFFFFFFFF)
    a = dr.opaque(UInt32, 5) + UInt32(1)
    dr.detail.new_scope(backend)
    b = a + UInt32(1)
    assert b[0] == 7
