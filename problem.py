import drjit as dr
from drjit.llvm import Float
dr.set_log_level(5)


#f(1)
@dr.syntax
def test28_if_stmt_struct_var_aliasing(t, variant):
    class Struct:
        v: t
        DRJIT_STRUCT = {'v': t }

    n = 8
    si = dr.zeros(Struct, n)
    v = dr.linspace(t, 0, 1, n)
    if v < 0.5:
        si.v = v
        if variant == 1:
            v += 1
    print(si.v)
    v2 = dr.linspace(t, 0, 1, n)
    assert dr.all(si.v == dr.select(v2 < 0.5, v, 0))

test28_if_stmt_struct_var_aliasing(Float, 0)
#test28_if_stmt_struct_var_aliasing(Float, 1)
