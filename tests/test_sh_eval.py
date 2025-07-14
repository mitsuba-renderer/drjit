import pytest
import drjit as dr

def test00_sh_eval():
    sph_harm_y = pytest.importorskip("scipy.special").sph_harm_y
    np = pytest.importorskip("numpy")
    from drjit.scalar import Array3f

    v = dr.normalize(Array3f(1, 2, 3))

    theta, phi = np.arccos(v.z), np.arctan2(v.y, v.x)

    r2 = []
    for l in range(10):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m > 0:
                Y = np.sqrt(2) * Y.real
            elif m < 0:
                Y = np.sqrt(2) * Y.imag
            r2.append(Y.real)


    r = dr.sh_eval(v, order=9)
    assert dr.allclose(r, r2)

    for i in range(9):
        r3 = dr.sh_eval(v, order=i)
        assert r[:len(r3)] == r3
