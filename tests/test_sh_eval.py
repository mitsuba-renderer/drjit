import pytest
import drjit as dr

def test00_sh_eval():
    special = pytest.importorskip("scipy.special").special
    np = pytest.importorskip("numpy")
    from drjit.scalar import Array3f

    v = dr.normalize(Array3f(1, 2, 3))

    theta, phi = np.arccos(v.z), np.arctan2(v.y, v.x)

    r2 = []
    for l in range(10):
        for m in range(-l, l + 1):
            sph_harm_y = getattr(special, 'sph_harm_y', None)

            # Use the newer sph_harm_y (with flipped argument pairs) if available
            if sph_harm_y is not None:
                Y = sph_harm_y(l, abs(m), theta, phi)
            else:
                Y = special.sph_harm(abs(m), l, phi, theta)

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
