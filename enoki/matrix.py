import enoki as _ek
from sys import modules as _modules


def rotate(target_type, axis, angle):
    if target_type.IsQuaternion:
        s, c = sincos(angle * .5)
        quat = target_type()
        quat.imag = axis * s
        quat.real = c
        return quat
    else:
        raise Exception("Unsupported target type!")


def transpose(a):
    if not _ek.is_matrix_v(a):
        raise Exception("Unsupported target type!")

    result = type(a)()
    for i in range(a.Size):
        for j in range(a.Size):
            result[j, i] = a[i, j]
    return result


def det(m):
    if not _ek.is_matrix_v(m):
        raise Exception("Unsupported target type!")

    if m.Size == 1:
        return m[0, 0]
    elif m.Size == 2:
        return _ek.fmsub(m[0, 0], m[1, 1], m[0, 1] * m[1, 0])
    elif m.Size == 3:
        return _ek.dot(m[0], _ek.cross(m[1], m[2]))
    elif m.Size == 4:
        col0, col1, col2, col3 = m

        col1 = _ek.shuffle((2, 3, 0, 1), col1)
        col3 = _ek.shuffle((2, 3, 0, 1), col3)

        temp = _ek.shuffle((1, 0, 3, 2), col2 * col3)
        row0 = col1 * temp
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row0 = _ek.fmsub(col1, temp, row0)

        temp = _ek.shuffle((1, 0, 3, 2), col1 * col2)
        row0 = _ek.fmadd(col3, temp, row0)
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row0 = _ek.fnmadd(col3, temp, row0)

        col1 = _ek.shuffle((2, 3, 0, 1), col1)
        col2 = _ek.shuffle((2, 3, 0, 1), col2)
        temp = _ek.shuffle((1, 0, 3, 2), col1 * col3)
        row0 = _ek.fmadd(col2, temp, row0)
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row0 = _ek.fnmadd(col2, temp, row0)

        return _ek.dot(col0, row0)
    else:
        raise Exception('Unsupported array size!')


def inverse_transpose(a):
    if not _ek.is_matrix_v(a):
        raise Exception("Unsupported target type!")
    return a.inverse_transpose_()


def inverse(a):
    return transpose(inverse_transpose(a))


def diag(a):
    if _ek.is_matrix_v(a):
        result = a.Value()
        for i in range(a.Size):
            result[i] = a[i, i]
        return result
    elif _ek.is_static_array_v(a):
        name = _ek.detail.array_name('Matrix', a.Type,
                                     (a.Size, *a.Shape), a.IsScalar)
        module = _modules.get(a.__module__)
        cls = getattr(module, name)
        result = _ek.zero(cls)
        for i in range(a.Size):
            result[i, i] = a[i]
        return result
    else:
        raise Exception('Unsupported type!')


def trace(a):
    if not _ek.is_matrix_v(a):
        raise Exception('Unsupported type!')
    result = a[0, 0]
    for i in range(1, a.Size):
        result += a[i, i]
    return result


def frob(a):
    if not _ek.is_matrix_v(a):
        raise Exception('Unsupported type!')

    result = _ek.sqr(a[0])
    for i in range(1, a.Size):
        value = a[i]
        result = _ek.fmadd(value, value, result)
    return _ek.hsum(result)
