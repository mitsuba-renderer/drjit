import enoki as _ek
from sys import modules as _modules


def rotate(target_type, axis, angle):
    if target_type.IsQuaternion:
        s, c = _ek.sincos(angle * .5)
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


def inverse_transpose(m):
    if not _ek.is_matrix_v(m):
        raise Exception("Unsupported target type!")

    t = type(m)
    if m.Size == 1:
        return t(_ek.rcp(m[0, 0]))
    elif m.Size == 2:
        inv_det = _ek.rcp(_ek.fmsub(m[0, 0], m[1, 1], m[0, 1] * m[1, 0]))
        return t(
            m[1, 1] * inv_det, -m[1, 0] * inv_det,
            -m[0, 1] * inv_det, m[0, 0] * inv_det
        )
    elif m.Size == 3:
        col0, col1, col2 = m
        row0 = _ek.cross(col1, col2)
        row1 = _ek.cross(col2, col0)
        row2 = _ek.cross(col0, col1)
        inv_det = _ek.rcp(_ek.dot(col0, row0))

        return t(
            row0 * inv_det,
            row1 * inv_det,
            row2 * inv_det
        )

    elif m.Size == 4:
        col0, col1, col2, col3 = m

        col1 = _ek.shuffle((2, 3, 0, 1), col1)
        col3 = _ek.shuffle((2, 3, 0, 1), col3)

        temp = _ek.shuffle((1, 0, 3, 2), col2 * col3)
        row0 = col1 * temp
        row1 = col0 * temp
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row0 = _ek.fmsub(col1, temp, row0)
        row1 = _ek.shuffle((2, 3, 0, 1), _ek.fmsub(col0, temp, row1))

        temp = _ek.shuffle((1, 0, 3, 2), col1 * col2)
        row0 = _ek.fmadd(col3, temp, row0)
        row3 = col0 * temp
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row0 = _ek.fnmadd(col3, temp, row0)
        row3 = _ek.shuffle((2, 3, 0, 1), _ek.fmsub(col0, temp, row3))

        temp = _ek.shuffle((1, 0, 3, 2),
                           _ek.shuffle((2, 3, 0, 1), col1) * col3)
        col2 = _ek.shuffle((2, 3, 0, 1), col2)
        row0 = _ek.fmadd(col2, temp, row0)
        row2 = col0 * temp
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row0 = _ek.fnmadd(col2, temp, row0)
        row2 = _ek.shuffle((2, 3, 0, 1), _ek.fmsub(col0, temp, row2))

        temp = _ek.shuffle((1, 0, 3, 2), col0 * col1)
        row2 = _ek.fmadd(col3, temp, row2)
        row3 = _ek.fmsub(col2, temp, row3)
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row2 = _ek.fmsub(col3, temp, row2)
        row3 = _ek.fnmadd(col2, temp, row3)

        temp = _ek.shuffle((1, 0, 3, 2), col0 * col3)
        row1 = _ek.fnmadd(col2, temp, row1)
        row2 = _ek.fmadd(col1, temp, row2)
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row1 = _ek.fmadd(col2, temp, row1)
        row2 = _ek.fnmadd(col1, temp, row2)

        temp = _ek.shuffle((1, 0, 3, 2), col0 * col2)
        row1 = _ek.fmadd(col3, temp, row1)
        row3 = _ek.fnmadd(col1, temp, row3)
        temp = _ek.shuffle((2, 3, 0, 1), temp)
        row1 = _ek.fnmadd(col3, temp, row1)
        row3 = _ek.fmadd(col1, temp, row3)

        inv_det = _ek.rcp(_ek.dot(col0, row0))

        return t(
            row0 * inv_det, row1 * inv_det,
            row2 * inv_det, row3 * inv_det
        )
    else:
        raise Exception('Unsupported array size!')


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


def polar_decomp(a, it=10):
    q = type(a)(a)
    for i in range(it):
        qi = _ek.inverse_transpose(q)
        gamma = _ek.sqrt(_ek.frob(qi) / _ek.frob(q))
        s1, s2 = gamma * .5, (_ek.rcp(gamma) * .5)
        for i in range(a.Size):
            q[i] = _ek.fmadd(q[i], s1, qi[i] * s2)
    return q, transpose(q) @ a


def quat_to_matrix(q, size = 4):
    if not _ek.is_quaternion_v(q):
        raise Exception('Unsupported type!')

    name = _ek.detail.array_name('Matrix', q.Type, (size, size), q.IsScalar)
    module = _modules.get(q.__module__)
    Matrix = getattr(module, name)

    q = q * _ek.SqrtTwo

    xx = q.x * q.x; yy = q.y * q.y; zz = q.z * q.z
    xy = q.x * q.y; xz = q.x * q.z; yz = q.y * q.z
    xw = q.x * q.w; yw = q.y * q.w; zw = q.z * q.w

    if size == 4:
        return Matrix(
            1.0 - (yy + zz), xy - zw, xz + yw, 0.0,
            xy + zw, 1.0 - (xx + zz), yz - xw, 0.0,
            xz - yw, yz + xw, 1.0 - (xx + yy), 0.0,
            0.0, 0.0, 0.0, 1.0)
    else:
        return Matrix(
            1.0 - (yy + zz), xy - zw, xz + yw,
            xy + zw, 1.0 - (xx + zz), yz - xw,
            xz - yw,  yz + xw, 1.0 - (xx + yy)
        )


def matrix_to_quat(m):
    if not _ek.is_matrix_v(m):
        raise Exception('Unsupported type!')

    name = _ek.detail.array_name('Quaternion', m.Type, [4], m.IsScalar)
    module = _modules.get(m.__module__)
    Quat4f = getattr(module, name)

    o = 1.0
    t0 = o + m[0, 0] - m[1, 1] - m[2, 2]
    q0 = Quat4f(t0, m[1, 0] + m[0, 1], m[0, 2] + m[2, 0], m[2, 1] - m[1, 2])

    t1 = o - m[0, 0] + m[1, 1] - m[2, 2]
    q1 = Quat4f(m[1, 0] + m[0, 1], t1, m[2, 1] + m[1, 2], m[0, 2] - m[2, 0])

    t2 = o - m[0, 0] - m[1, 1] + m[2, 2]
    q2 = Quat4f(m[0, 2] + m[2, 0], m[2, 1] + m[1, 2], t2, m[1, 0] - m[0, 1])

    t3 = o + m[0, 0] + m[1, 1] + m[2, 2]
    q3 = Quat4f(m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], t3)

    mask0 = m[0, 0] > m[1, 1]
    t01 = _ek.select(mask0, t0, t1)
    q01 = _ek.select(mask0, q0, q1)

    mask1 = m[0, 0] < -m[1, 1]
    t23 = _ek.select(mask1, t2, t3)
    q23 = _ek.select(mask1, q2, q3)

    mask2 = m[2, 2] < 0.0
    t0123 = _ek.select(mask2, t01, t23)
    q0123 = _ek.select(mask2, q01, q23)

    return q0123 * (_ek.rsqrt(t0123) * 0.5)


def quat_to_euler(q):
    q_y_2 = _ek.sqr(q.y)

    sinr_cosp = 2 * _ek.fmadd(q.w, q.x, q.y * q.z)
    cosr_cosp = _ek.fnmadd(2, _ek.fmadd(q.x, q.x, q_y_2), 1)
    roll = _ek.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * _ek.fmsub(q.w, q.y, q.z * q.x)
    if (_ek.abs(sinp) >= 1.0):
        pitch = _ek.copysign(0.5 * _ek.Pi, sinp)
    else:
        pitch = _ek.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * _ek.fmadd(q.w, q.z, q.x * q.y)
    cosy_cosp = _ek.fnmadd(2, _ek.fmadd(q.z, q.z, q_y_2), 1)
    yaw = _ek.atan2(siny_cosp, cosy_cosp)

    name = _ek.detail.array_name('Array', q.Type, [3], q.IsScalar)
    module = _modules.get(q.__module__)
    Array3f = getattr(module, name)

    return Array3f(roll, pitch, yaw)


def transform_decompose(a, it=10):
    if not _ek.is_matrix_v(a):
        raise Exception('Unsupported type!')

    name = _ek.detail.array_name('Array', a.Type, [3], a.IsScalar)
    module = _modules.get(a.__module__)
    Array3f = getattr(module, name)

    name = _ek.detail.array_name('Matrix', a.Type, (3, 3), a.IsScalar)
    Matrix3f = getattr(module, name)

    Q, P = polar_decomp(Matrix3f(a), it)

    sign_q = det(Q)
    Q = _ek.mulsign(Q, sign_q)
    P = _ek.mulsign(P, sign_q)

    return P, matrix_to_quat(Q), Array3f(a[3][0], a[3][1], a[3][2])


def transform_compose(s, q, t):
    if not _ek.is_matrix_v(s) or not _ek.is_quaternion_v(q):
        raise Exception('Unsupported type!')

    name = _ek.detail.array_name('Matrix', q.Type, (4, 4), q.IsScalar)
    module = _modules.get(q.__module__)
    Matrix4f = getattr(module, name)

    result = Matrix4f(quat_to_matrix(q, 3) @ s)
    result[3] = [t[0], t[1], t[2], 1.0]

    return result
