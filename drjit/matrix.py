import drjit as _dr
from sys import modules as _modules


def rotate(dtype, axis, angle):
    '''
    Constructs a rotation quaternion, which rotates by ``angle`` radians around
    ``axis``.

    The function requires ``axis`` to be normalized.

    Args:
        dtype (type): Desired Dr.Jit quaternion type.

        axis (drjit.ArrayBase): A 3-dimensional Dr.Jit array representing the rotation axis

        angle (float | drjit.ArrayBase): Rotation angle.

    Returns:
        drjit.ArrayBase: The rotation quaternion
    '''
    if dtype.IsQuaternion:
        s, c = _dr.sincos(angle * .5)
        quat = dtype()
        quat.imag = axis * s
        quat.real = c
        return quat
    else:
        raise Exception("Unsupported target type!")


def transpose(a, /):
    '''
    transpose(arg, /)
    Transpose the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The transposed matrix
    '''
    if not _dr.is_matrix_v(a):
        raise Exception("Unsupported target type!")

    result = type(a)()
    for i in range(a.Size):
        for j in range(a.Size):
            result[j, i] = a[i, j]
    return result


def inverse_transpose(m, /):
    '''
    inverse_transpose(arg, /)
    Returns the inverse transpose of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The inverse transposed matrix
    '''
    if not _dr.is_matrix_v(m):
        raise Exception("Unsupported target type!")

    t = type(m)
    if m.Size == 1:
        return t(_dr.rcp(m[0, 0]))
    elif m.Size == 2:
        inv_det = _dr.rcp(_dr.fma(m[0, 0], m[1, 1], -m[0, 1] * m[1, 0]))
        return t(
            m[1, 1] * inv_det, -m[1, 0] * inv_det,
            -m[0, 1] * inv_det, m[0, 0] * inv_det
        )
    elif m.Size == 3:
        col0, col1, col2 = m
        row0 = _dr.cross(col1, col2)
        row1 = _dr.cross(col2, col0)
        row2 = _dr.cross(col0, col1)
        inv_det = _dr.rcp(_dr.dot(col0, row0))

        return t(
            row0 * inv_det,
            row1 * inv_det,
            row2 * inv_det
        )

    elif m.Size == 4:
        col0, col1, col2, col3 = m

        col1 = _dr.shuffle((2, 3, 0, 1), col1)
        col3 = _dr.shuffle((2, 3, 0, 1), col3)

        temp = _dr.shuffle((1, 0, 3, 2), col2 * col3)
        row0 = col1 * temp
        row1 = col0 * temp
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row0 = _dr.fma(col1, temp, -row0)
        row1 = _dr.shuffle((2, 3, 0, 1), _dr.fma(col0, temp, -row1))

        temp = _dr.shuffle((1, 0, 3, 2), col1 * col2)
        row0 = _dr.fma(col3, temp, row0)
        row3 = col0 * temp
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row0 = _dr.fma(-col3, temp, row0)
        row3 = _dr.shuffle((2, 3, 0, 1), _dr.fma(col0, temp, -row3))

        temp = _dr.shuffle((1, 0, 3, 2),
                           _dr.shuffle((2, 3, 0, 1), col1) * col3)
        col2 = _dr.shuffle((2, 3, 0, 1), col2)
        row0 = _dr.fma(col2, temp, row0)
        row2 = col0 * temp
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row0 = _dr.fma(-col2, temp, row0)
        row2 = _dr.shuffle((2, 3, 0, 1), _dr.fma(col0, temp, -row2))

        temp = _dr.shuffle((1, 0, 3, 2), col0 * col1)
        row2 = _dr.fma(col3, temp, row2)
        row3 = _dr.fma(col2, temp, -row3)
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row2 = _dr.fma(col3, temp, -row2)
        row3 = _dr.fma(-col2, temp, row3)

        temp = _dr.shuffle((1, 0, 3, 2), col0 * col3)
        row1 = _dr.fma(-col2, temp, row1)
        row2 = _dr.fma(col1, temp, row2)
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row1 = _dr.fma(col2, temp, row1)
        row2 = _dr.fma(-col1, temp, row2)

        temp = _dr.shuffle((1, 0, 3, 2), col0 * col2)
        row1 = _dr.fma(col3, temp, row1)
        row3 = _dr.fma(-col1, temp, row3)
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row1 = _dr.fma(-col3, temp, row1)
        row3 = _dr.fma(col1, temp, row3)

        inv_det = _dr.rcp(_dr.dot(col0, row0))

        return t(
            row0 * inv_det, row1 * inv_det,
            row2 * inv_det, row3 * inv_det
        )
    else:
        raise Exception('Unsupported array size!')


def det(m, /):
    '''
    det(arg, /)
    Compute the determinant of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The determinant value of the input matrix
    '''
    if not _dr.is_matrix_v(m):
        raise Exception("Unsupported target type!")

    if m.Size == 1:
        return m[0, 0]
    elif m.Size == 2:
        return _dr.fma(m[0, 0], m[1, 1], -m[0, 1] * m[1, 0])
    elif m.Size == 3:
        return _dr.dot(m[0], _dr.cross(m[1], m[2]))
    elif m.Size == 4:
        col0, col1, col2, col3 = m

        col1 = _dr.shuffle((2, 3, 0, 1), col1)
        col3 = _dr.shuffle((2, 3, 0, 1), col3)

        temp = _dr.shuffle((1, 0, 3, 2), col2 * col3)
        row0 = col1 * temp
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row0 = _dr.fma(col1, temp, -row0)

        temp = _dr.shuffle((1, 0, 3, 2), col1 * col2)
        row0 = _dr.fma(col3, temp, row0)
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row0 = _dr.fma(-col3, temp, row0)

        col1 = _dr.shuffle((2, 3, 0, 1), col1)
        col2 = _dr.shuffle((2, 3, 0, 1), col2)
        temp = _dr.shuffle((1, 0, 3, 2), col1 * col3)
        row0 = _dr.fma(col2, temp, row0)
        temp = _dr.shuffle((2, 3, 0, 1), temp)
        row0 = _dr.fma(-col2, temp, row0)

        return _dr.dot(col0, row0)
    else:
        raise Exception('Unsupported array size!')


def inverse(arg, /):
    '''
    inverse(arg, /)
    Returns the inverse matrix of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The inverse matrix of the input matrix
    '''
    return transpose(inverse_transpose(arg))


def diag(a, /):
    '''
    diag(arg, /)
    Returns the diagonal matrix of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The diagonal matrix of the input matrix
    '''
    if _dr.is_matrix_v(a):
        result = a.Value()
        for i in range(a.Size):
            result[i] = a[i, i]
        return result
    elif _dr.is_static_array_v(a):
        name = _dr.detail.array_name('Matrix', a.Type,
                                     (a.Size, *a.Shape), a.IsScalar)
        module = _modules.get(a.__module__)
        cls = getattr(module, name)
        result = _dr.zeros(cls)
        for i in range(a.Size):
            result[i, i] = a[i]
        return result
    else:
        raise Exception('Unsupported type!')


def trace(a, /):
    '''
    trace(arg, /)
    Returns the trace of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The trace of the input matrix
    '''
    if not _dr.is_matrix_v(a):
        raise Exception('Unsupported type!')
    result = a[0, 0]
    for i in range(1, a.Size):
        result += a[i, i]
    return result


def frob(a, /):
    '''
    frob(arg, /)
    Returns the squared Frobenius norm of the provided Dr.Jit matrix.

    The squared Frobenius norm is defined as the sum of the squares of its elements:

    .. math::

        \sum_{i=1}^m \sum_{j=1}^n a_{i j}^2

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The squared Frobenius norm of the input matrix
    '''
    if not _dr.is_matrix_v(a):
        raise Exception('Unsupported type!')

    result = _dr.sqr(a[0])
    for i in range(1, a.Size):
        value = a[i]
        result = _dr.fma(value, value, result)
    return _dr.sum(result)


def quat_to_matrix(q, size=4):
    '''
    quat_to_matrix(arg, size=4)
    Converts a quaternion into its matrix representation.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit quaternion type
        size (int): Controls whether to construct a 3x3 or 4x4 matrix.

    Returns:
        drjit.ArrayBase: The Dr.Jit matrix corresponding the to input quaternion.
    '''
    if not _dr.is_quaternion_v(q):
        raise Exception('Unsupported type!')

    name = _dr.detail.array_name('Matrix', q.Type, (size, size), q.IsScalar)
    module = _modules.get(q.__module__)
    Matrix = getattr(module, name)

    q = q * _dr.sqrt_two

    xx = q.x * q.x; yy = q.y * q.y; zz = q.z * q.z
    xy = q.x * q.y; xz = q.x * q.z; yz = q.y * q.z
    xw = q.x * q.w; yw = q.y * q.w; zw = q.z * q.w

    if size == 4:
        return Matrix(
            1.0 - (yy + zz), xy - zw, xz + yw, 0.0,
            xy + zw, 1.0 - (xx + zz), yz - xw, 0.0,
            xz - yw, yz + xw, 1.0 - (xx + yy), 0.0,
            0.0, 0.0, 0.0, 1.0)
    elif size == 3:
        return Matrix(
            1.0 - (yy + zz), xy - zw, xz + yw,
            xy + zw, 1.0 - (xx + zz), yz - xw,
            xz - yw,  yz + xw, 1.0 - (xx + yy)
        )
    else:
        raise Exception('Unsupported size!')


def matrix_to_quat(m, /):
    '''
    matrix_to_quat(arg, /)
    Converts a 3x3 or 4x4 homogeneous containing
    a pure rotation into a rotation quaternion.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The Dr.Jit quaternion corresponding the to input matrix.
    '''
    if not _dr.is_matrix_v(m):
        raise Exception('Unsupported type!')

    name = _dr.detail.array_name('Quaternion', m.Type, [4], m.IsScalar)
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
    t01 = _dr.select(mask0, t0, t1)
    q01 = _dr.select(mask0, q0, q1)

    mask1 = m[0, 0] < -m[1, 1]
    t23 = _dr.select(mask1, t2, t3)
    q23 = _dr.select(mask1, q2, q3)

    mask2 = m[2, 2] < 0.0
    t0123 = _dr.select(mask2, t01, t23)
    q0123 = _dr.select(mask2, q01, q23)

    return q0123 * (_dr.rsqrt(t0123) * 0.5)


def quat_to_euler(q, /):
    '''
    quat_to_euler(arg, /)
    Converts a quaternion into its Euler angles representation.

    The order for Euler angles is XYZ.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit quaternion type

    Returns:
        drjit.ArrayBase: A 3D Dr.Jit array containing the Euler angles.
    '''
    name = _dr.detail.array_name('Array', q.Type, [3], q.IsScalar)
    module = _modules.get(q.__module__)
    Array3f = getattr(module, name)

    # Clamp the result to stay in the valid range for asin
    sinp = _dr.clamp(2 * _dr.fma(q.w, q.y, -q.z * q.x), -1.0, 1.0)
    gimbal_lock = _dr.abs(sinp) > (1.0 - 5e-8)

    # roll (x-axis rotation)
    q_y_2 = _dr.sqr(q.y)
    sinr_cosp = 2 * _dr.fma(q.w, q.x, q.y * q.z)
    cosr_cosp = _dr.fma(-2, _dr.fma(q.x, q.x, q_y_2), 1)
    roll = _dr.select(gimbal_lock, 2 * _dr.atan2(q.x, q.w), _dr.atan2(sinr_cosp, cosr_cosp))

    # pitch (y-axis rotation)
    pitch = _dr.select(gimbal_lock, _dr.copysign(0.5 * _dr.pi, sinp), _dr.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2 * _dr.fma(q.w, q.z, q.x * q.y)
    cosy_cosp = _dr.fma(-2, _dr.fma(q.z, q.z, q_y_2), 1)
    yaw = _dr.select(gimbal_lock, 0, _dr.atan2(siny_cosp, cosy_cosp))

    return Array3f(roll, pitch, yaw)


def euler_to_quat(a, /):
    '''
    euler_to_quat(arg, /)
    Converts Euler angles into a Dr.Jit quaternion.

    The order for input Euler angles must be XYZ.

    Args:
        arg (drjit.ArrayBase): A 3D Dr.Jit array type

    Returns:
        drjit.ArrayBase: A Dr.Jit quaternion representing the input Euler angles.
    '''
    name = _dr.detail.array_name('Quaternion', a.Type, [4], a.IsScalar)
    module = _modules.get(a.__module__)
    Quat4f = getattr(module, name)

    angles = a / 2.0
    sr, cr = _dr.sincos(angles.x)
    sp, cp = _dr.sincos(angles.y)
    sy, cy = _dr.sincos(angles.z)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return Quat4f(x, y, z, w)


def polar_decomp(arg, it=10):
    '''
    Returns the polar decomposition of the provided Dr.Jit matrix.

    The polar decomposition separates the matrix into a rotation followed by a
    scaling along each of its eigen vectors. This decomposition always exists
    for square matrices.

    The implementation relies on an iterative algorithm, where the number of
    iterations can be controlled by the argument ``it`` (tradeoff between
    precision and computational cost).

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

        it (int): Number of iterations to be taken by the algorithm.

    Returns:
        tuple: A tuple containing the rotation matrix and the scaling matrix resulting from the decomposition.
    '''
    q = type(arg)(arg)
    for i in range(it):
        qi = _dr.inverse_transpose(q)
        gamma = _dr.sqrt(_dr.frob(qi) / _dr.frob(q))
        s1, s2 = gamma * .5, (_dr.rcp(gamma) * .5)
        for i in range(arg.Size):
            q[i] = _dr.fma(q[i], s1, qi[i] * s2)
    return q, transpose(q) @ arg


def transform_decompose(a, it=10):
    '''
    transform_decompose(arg, it=10)
    Performs a polar decomposition of a non-perspective 4x4 homogeneous
    coordinate matrix and returns a tuple of

    1. A positive definite 3x3 matrix containing an inhomogeneous scaling operation
    2. A rotation quaternion
    3. A 3D translation vector

    This representation is helpful when animating keyframe animations.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

        it (int): Number of iterations to be taken by the polar decomposition algorithm.

    Returns:
        tuple: The tuple containing the scaling matrix, rotation quaternion and 3D translation vector.
    '''
    if not _dr.is_matrix_v(a):
        raise Exception('Unsupported type!')

    name = _dr.detail.array_name('Array', a.Type, [3], a.IsScalar)
    module = _modules.get(a.__module__)
    Array3f = getattr(module, name)

    name = _dr.detail.array_name('Matrix', a.Type, (3, 3), a.IsScalar)
    Matrix3f = getattr(module, name)

    Q, P = polar_decomp(Matrix3f(a), it)

    sign_q = det(Q)
    Q = _dr.mulsign(Q, sign_q)
    P = _dr.mulsign(P, sign_q)

    return P, matrix_to_quat(Q), Array3f(a[3][0], a[3][1], a[3][2])


def transform_compose(s, q, t, /):
    '''
    transform_compose(S, Q, T, /)
    This function composes a 4x4 homogeneous coordinate transformation from the
    given scale, rotation, and translation. It performs the reverse of
    :py:func:`transform_decompose`.

    Args:
        S (drjit.ArrayBase): A Dr.Jit matrix type representing the scaling part
        Q (drjit.ArrayBase): A Dr.Jit quaternion type representing the rotation part
        T (drjit.ArrayBase): A 3D Dr.Jit array type representing the translation part

    Returns:
        drjit.ArrayBase: The Dr.Jit matrix resulting from the composition described above.
    '''
    if not _dr.is_matrix_v(s) or not _dr.is_quaternion_v(q):
        raise Exception('Unsupported type!')

    name = _dr.detail.array_name('Matrix', q.Type, (4, 4), q.IsScalar)
    module = _modules.get(q.__module__)
    Matrix4f = getattr(module, name)

    result = Matrix4f(quat_to_matrix(q, 3) @ s)
    result[3] = [t[0], t[1], t[2], 1.0]

    return result
