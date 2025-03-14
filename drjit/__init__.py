from . import detail

with detail.scoped_rtld_deepbind():
    try:
        from . import _drjit_ext as _drjit_ext
    except ImportError as e:
        import platform
        py_ver_pkg = detail._PYTHON_VERSION
        py_ver_cur = platform.python_version()

        err = ImportError(
            f'Could not import the Dr.Jit binary extension. It is likely that '
            f'the Python version for which Dr.Jit was compiled ({py_ver_pkg}) '
            f'is incompatible with the current interpreter ({py_ver_cur}).')

        err.__cause__ = e
        raise err

import sys as _sys
if _sys.version_info < (3, 11):
    try:
        from typing_extensions import overload, Optional, Type, Tuple, Sequence, Union, Literal, Callable
    except ImportError:
        raise RuntimeError(
            "Dr.Jit requires the 'typing_extensions' package on Python <3.11")
else:
    from typing import overload, Optional, Type, Tuple, Sequence, Union, Literal, Callable

from .ast import syntax, hint
from .interop import wrap
import warnings as _warnings


def get_cmake_dir() -> str:
    "Return the path to the Dr.Jit CMake module directory."
    import os
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "cmake", "drjit")


# -------------------------------------------------------------------
#  Predicates and comparison operations for floating point arrays
# -------------------------------------------------------------------

def isnan(arg, /):
    """
    Performs an elementwise test for *NaN* (Not a Number) values

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test.
    """
    result = arg == arg
    if isinstance(result, bool):
        return not result
    else:
        return ~result


def isinf(arg, /):
    """
    Performs an elementwise test for positive or negative infinity

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test
    """
    return abs(arg) == float('inf')


def isfinite(arg, /):
    """
    Performs an elementwise test that checks whether values are finite and not
    equal to *NaN* (Not a Number)

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test
    """
    return abs(arg) < float('inf')


def allclose(
    a: object,
    b: object,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
) -> bool:
    r'''
    Returns ``True`` if two arrays are element-wise equal within a given error
    tolerance.

    The function considers both absolute and relative error thresholds. In particular,
    **a** and **b** are considered equal if all elements satisfy

    .. math::
        |a - b| \le |b| \cdot \texttt{rtol} + \texttt{atol}.

    If not specified, the constants ``atol`` and ``rtol`` are chosen depending
    on the precision of the input arrays:

    .. list-table::
       :header-rows: 1

       * - Precision
         - ``rtol``
         - ``atol``
       * - ``float64``
         - ``1e-5``
         - ``1e-8``
       * - ``float32``
         - ``1e-3``
         - ``1e-5``
       * - ``float16``
         - ``1e-1``
         - ``1e-2``

    Note that these constants used are fairly loose and *far* larger than the
    roundoff error of the underlying floating point representation. The double
    precision parameters were chosen to match the behavior of
    ``numpy.allclose()``.

    Args:
        a (object): A Dr.Jit array or other kind of numeric sequence type.

        b (object): A Dr.Jit array or other kind of numeric sequence type.

        rtol (float): A relative error threshold chosen according to the above table.

        atol (float): An absolute error threshold according to the above table.

        equal_nan (bool): If **a** and **b** *both* contain a *NaN* (Not a Number) entry,
                          should they be considered equal? The default is ``False``.

    Returns:
        bool: The result of the comparison.
    '''

    if is_array_v(a) or is_array_v(b):
        # No derivative tracking in the following
        a, b = detach(a), detach(b)

        if is_special_v(a):
            a = array_t(a)(a)
        if is_special_v(b):
            b = array_t(b)(b)

        if is_array_v(a):
            diff = a - b
        else:
            diff = b - a

        a = type(diff)(a)
        b = type(diff)(b)

        vt = type_v(diff)

        if vt == VarType.Float16:
            rtol_ref, atol_ref = 1e-2, 1e-2
        elif vt == VarType.Float32:
            rtol_ref, atol_ref = 1e-3, 1e-5
        else:
            rtol_ref, atol_ref = 1e-5, 1e-8

        atol_c = atol_ref if atol is None else atol
        rtol_c = rtol_ref if rtol is None else rtol

        cond = abs(diff) <= abs(b) * rtol_c + atol_c

        # plus/minus infinity
        if is_float_v(a):
            cond |= a == b

        if equal_nan:
            cond |= isnan(a) & isnan(b)

        return all(cond, axis=None)

    def safe_len(x):
        try:
            return len(x)
        except TypeError:
            return 0

    def safe_getitem(x, len_x, i):
        if len_x == 0:
            return x
        elif len_x == 1:
            return x[0]
        else:
            return x[i]

    len_a, len_b = safe_len(a), safe_len(b)
    len_ab = maximum(len_a, len_b)

    if len_a != len_ab and len_a > 1 or \
       len_b != len_ab and len_b > 1:
        raise RuntimeError('drjit.allclose(): incompatible sizes '
                           '(%i and %i)!' % (len_a, len_b))
    elif len_ab == 0:
        if equal_nan and isnan(a) and isnan(b):
            return True

        rtol_c = 1e-5 if rtol is None else rtol
        atol_c = 1e-8 if atol is None else atol

        return abs(a - b) <= abs(b) * rtol_c + atol_c
    else:
        for i in range(len_ab):
            ia = safe_getitem(a, len_a, i)
            ib = safe_getitem(b, len_b, i)
            if not allclose(ia, ib, rtol, atol, equal_nan):
                return False
        return True

# -------------------------------------------------------------------
#   "Safe" functions that avoid domain errors due to rounding
# -------------------------------------------------------------------

def safe_sqrt(arg: T, /) -> T:
    '''
    Safely evaluate the square root of the provided input avoiding domain errors.

    Negative inputs produce zero-valued output. When differentiated via AD,
    this function also avoids generating infinite derivatives at ``x=0``.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input
    '''
    result = sqrt(maximum(arg, 0))
    if is_diff_v(arg) and grad_enabled(arg):
        alt = sqrt(maximum(arg, epsilon(arg)))
        result = replace_grad(result, alt)
    return result


def safe_asin(arg: T, /) -> T:
    '''
    Safe wrapper around :py:func:`drjit.asin` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain. When differentiated
    via AD, this function also avoids generating infinite derivatives at the
    boundaries of the domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arcsine approximation
    '''
    result = asin(clip(arg, -1, 1))
    if is_diff_v(arg) and grad_enabled(arg):
        alt = asin(clip(arg, -one_minus_epsilon(arg), one_minus_epsilon(arg)))
        result = replace_grad(result, alt)
    return result


def safe_acos(arg: T, /) -> T:
    '''
    Safe wrapper around :py:func:`drjit.acos` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain. When differentiated
    via AD, this function also avoids generating infinite derivatives at the
    boundaries of the domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arccosine approximation
    '''
    result = acos(clip(arg, -1, 1))
    if is_diff_v(arg) and grad_enabled(arg):
        alt = acos(clip(arg, -one_minus_epsilon(arg), one_minus_epsilon(arg)))
        result = replace_grad(result, alt)
    return result


def clip(value, min, max):
    '''
    Clip the provided input to the given interval.

    This function is equivalent to

    .. code-block::

        dr.maximum(dr.minimum(value, max), min)

    Args:
        value (int | float | drjit.ArrayBase): A Python or Dr.Jit type
        min (int | float | drjit.ArrayBase): A Python or Dr.Jit type
        max (int | float | drjit.ArrayBase): A Python or Dr.Jit type

    Returns:
        float | drjit.ArrayBase: Clipped input
    '''
    return maximum(minimum(value, max), min)


def lerp(a, b, t):
    r'''
    Linearly blend between two values.

    This function computes

    .. math::

       \mathrm{lerp}(a, b, t) = (1-t) a + t b

    In other words, it linearly blends between :math:`a` and :math:`b` based on
    the value :math:`t` that is typically on the interval :math:`[0, 1]`.

    It does so using two fused multiply-additions (:py:func:`drjit.fma`) to
    improve performance and avoid numerical errors.

    Args:
        a (int | float | drjit.ArrayBase): A Python or Dr.Jit type
        b (int | float | drjit.ArrayBase): A Python or Dr.Jit type
        t (int | float | drjit.ArrayBase): A Python or Dr.Jit type

    Returns:
        float | drjit.ArrayBase: Interpolated result
    '''

    return fma(b, t, fma(a, -t, a));


# -------------------------------------------------------------------
#     Deprecated wrappers for old Dr.Jit operations
# -------------------------------------------------------------------

def transpose(arg, /):
    _warnings.warn("transpose(x) is deprecated, please use x.T",
                   DeprecationWarning, stacklevel=2)
    return arg.T


def inverse(arg, /):
    _warnings.warn("inverse(x) is deprecated, please use rcp(x)",
                   DeprecationWarning, stacklevel=2)
    return rcp(arg)


def wrap_ad(*args, **kwargs):
    _warnings.warn("@wrap_ad is deprecated, please use @wrap",
                   DeprecationWarning, stacklevel=2)
    return wrap(*args, **kwargs)


def sqr(arg, /):
    _warnings.warn("sqr() is deprecated, please use square(arg)",
                  DeprecationWarning, stacklevel=2)
    return square(arg)


def all_nested(arg, /):
    _warnings.warn("all_nested() is deprecated, please use all(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return all(arg, axis=None)


def any_nested(arg, /):
    _warnings.warn("any_nested() is deprecated, please use any(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return any(arg, axis=None)


def sum_nested(arg, /):
    _warnings.warn("sum_nested() is deprecated, please use sum(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return sum(arg, axis=None)


def prod_nested(arg, /):
    _warnings.warn("prod_nested() is deprecated, please use prod(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return prod(arg, axis=None)


def min_nested(arg, /):
    _warnings.warn("min_nested() is deprecated, please use min(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return min(arg, axis=None)


def max_nested(arg, /):
    _warnings.warn("max_nested() is deprecated, please use max(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return max(arg, axis=None)


def none_nested(arg, /):
    _warnings.warn("none_nested() is deprecated, please use none(arg, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return none(arg, axis=None)


def clamp(value, min, max, /):
    _warnings.warn("clamp() is deprecated, please use clip(...)",
                  DeprecationWarning, stacklevel=2)
    return clip(value, min, max)

# -------------------------------------------------------------------
#  Special array operations (matrices, quaternions, complex numbers)
# -------------------------------------------------------------------

def arg(z, /):
    r'''
    Return the argument of a complex Dr.Jit array.

    The *argument* refers to the angle (in radians) between the positive real
    axis and a vector towards ``z`` in the complex plane. When the input isn't
    complex-valued, the function returns :math:`0` or :math:`\pi` depending on
    the sign of ``z``.

    Args:
        z (int | float | complex | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Argument of the complex input array
    '''
    if is_complex_v(z) or isinstance(z, complex):
        return atan2(z.imag, z.real)
    else:
        return select(z >= 0, 0, pi)

def real(arg, /):
    '''
    Return the real part of a complex or quaternion-valued input.

    When the input isn't complex- or quaternion-valued, the function returns
    the input unchanged.

    Args:
        arg (int | float | complex | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Real part of the input array
    '''
    if is_complex_v(arg) or isinstance(arg, complex):
        return arg.real
    elif is_quaternion_v(arg):
        return arg[3]
    else:
        return arg


def imag(arg, /):
    '''
    Return the imaginary part of a complex or quaternion-valued input.

    When the input isn't complex- or quaternion-valued, the function returns
    zero.

    Args:
        arg (int | float | complex | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Imaginary part of the input array
    '''
    tp = type(arg)
    if is_complex_v(tp) or issubclass(tp, complex):
        return arg.imag
    elif is_quaternion_v(tp):
        m = _sys.modules[tp.__module__]
        Array3f = replace_type_t(m.Array3f, type_v(arg))
        return Array3f(arg[0], arg[1], arg[2])
    else:
        return tp(0)


def conj(arg, /):
    '''
    Returns the conjugate of the provided complex or quaternion-valued array.
    For all other types, it returns the input unchanged.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit 3D array

    Returns:
        drjit.ArrayBase: Conjugate form of the input
    '''

    t = type(arg)

    if is_complex_v(t):
        return t(arg.real, -arg.imag)
    elif is_quaternion_v(t):
        return t(-arg.x, -arg.y, -arg.z, arg.w)
    else:
        return arg


def cross(arg0: ArrayT, arg1: ArrayT, /) -> ArrayT:
    '''
    Returns the cross-product of the two input 3D arrays

    Args:
        arg0 (drjit.ArrayBase): A Dr.Jit 3D array
        arg1 (drjit.ArrayBase): A Dr.Jit 3D array

    Returns:
        drjit.ArrayBase: Cross-product of the two input 3D arrays
    '''

    if size_v(arg0) != 3 or size_v(arg1) != 3:
        raise Exception("cross(): requires 3D input arrays!")

    return fma(arg0.yzx, arg1.zxy, -arg0.zxy * arg1.yzx)


def det(arg, /):
    '''
    det(arg, /)
    Compute the determinant of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The determinant value of the input matrix
    '''
    if not is_matrix_v(arg):
        raise Exception("Unsupported target type!")

    size = size_v(arg)

    if size == 1:
        return arg[0, 0]
    elif size == 2:
        return fma(arg[0, 0], arg[1, 1], -arg[0, 1] * arg[1, 0])
    elif size == 3:
        return dot(arg[0], cross(arg[1], arg[2]))
    elif size == 4:
        row0, row1, row2, row3 = arg

        shuffle = lambda perm, arr: type(arr)(arr[i] for i in perm)

        row1 = shuffle((2, 3, 0, 1), row1)
        row3 = shuffle((2, 3, 0, 1), row3)

        temp = shuffle((1, 0, 3, 2), row2 * row3)
        col0 = row1 * temp
        temp = shuffle((2, 3, 0, 1), temp)
        col0 = fma(row1, temp, -col0)

        temp = shuffle((1, 0, 3, 2), row1 * row2)
        col0 = fma(row3, temp, col0)
        temp = shuffle((2, 3, 0, 1), temp)
        col0 = fma(-row3, temp, col0)

        row1 = shuffle((2, 3, 0, 1), row1)
        row2 = shuffle((2, 3, 0, 1), row2)
        temp = shuffle((1, 0, 3, 2), row1 * row3)
        col0 = fma(row2, temp, col0)
        temp = shuffle((2, 3, 0, 1), temp)
        col0 = fma(-row2, temp, col0)

        return dot(row0, col0)
    else:
        raise Exception('Unsupported array size!')


def diag(arg, /):
    '''
    diag(arg, /)
    This function either returns the diagonal entries of the provided Dr.Jit
    matrix, or it constructs a new matrix from the diagonal entries.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The diagonal matrix of the input matrix
    '''
    tp = type(arg)
    if is_matrix_v(tp):
        result = value_t(arg)()
        for i in range(len(arg)):
            result[i] = arg[i, i]
        return result
    elif is_array_v(arg):
        mat_tp = matrix_t(arg)
        if mat_tp is None:
            raise Exception('drjit.diag(): unsupported type!')
        result = zeros(mat_tp, width(arg))
        for i, v in enumerate(arg):
            result[i, i] = v
        return result
    else:
        raise Exception('drjit.diag(): unsupported type!')


def identity(dtype, size=1):
    '''
    Return the identity array of the desired type and size

    This function can create identity instances of various types. In
    particular, ``dtype`` can be:

    - A Dr.Jit matrix type (like :py:class:`drjit.cuda.Matrix4f`).

    - A Dr.Jit complex type (like :py:class:`drjit.cuda.Quaternion4f`).

    - Any other Dr.Jit array type. In this case this function is equivalent to ``full(dtype, 1, size)``

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``size``
      parameter is ignored in this case.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`custom data structure <custom-struct>`.

        size (int): Size of the desired array | matrix

    Returns:
        object: The identity array of type ``dtype`` of size ``size``
    '''
    result = zeros(dtype, size)
    result += dtype(1)
    return result


def trace(arg, /):
    '''
    trace(arg, /)
    Returns the trace of the provided Dr.Jit matrix.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.value_t(arg): The trace of the input matrix
    '''
    if is_matrix_v(arg):
        accum = arg[0, 0]
        accum = type(accum)(accum)
        for i in range(1, len(arg)):
            accum += arg[i, i]
        return accum
    else:
        raise Exception('drjit.trace(): unsupported input type!')


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
    if not is_quaternion_v(dtype):
        raise Exception("drjit.rotate(): unsupported input type!")

    s, c = sincos(angle * .5)
    q = dtype(c, *(axis * s))
    return q


def frob(a, /):
    r'''
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
    if not is_matrix_v(a):
        raise Exception('frob() : unsupported type!')

    result = square(a[0])
    for i in range(1, size_v(a)):
        value = a[i]
        result = fma(value, value, result)
    return sum(result)


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
    if not is_matrix_v(arg):
        raise Exception('drjit.polar_decomp(): unsupported input type!')

    def func(q,i):
        qi = rcp(q.T)
        gamma = sqrt(frob(qi) / frob(q))
        s1, s2 = gamma * .5, (rcp(gamma) * .5)
        for j in range(size_v(arg)):
            q[j] = fma(q[j], s1, qi[j] * s2)
        i += 1
        return q, i

    tp = type(arg)
    m = _sys.modules[tp.__module__]

    q = type(arg)(arg)
    i = m.UInt32(0)
    q0, i0 = while_loop(
        state=(q, i),
        cond=lambda q, i: i < it,
        body=func
    )
    return q0, q0.T @ arg


def matrix_to_quat(mtx, /):
    '''
    matrix_to_quat(arg, /)
    Converts a 3x3 or 4x4 homogeneous containing
    a pure rotation into a rotation quaternion.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit matrix type

    Returns:
        drjit.ArrayBase: The Dr.Jit quaternion corresponding the to input matrix.
    '''
    if not is_matrix_v(mtx):
        raise Exception('drjit.matrix_to_quat(): unsupported input type!')

    s = mtx.shape[:2]
    if s != (3,3) and s != (4, 4):
        raise Exception('drjit.matrix_to_quat(): invalid input shape!')

    m = _sys.modules[mtx.__module__]
    Q = replace_type_t(m.Quaternion4f, type_v(mtx))

    o = 1.0
    t0 = o + mtx[0, 0] - mtx[1, 1] - mtx[2, 2]
    q0 = Q(t0, mtx[1, 0] + mtx[0, 1], mtx[0, 2] + mtx[2, 0], mtx[2, 1] - mtx[1, 2])

    t1 = o - mtx[0, 0] + mtx[1, 1] - mtx[2, 2]
    q1 = Q(mtx[1, 0] + mtx[0, 1], t1, mtx[2, 1] + mtx[1, 2], mtx[0, 2] - mtx[2, 0])

    t2 = o - mtx[0, 0] - mtx[1, 1] + mtx[2, 2]
    q2 = Q(mtx[0, 2] + mtx[2, 0], mtx[2, 1] + mtx[1, 2], t2, mtx[1, 0] - mtx[0, 1])

    t3 = o + mtx[0, 0] + mtx[1, 1] + mtx[2, 2]
    q3 = Q(mtx[2, 1] - mtx[1, 2], mtx[0, 2] - mtx[2, 0], mtx[1, 0] - mtx[0, 1], t3)

    mask0 = mtx[0, 0] > mtx[1, 1]
    t01 = select(mask0, t0, t1)
    q01 = select(mask0, q0, q1)

    mask1 = mtx[0, 0] < -mtx[1, 1]
    t23 = select(mask1, t2, t3)
    q23 = select(mask1, q2, q3)

    mask2 = mtx[2, 2] < 0.0
    t0123 = select(mask2, t01, t23)
    q0123 = select(mask2, q01, q23)

    return Q(q0123 * (rsqrt(t0123) * 0.5))


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
    if not is_quaternion_v(q):
        raise Exception('drjit.quat_to_matrix(): unsupported input type!')

    if size != 3 and size != 4:
        raise Exception('drjit.quat_to_matrix(): Unsupported input size!')

    m = _sys.modules[q.__module__]
    Matrix3f = replace_type_t(m.Matrix3f, type_v(q))
    Matrix4f = replace_type_t(m.Matrix4f, type_v(q))

    q = q * sqrt_two

    xx = q.x * q.x; yy = q.y * q.y; zz = q.z * q.z
    xy = q.x * q.y; xz = q.x * q.z; yz = q.y * q.z
    xw = q.x * q.w; yw = q.y * q.w; zw = q.z * q.w

    if size == 4:
        return Matrix4f(
            1.0 - (yy + zz), xy - zw, xz + yw, 0.0,
            xy + zw, 1.0 - (xx + zz), yz - xw, 0.0,
            xz - yw, yz + xw, 1.0 - (xx + yy), 0.0,
            0.0, 0.0, 0.0, 1.0)
    elif size == 3:
        return Matrix3f(
            1.0 - (yy + zz), xy - zw, xz + yw,
            xy + zw, 1.0 - (xx + zz), yz - xw,
            xz - yw,  yz + xw, 1.0 - (xx + yy)
        )


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

    if not is_quaternion_v(q):
        raise Exception('drjit.quat_to_euler(): unsupported input type!')

    m = _sys.modules[q.__module__]
    Array3f = replace_type_t(m.Array3f, type_v(q))

    # Clamp the result to stay in the valid range for asin
    sinp = clip(2 * fma(q.w, q.y, -q.z * q.x), -1.0, 1.0)
    gimbal_lock = abs(sinp) > (1.0 - 5e-8)

    # roll (x-axis rotation)
    q_y_2 = square(q.y)
    sinr_cosp = 2 * fma(q.w, q.x, q.y * q.z)
    cosr_cosp = fma(-2, fma(q.x, q.x, q_y_2), 1)
    roll = select(gimbal_lock, 2 * atan2(q.x, q.w), atan2(sinr_cosp, cosr_cosp))

    # pitch (y-axis rotation)
    pitch = select(gimbal_lock, copysign(0.5 * pi, sinp), asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2 * fma(q.w, q.z, q.x * q.y)
    cosy_cosp = fma(-2, fma(q.z, q.z, q_y_2), 1)
    yaw = select(gimbal_lock, 0, atan2(siny_cosp, cosy_cosp))

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
    if not is_array_v(a):
        raise Exception('drjit.euler_to_quat(): unsupported input type!')

    if len(a) != 3:
        raise Exception('drjit.euler_to_quat(): input has invalid shape!')

    m = _sys.modules[a.__module__]
    Quaternion4f = replace_type_t(m.Quaternion4f, type_v(a))

    angles = a / 2.0
    sr, cr = sincos(angles.x)
    sp, cp = sincos(angles.y)
    sy, cy = sincos(angles.z)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return Quaternion4f(x, y, z, w)


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
    if not is_matrix_v(a):
        raise Exception('drjit.transform_decompose(): unsupported input type!')

    if a.shape[:2] != (4,4):
        raise Exception('drjit.transform_decompose(): invalid input shape!')

    m = _sys.modules[a.__module__]
    Matrix3f = replace_type_t(m.Matrix3f, type_v(a))
    Array3f  = replace_type_t(m.Array3f, type_v(a))

    m33 = Matrix3f(
        a[0][0], a[0][1], a[0][2],
        a[1][0], a[1][1], a[1][2],
        a[2][0], a[2][1], a[2][2]
    )

    Q, P = polar_decomp(m33, it)

    sign_q = det(Q)
    Q = mulsign(Q, sign_q)
    P = mulsign(P, sign_q)

    return P, matrix_to_quat(Q), Array3f(a[0][3], a[1][3], a[2][3])


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
    if not is_matrix_v(s) or not is_quaternion_v(q):
        raise Exception('drjit.transform_compose(): unsupported input type!')

    if s.shape[:2] != (3,3):
        raise Exception('drjit.transform_decompose(): scale has invalid shape!')

    if len(t) != 3:
        raise Exception('drjit.transform_decompose(): translation has invalid shape!')

    m = _sys.modules[s.__module__]
    Matrix3f = replace_type_t(m.Matrix3f, type_v(s))
    Matrix4f = replace_type_t(m.Matrix4f, type_v(s))

    m33 = Matrix3f(quat_to_matrix(q, 3) @ s)

    m44 = Matrix4f(
        m33[0][0], m33[0][1], m33[0][2], t[0],
        m33[1][0], m33[1][1], m33[1][2], t[1],
        m33[2][0], m33[2][1], m33[2][2], t[2],
        0        , 0        , 0        , 1.0
    )

    return m44

def zeros_like(arg: T, /) -> T:
    """
    Return an array of zeros with the same shape and type as a given array.
    """
    return zeros(type(arg), shape(arg))

def ones_like(arg: T, /) -> T:
    """
    Return an array of ones with the same shape and type as a given array.
    """
    return ones(type(arg), shape(arg))

def empty_like(arg: T, /) -> T:
    """
    Return an empty array with the same shape and type as a given array.
    """
    return empty(type(arg), shape(arg))

# -------------------------------------------------------------------
#                      Mathematical constants
# -------------------------------------------------------------------

e                     = 2.71828182845904523536  # noqa
log_two               = 0.69314718055994530942  # noqa
inv_log_two           = 1.44269504088896340736  # noqa

pi                    = 3.14159265358979323846  # noqa
inv_pi                = 0.31830988618379067154  # noqa
sqrt_pi               = 1.77245385090551602793  # noqa
inv_sqrt_pi           = 0.56418958354775628695  # noqa

two_pi                = 6.28318530717958647692  # noqa
inv_two_pi            = 0.15915494309189533577  # noqa
sqrt_two_pi           = 2.50662827463100050242  # noqa
inv_sqrt_two_pi       = 0.39894228040143267794  # noqa

four_pi               = 12.5663706143591729539  # noqa
inv_four_pi           = 0.07957747154594766788  # noqa
sqrt_four_pi          = 3.54490770181103205460  # noqa
inv_sqrt_four_pi      = 0.28209479177387814347  # noqa

sqrt_two              = 1.41421356237309504880  # noqa
inv_sqrt_two          = 0.70710678118654752440  # noqa

inf                   = float('inf')  # noqa
nan                   = float('nan')  # noqa

def epsilon(arg, /):
    '''
    Returns the machine epsilon.

    The machine epsilon gives an upper bound on the relative approximation
    error due to rounding in floating point arithmetic.

    Args:
        arg (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The machine epsilon.
    '''
    vt = type_v(arg)
    if vt == VarType.Float64:
        return float.fromhex('0x1p-53')
    elif vt == VarType.Float32:
        return float.fromhex('0x1p-24')
    elif vt == VarType.Float16:
        return float.fromhex('0x1p-11')
    else:
        raise TypeError("epsilon(): input is not a Dr.Jit array or array type!")


def one_minus_epsilon(arg, /):
    '''
    Returns one minus the machine epsilon value.

    Args:
        arg (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: One minus the machine epsilon.
    '''
    vt = type_v(arg)
    if vt == VarType.Float64:
        return float.fromhex('0x1.fffffffffffffp-1')
    elif vt == VarType.Float32:
        return float.fromhex('0x1.fffffep-1')
    elif vt == VarType.Float16:
        return float.fromhex('0x1.ffcp-1')
    else:
        raise TypeError("one_minus_epsilon(): input is not a Dr.Jit array or array type!")


def recip_overflow(arg, /):
    '''
    Returns the reciprocal overflow threshold value.

    Any numbers equal to this threshold or a smaller value will overflow to
    infinity when reciprocated.

    Args:
        arg (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The reciprocal overflow threshold value.
    '''
    vt = type_v(arg)
    if vt == VarType.Float64:
        return float.fromhex('0x1p-1024')
    elif vt == VarType.Float32:
        return float.fromhex('0x1p-128')
    elif vt == VarType.Float16:
        return float.fromhex('0x1p-16')
    else:
        raise TypeError("recip_overflow(): input is not a Dr.Jit array or array type!")


def smallest(arg, /):
    '''
    Returns the smallest representable normalized floating point value.

    Args:
        arg (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The smallest representable normalized floating point value.
    '''
    vt = type_v(arg)
    if vt == VarType.Float64:
        return float.fromhex('0x1p-1022')
    elif vt == VarType.Float32:
        return float.fromhex('0x1p-126')
    elif vt == VarType.Float16:
        return float.fromhex('0x1p-14')
    else:
        raise TypeError("smallest(): input is not a Dr.Jit array or array type!")

def largest(arg, /):
    '''
    Returns the largest representable finite floating point value for `t`.

    Args:
        arg (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The largest representable finite floating point value.
    '''
    vt = type_v(arg)
    if vt == VarType.Float64:
        return float.fromhex('0x1.fffffffffffffp+1023')
    elif vt == VarType.Float32:
        return float.fromhex('0x1.fffffep+127')
    elif vt == VarType.Float16:
        return float.fromhex('0x1.ffcp+15')
    else:
        raise TypeError("largest(): input is not a Dr.Jit array or array type!")


# -------------------------------------------------------------------
#                        Enabling/disabling AD
# -------------------------------------------------------------------


def suspend_grad(*args, when=True):
    """
    Python context manager to temporarily disable gradient tracking globally,
    or for a specific set of variables.

    This context manager can be used as follows to completely disable all
    gradient tracking. Newly created variables will be detached from Dr.Jit's
    AD graph.

    .. code-block:: python

       with dr.suspend_grad():
           # .. code coes here ..

    You may also specify any number of Dr.Jit arrays, tensors, or :ref:`PyTrees
    <pytrees>`. In this case, the context manager behaves differently by
    disabling gradient tracking more selectively for the specified variables.

    .. code-block:: python

       with dr.suspend_grad(x):
           z = x + y  # 'z' will not track any gradients arising from 'x'

    The :py:func:`suspend_grad` and :py:func:`resume_grad` context manager can
    be arbitrarily nested and suitably update the set of tracked variables.

    A note about the interaction with :py:func:`drjit.enable_grad`: it is legal
    to register further AD variables within a scope that disables gradient
    tracking for specific variables.

    .. code-block:: python

       with dr.suspend_grad(x):
           y = Float(1)
           dr.enable_grad(y)

           # The following condition holds
           assert not dr.grad_enabled(x) and \
                  dr.grad_enabled(y)

    In contrast, a :py:func:`suspend_grad` environment without arguments that
    completely disables AD does *not* allow further variables to be registered:

    .. code-block:: python

       with dr.suspend_grad():
           y = Float(1)
           dr.enable_grad(y) # ignored

           # The following condition holds
           assert not dr.grad_enabled(x) and \
                  not dr.grad_enabled(y)

    Args:
        *args (tuple): Arbitrary list of Dr.Jit arrays, tuples, or :ref:`PyTrees
          <pytrees>`. Elements of data structures that could not possibly be
          attached to the AD graph (e.g., Python scalars) are ignored.

        when (bool): Optional keyword argument that can be specified to turn the
          context manager into a no-op via ``when=False``. The default value is
          ``when=True``.
    """
    if not when:
        return detail.NullContextManager()

    array_indices = detail.collect_indices(args)

    if len(args) > 0 and len(array_indices) == 0:
        array_indices.append(0)

    return detail.ADContextManager(detail.ADScope.Suspend, array_indices)


def resume_grad(*args, when=True):
    """
    Python context manager to temporarily resume gradient tracking globally,
    or for a specific set of variables.

    This context manager can be used as follows to fully re-enable all
    gradient tracking following a previous call to
    :py:func:`drjit.suspend_grad()`. Newly created variables will then again be
    attached to Dr.Jit's AD graph.

    .. code-block:: python

       with dr.suspend_grad():
           # ..

           with dr.resume_grad():
               # In this scope, the effect of the outer context
               # manager is effectively disabled

    You may also specify any number of Dr.Jit arrays, tensors, or :ref:`PyTrees
    <pytrees>`. In this case, the context manager behaves differently by
    enabling gradient tracking more selectively for the specified variables.

    .. code-block::

       with dr.suspend_grad():
           with dr.resume_grad(x):
               z = x + y  # 'z' will only track gradients arising from 'x'

    The :py:func:`suspend_grad` and :py:func:`resume_grad` context manager can
    be arbitrarily nested and suitably update the set of tracked variables.

    Args:
        *args (tuple): Arbitrary list of Dr.Jit arrays, tuples, or :ref:`PyTrees
          <pytrees>`. Elements of data structures that could not possibly be
          attached to the AD graph (e.g., Python scalars) are ignored.

        when (bool): Optional keyword argument that can be specified to turn the
          context manager into a no-op via ``when=False``. The default value is
          ``when=True``.
    """
    if not when:
        return detail.NullContextManager()

    array_indices = []
    array_indices = detail.collect_indices(args)

    if len(args) > 0 and len(array_indices) == 0:
        array_indices.append(0)

    return detail.ADContextManager(detail.ADScope.Resume, array_indices)


def isolate_grad(when=True):
    """
    Python context manager to isolate and partition AD traversals into multiple
    distinct phases.

    Consider a sequence of steps being differentiated in reverse mode, like so:

    .. code-block:: python

       x = ..
       dr.enable_grad(x)

       y = f(x)
       z = g(y)
       dr.backward(z)

    The :py:func:`drjit.backward` call would automatically traverse the AD
    graph nodes created during the execution of the function ``f()`` and
    ``g()``.

    However, sometimes this is undesirable and more control is needed. For
    example, Dr.Jit may be in an execution context (a symbolic loop or call)
    that temporarily disallows differentiation of the ``f()`` part. The
    :py:func:`drjit.isolate_grad` context manager addresses this need:

    .. code-block::

       dr.enable_grad(x)
       y = f(x)

       with dr.isolate_grad():
           z = g(y)
           dr.backward(z)

    Any reverse-mode AD traversal of an edge that crosses the isolation
    boundary is postponed until leaving the scope. This is mathematically
    equivalent but produces two smaller separate AD graph traversals.

    Dr.Jit operations like symbolic loops and calls internally create such an
    isolation boundary, hence it is rare that you would need to do so yourself.
    :py:func:`isolate_grad` is not useful for forward mode AD.

    Args:
        when (bool): Optional keyword argument that can be specified to turn the
          context manager into a no-op via ``when=False``. The default value is
          ``when=True``.
    """
    if not when:
        return detail.NullContextManager()

    return detail.ADContextManager(detail.ADScope.Isolate, [])


# -------------------------------------------------------------------
#      Miscellaneous
# -------------------------------------------------------------------

def copy(arg: T, /) -> T:
    """
    Create a deep copy of a PyTree

    This function recursively traverses PyTrees and replaces Dr.Jit arrays with
    copies created via the ordinary copy constructor. It also rebuilds tuples,
    lists, dictionaries, and other :ref:`custom data strutures <custom_types_py>`.
    """

    return detail.copy(arg)


def sign(arg, /):
    r'''
    sign(arg, /)
    Return the element-wise sign of the provided array.

    The function returns

    .. math::

       \mathrm{sign}(\texttt{arg}) = \begin{cases}
           1&\texttt{arg}>=0,\\
           -1&\mathrm{otherwise}.
       \end{cases}

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | int | drjit.ArrayBase: Sign of the input array
    '''
    t = type(arg)
    return select(arg >= 0, t(1), t(-1))


def copysign(arg0, arg1, /):
    '''
    Copy the sign of ``arg1`` to ``arg0`` element-wise.

    Args:
        arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to change the sign of
        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to copy the sign from

    Returns:
        float | int | drjit.ArrayBase: The values of ``arg0`` with the sign of ``arg1``
    '''
    arg0_a = abs(arg0)
    return select(arg1 >= 0, arg0_a, -arg0_a)


def mulsign(arg0, arg1, /):
    '''
    Multiply ``arg0`` by the sign of ``arg1`` element-wise.

    This function is equivalent to

    .. code-block::

        a * dr.sign(b)

    Args:
        arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to multiply the sign of
        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to take the sign from

    Returns:
        float | int | drjit.ArrayBase: The values of ``arg0`` multiplied with the sign of ``arg1``
    '''
    return select(arg1 >= 0, arg0, -arg0)


def hypot(a, b, /):
    '''
    Computes :math:`\\sqrt{a^2+b^2}` while avoiding overflow and underflow.

    Args:
        arg (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Hypotenuse value
    '''
    a, b = abs(a), abs(b)
    maxval = maximum(a, b)
    minval = minimum(a, b)
    ratio = minval / maxval

    return select(
        (a < inf) & (b < inf) & (ratio < inf),
        maxval * sqrt(fma(ratio, ratio, 1)),
        a + b
    )

def log2i(arg: T, /) -> T:
    '''
    Return the floor of the base-2 logarithm.

    This function evaluates the component-wise floor of the base-2 logarithm of
    the input scalar, array, or tensor. This function assumes that ``arg`` is
    either an arbitrary Dr.Jit integer array or a 32 bit-sized scalar integer
    value.

    The operation overflows when ``arg`` is zero.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of leading zero bits in the input array
    '''

    sz = itemsize_v(arg) if is_array_v(arg) else 4
    return (sz * 8 - 1) - lzcnt(arg)


def rad2deg(arg: T, /) -> T:
    '''
    Convert angles from radians to degrees.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: The equivalent angle in degrees.
    '''
    return arg * (180.0 / pi)


def deg2rad(arg: T, /) -> T:
    '''
    Convert angles from degrees to radians.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: The equivalent angle in radians.
    '''
    return arg * (pi / 180.0)


def normalize(arg: T, /) -> T:
    '''
    Normalize the input vector so that it has unit length and return the
    result.

    This operation is equivalent to

    .. code-block:: python

       arg * dr.rsqrt(dr.squared_norm(arg))

    Args:
        arg (drjit.ArrayBase): A Dr.Jit array type

    Returns:
        drjit.ArrayBase: Unit-norm version of the input
    '''

    return arg * rsqrt(squared_norm(arg))


def hypot(a, b):
    '''
    Computes :math:`\\sqrt{x^2+y^2}` while avoiding overflow and underflow.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        float | drjit.ArrayBase: The computed hypotenuse.
    '''

    a, b = abs(a), abs(b)
    maxval = maximum(a, b)
    minval = minimum(a, b)
    ratio = minval / maxval

    return select(
        (a < inf) & (b < inf) & (ratio < inf),
        maxval * sqrt(fma(ratio, ratio, 1)),
        a + b
    )


def reverse(value, axis: int = 0):
    '''
    Reverses the given Dr.Jit array or Python sequence along the
    specified axis.

    Args:
        value (ArrayBase|Sequence): Dr.Jit array or Python sequence type

        axis (int): Axis along which the reversal should be performed. Only
          ``axis==0`` is supported for now.

    Returns:
        object: An output of the same type as `value` containing a copy of the
        reversed array.
    '''
    tp = type(value)
    n = len(value)

    if axis != 0:
        raise Exception("reverse(): only the axis=0 case is implemented so far!")

    if is_dynamic_v(tp) and depth_v(tp) == 1:
        return gather(tp, value, n - 1 - arange(uint32_array_t(tp), n))
    else:
        result = []
        for i in range(n):
            result.append(value[n-1-i])
        if not isinstance(result, tp):
            result = tp(result)
        return result

def sh_eval(d: ArrayBase, order: int) -> list:
    """
    Evalute real spherical harmonics basis function up to a specified order.

    The input ``d`` must be a normalized 3D Cartesian coordinate vector. The
    function returns a list containing all spherical harmonic basis functions
    evaluated with respect to ``d`` up to the desired order, for a total of
    ``(order+1)**2`` output values.

    The implementation relies on efficient pre-generated branch-free code with
    aggressive constant folding and common subexpression elimination. It admits
    scalar and Jit-compiled input arrays. Evaluation routines are included for
    orders ``0`` to ``10``. Requesting higher orders triggers a runtime
    exception.

    This automatically generated code is based on the paper `Efficient
    Spherical Harmonic Evaluation <http://jcgt.org/published/0002/02/06/>`__,
    *Journal of Computer Graphics Techniques (JCGT)*, vol. 2, no. 2, 84-90,
    2013 by `Peter-Pike Sloan <http://www.ppsloan.org/publications/>`__.

    The SciPy equivalent of this function is given by

    .. code-block:: python

       def sh_eval(d, order: int):
           from scipy.special import sph_harm
           theta, phi = np.arccos(d.z), np.arctan2(d.y, d.x)
           r = []
           for l in range(order + 1):
               for m in range(-l, l + 1):
                   Y = sph_harm(abs(m), l, phi, theta)
                   if m > 0:
                       Y = np.sqrt(2) * Y.real
                   elif m < 0:
                       Y = np.sqrt(2) * Y.imag
                   r.append(Y.real)
           return r

    The Mathematica equivalent of a specific entry is given by:

    .. code-block:: wolfram-language

        SphericalHarmonicQ[l_, m_, d_] := Block[{θ, ϕ},
          θ = ArcCos[d[[3]]];
          ϕ = ArcTan[d[[1]], d[[2]]];
          Piecewise[{
            {SphericalHarmonicY[l, m, θ, ϕ], m == 0},
            {Sqrt[2] * Re[SphericalHarmonicY[l,  m, θ, ϕ]], m > 0},
            {Sqrt[2] * Im[SphericalHarmonicY[l, -m, θ, ϕ]], m < 0}
          }]
        ]
    """

    if order < 0 or order > 9:
        raise RuntimeError("sh_eval(): order must be in [0, 9]");
    r = [None]*(order+1)*(order + 1)
    from . import _sh_eval as _sh_eval
    getattr(_sh_eval, f'sh_eval_{order}')(d, r)
    return r


def meshgrid(*args, indexing='xy') -> tuple: # <- proper type signature in stubs
    '''
    Return flattened N-D coordinate arrays from a sequence of 1D coordinate vectors.

    This function constructs flattened coordinate arrays that are convenient
    for evaluating and plotting functions on a regular grid. An example is
    shown below:

    .. code-block::

        import drjit as dr

        x, y = dr.meshgrid(
            dr.arange(dr.llvm.UInt, 4),
            dr.arange(dr.llvm.UInt, 4)
        )

        # x = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        # y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    This function carefully reproduces the behavior of ``numpy.meshgrid``
    except for one major difference: the output coordinates are returned in
    flattened/raveled form. Like the NumPy version, the ``indexing=='xy'`` case
    internally reorders the first two elements of ``*args``.

    Args:
        *args: A sequence of 1D coordinate arrays

        indexing (str): Specifies the indexing convention. Must be either set
        to ``'xy'`` (the default) or ``'ij'``.

    Returns:
        tuple: A tuple of flattened coordinate arrays (one per input)
    '''

    if indexing != "ij" and indexing != "xy":
        raise Exception("meshgrid(): 'indexing' argument must equal"
                        " 'ij' or 'xy'!")

    if len(args) == 0:
        return ()
    elif len(args) == 1:
        return args[0]

    t = type(args[0])
    for v in args:
        if not is_array_v(v) or depth_v(v) != 1 or type(v) is not t:
            raise Exception("meshgrid(): consistent 1D dynamic arrays expected!")

    size = prod((len(v) for v in args))
    index = arange(uint32_array_t(t), size)

    result = []

    # This seems non-symmetric but is necessary to be consistent with NumPy
    if indexing == "xy":
        args = (args[1], args[0], *args[2:])

    for v in args:
        size //= len(v)
        index_v = index // size
        index = fma(-index_v, size, index)
        result.append(gather(t, v, index_v))

    if indexing == "xy":
        result[0], result[1] = result[1], result[0]

    return tuple(result)


def _compute_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    """Turn a shape tuple into a C-style strides tuple"""
    val, ndim = 1, len(shape)
    strides = [0] * ndim
    for i in reversed(range(ndim)):
        strides[i] = val
        val *= shape[i]
    return tuple(strides)


def concat(arr: Sequence[ArrayT], /, axis: Optional[int] = 0) -> ArrayT:
    """
    Concatenate a sequence of arrays or tensors along a given axis.

    The inputs must all be of the same type, and they must have the same shape
    except for the axis being concatenated. Negative ``axis`` values count
    backwards from the last dimension.

    When ``axis=None``, the function ravels the input arrays or tensors prior
    to concatenating them.
    """

    if is_array_v(arr):
        raise TypeError("Input should be Python sequence of arrays.")

    if len(arr) == 0:
        raise RuntimeError("At least one input array/tensor is required!")
    elif len(arr) == 1:
        return arr[0]

    if axis is None:
        arr = tuple(ravel(v) for v in arr)
        axis = 0

    ref = arr[0]
    ref_tp = type(ref)
    if not is_jit_v(ref_tp):
        raise TypeError(f"The input arrays must be JIT-compiled Dr.Jit types (encountered {ref_tp})!")

    ref_shape = ref.shape
    ref_ndim = len(ref_shape)

    # Examine the 'axis' parameter
    if axis < 0:
        axis = len(ref_shape) + axis
    if axis >= len(ref_shape):
        raise RuntimeError(f"The value axis={axis} is out of bounds!")

    # Compute output shape and check compatibility
    axis_size = 0
    for i, arg in enumerate(arr):
        arg_tp = type(arg)
        if arg_tp is not ref_tp:
            raise TypeError(f"The input arrays must all have the same type (encountered {ref_tp} and {arg_tp})!")
        if not is_jit_v(arg):
            raise TypeError(f"The input arrays must be JIT-compiled Dr.Jit types (encountered {arg_tp})!")
        arg_shape = arg.shape
        arg_ndim = len(arg_shape)

        if arg_ndim != ref_ndim:
            raise TypeError(f"The input arrays must all have the same dimension (encountered {ref_ndim} vs {arg_ndim})!")

        for j in range(ref_ndim):
            aj, rj = arg_shape[j], ref_shape[j]
            if j == axis:
                axis_size += aj
            elif aj != rj:
                raise TypeError(f"Input array {i} has an incompatible size on axis {j} (encountered {rj} vs {aj})!")

    out_shape = list(ref_shape)
    out_shape[axis] = axis_size
    out_strides = _compute_strides(out_shape)

    result = empty(ref_tp, out_shape)
    result_array = result.array
    Index = uint32_array_t(type(result_array))

    axis_size = 0
    for i, arg in enumerate(arr):
        arg_shape = arg.shape
        arg_strides = _compute_strides(arg_shape)
        arg_size = prod(arg_shape)
        arg_array = arg.array
        index_in = arange(Index, arg_size)
        index_out = zeros(Index, arg_size)

        for j in range(ref_ndim):
            pos_in = index_in // arg_strides[j]
            pos_out = pos_in

            if j == axis:
                pos_out = pos_out + axis_size
                axis_size += arg_shape[j]

            index_in -= pos_in * arg_strides[j]
            index_out += pos_out * out_strides[j]

            if j == axis:
                index_out += index_in
                break

        scatter(
            target=result_array,
            index=index_out,
            value=arg_array,
            mode=ReduceMode.Permute
        )

    return result

class _ResampleOp(CustomOp):
    """Implementation detail of the function drjit.resample()"""
    def eval(self, resampler, source, stride):
        self.resampler, self.stride = resampler, stride
        return type(source)(resampler.resample_fwd(detach(source, False), stride))

    def forward(self):
        grad_source = detach(self.grad_in('source'), False)
        self.set_grad_out(
            self.resampler.resample_fwd(grad_source, self.stride)
        )

    def backward(self):
        grad_out = detach(self.grad_out(), False)

        self.set_grad_in(
            'source',
            self.resampler.resample_bwd(grad_out, self.stride)
        )

_resample_cache = {}

def resample(
    source: ArrayT,
    shape: Sequence[int],
    *,
    filter: Union[Literal["box", "linear", "hamming", "cubic", "lanczos"], Callable[[float], float]] = "cubic",
    filter_radius: Optional[float] = None
) -> ArrayT:
    """
    Resample an input array/tensor to increase or decrease its resolution along
    a set of axes.

    This function up- and/or downsamples a given array or tensor along a
    specified set of axes. Given an input array (``source``) and target shape
    (``shape``), it returns a compatible array of the specified configuration.
    This is implemented using a sequence of successive 1D resampling steps for
    each mismatched axis.

    Example usage:

    .. code-block:: python

       image: TensorXf = ...  # a RGB image
       width, height, channels = image.shape

       scaled_image = dr.resample(
           image,
           (width // 2, height // 2, channels)
       )

    Resampling uses a `reconstruction filter
    <https://en.wikipedia.org/wiki/Reconstruction_filter>`__. The following
    options are available, where :math:`n` refers to the number of dimensions
    being resampled:

    - ``"box"``: use nearest-neighbor interpolation/averaging. This is very
      efficient but generally produces sub-par output that is either pixelated
      (when upsampling) or aliased (when downsampling).

    - ``"linear"``: use linear ramp / tent filter that uses :math:`2^n`
      neighbors to reconstruct each output sample when upsampling. Tends to
      produce relatively blurry results.

    - ``"hamming"``: uses the same number of input samples as ``"linear"`` but
      better preserves sharpness when downscaling. Do not use for upscaling.

    - ``"cubic"``: use cubic filter kernel that uses :math:`4^n`
      neighbors to reconstruct each output sample when upsampling. Produces
      high-quality results. This is the default.

    - ``"lanczos"``: use a windowed Lanczos filter that uses :math:`6^n`
      neighbors to reconstruct each output sample when upsampling. This is the
      best filter for smooth signals, but also the costliest. The Lanczos
      filter is susceptible to ringing when the input array contains
      discontinuities.

    - Besides the above choices, it is also possible to specify a custom filter.
      To do so, use the ``filter`` argument to pass a Python callable with
      signature ``Callable[[float], float]``. In this case, you must also
      specify a filter radius via the ``filter_radius`` parameter.

    The implementation was extensively tested against `Image.resize()
    <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize>`__
    from the Pillow library and should be a drop-in replacement with added
    support for JIT tracing / GPU evaluation, differentiability, and
    compatibility with higher-dimensional tensors.

    .. warning::

       When using ``filter="hamming"``, ``"cubic"``, or ``"lanczos"``, the
       range of the output array can exceed that of the input array. For
       example, positive-valued data may contain negative values following
       resampling. Clamp the output in case it is important that array values
       remain within a fixed range (e.g., :math:`[0,1]`).

    Args:
        source (dr.ArrayBase): The Dr.Jit tensor or 1D array to be resampled.

        shape (Sequence[int]): The desired output shape.

        filter (str | Callable[[float], float])
          The desired reconstruction filter, see the above text for an overview.
          Alternatively, a custom reconstruction filter function can also be
          specified.

        filter_radius (float | None)
          The radius of the pixel filter in the output sample space. Should
          only be specified when using a custom reconstruction filter.

    Returns:
        drjit.ArrayBase: The resampled output array. Its type matches
        ``source``, and its shape matches ``shape``.
    """

    source_shape = source.shape
    strides = _compute_strides(shape)
    ndim = len(source_shape)
    tp = type(source)
    value = source.array

    if len(shape) != ndim:
        raise RuntimeError(
            "drjit.resample(): 'source' and 'shape' must have the same number of axes."
        )

    for i in reversed(range(ndim)):
        source_res = source_shape[i]
        target_res = shape[i]

        if source_res == target_res:
            continue

        # Cache resampler in case it can be reused
        key = (source_res, target_res, filter, filter_radius)

        resampler = _resample_cache.get(key, None)
        if resampler is None:
            resampler = detail.Resampler(
                source_res=source_res,
                target_res=target_res,
                filter=filter,
                filter_radius=filter_radius,
            )
            _resample_cache[key] = resampler

        value = custom(_ResampleOp,
            resampler=resampler,
            source=value,
            stride=strides[i])

    if is_tensor_v(tp):
        return tp(value, shape)
    else:
        return value


def upsample(t, shape=None, scale_factor=None):
    '''
    upsample(source, shape=None, scale_factor=None)
    Up-sample the input tensor or texture according to the provided shape.

    Alternatively to specifying the target shape, a scale factor can be provided.

    The behavior of this function depends on the type of ``source``:

    1. When ``source`` is a Dr.Jit tensor, nearest neighbor up-sampling will use
    hence the target ``shape`` values must be multiples of the source shape
    values. When `scale_factor` is used, its values must be integers.

    2. When ``source`` is a Dr.Jit texture type, the up-sampling will be
    performed according to the filter mode set on the input texture. Target
    ``shape`` values are not required to be multiples of the source shape values.
    When `scale_factor` is used, its values must be integers.

    .. warning::

       This function is deprecated and will be removed in a future release.
       Instead, please use the function :py:func:`drjit.resample()`.

    Args:
        source (object): A Dr.Jit tensor or texture type.

        shape (list): The target shape (optional)

        scale_factor (list): The scale factor to apply to the current shape (optional)

    Returns:
        object: the up-sampled tensor or texture object. The type of the output will be the same as the type of the source.
    '''
    from collections.abc import Sequence as _Sequence

    _warnings.warn("drjit.upsample() is deprecated, please use drjit.resample() instead.",
                   DeprecationWarning, stacklevel=2)

    if  not getattr(t, 'IsTexture', False) and not is_tensor_v(t):
        raise TypeError("upsample(): unsupported input type, expected Jit "
                        "tensor or texture type!")

    if shape is not None and scale_factor is not None:
        raise TypeError("upsample(): shape and scale_factor arguments cannot "
                        "be defined at the same time!")

    if shape is not None:
        if not isinstance(shape, _Sequence):
            raise TypeError("upsample(): unsupported shape type, expected a list!")

        if len(shape) > len(t.shape):
            raise TypeError("upsample(): invalid shape size!")

        shape = list(shape) + list(t.shape[len(shape):])

        scale_factor = []
        for i, s in enumerate(shape):
            if type(s) is not int:
                raise TypeError("upsample(): target shape must contain integer values!")

            if s < t.shape[i]:
                raise TypeError("upsample(): target shape values must be larger "
                                "or equal to input shape! (%i vs %i)" % (s, t.shape[i]))

            if is_tensor_v(t):
                factor = s / float(t.shape[i])
                if factor != int(factor):
                    raise TypeError("upsample(): target shape must be multiples of "
                                    "the input shape! (%i vs %i)" % (s, t.shape[i]))
    else:
        if not isinstance(scale_factor, _Sequence):
            raise TypeError("upsample(): unsupported scale_factor type, expected a list!")

        if len(scale_factor) > len(t.shape):
            raise TypeError("upsample(): invalid scale_factor size!")

        scale_factor = list(scale_factor)
        for i in range(len(t.shape) - len(scale_factor)):
            scale_factor.append(1)

        shape = []
        for i, factor in enumerate(scale_factor):
            if type(factor) is not int:
                raise TypeError("upsample(): scale_factor must contain integer values!")

            if factor < 1:
                raise TypeError("upsample(): scale_factor values must be greater "
                                "than 0!")

            shape.append(factor * t.shape[i])

    if getattr(t, 'IsTexture', False):
        value_type = type(t.value())
        dim = len(t.shape) - 1

        if t.shape[dim] != shape[dim]:
            raise TypeError("upsample(): channel counts doesn't match input texture!")

         # Create the query coordinates
        coords = list(meshgrid(*[
                linspace(value_type, 0.0, 1.0, shape[i], endpoint=False)
                for i in range(dim)
            ],
            indexing='ij'
        ))

        # Offset coordinates by half a voxel to hit the center of the new voxels
        for i in range(dim):
            coords[i] += 0.5 / shape[i]

        # Reverse coordinates order according to dr.Texture convention
        coords.reverse()

        # Evaluate the texture at all voxel coordinates with interpolation
        values = t.eval(coords)

        # Concatenate output values to a flatten buffer
        channels = len(values)
        w = width(values[0])
        index = arange(uint32_array_t(value_type), w)
        data = zeros(value_type, w * channels)
        for c in range(channels):
            scatter(data, values[c], channels * index + c)

        # Create the up-sampled texture
        texture = type(t)(shape[:-1], channels,
                          use_accel=t.use_accel(),
                          filter_mode=t.filter_mode(),
                          wrap_mode=t.wrap_mode())
        texture.set_value(data)

        return texture
    else:
        dim = len(shape)
        size = prod(shape[:dim])
        base = arange(uint32_array_t(type(t.array)), size)

        index = 0
        stride = 1
        for i in reversed(range(dim)):
            ratio = shape[i] // t.shape[i]
            index += (base // ratio % t.shape[i]) * stride
            base //= shape[i]
            stride *= t.shape[i]

        return type(t)(gather(type(t.array), t.array, index), tuple(shape))


_rand_seed : int = 0

def seed(value: int):
    """
    Reset the seed value that is used for pseudorandom number generation.

    Every successive call to :py:func:`rand` and :py:func:`normal` (without
    manually specified ``seed``) increments an internal counter that is used to
    initialize the random number generator to ensure independent output.

    This function can be used to reset this counter to a specific value.
    """
    global _rand_seed
    _rand_seed = value

def rand(dtype: Type[ArrayT],
         shape: Union[int, Tuple[int, ...]],
         *,
         seed: Union[int, AnyArray, None] = None,
         version: int = 1,
         _func_name='next_float') -> ArrayT:
    """
    Return a Dr.Jit array or tensor containing uniformly distributed
    pseudorandom variates.

    This function supports floating point arrays/tensors of various
    configurations and precisions, e.g.:

    .. code-block:: python

       from drjit.cuda import Float, TensorXf, Array3f, Matrix4f

       # Example usage
       rand_array = dr.rand(Float, 128)
       rand_tensor = dr.rand(TensorXf16, (128, 128))
       rand_vec = dr.rand(Array3f, (3, 128))
       rand_mat = dr.rand(Matrix4f64, (4, 4, 128))

    The output is uniformly distributed the interval :math:`[0, 1)`. Integer
    arrays are not supported.

    Successive calls to :py:func:`drjit.rand()` produce independent random
    variates. You can manually specify a 64-bit integer via the ``seed``
    parameter to avoid this. Use the :py:func:`drjit.seed()` function to reset
    the global default seed value.

    .. warning::

       This function is still considered experimental, and the algorithm used
       to generate random variates may change in future versions of Dr.Jit.
       Specify ``version=1`` to to ensure that your program remains unaffected
       by such future changes.

    .. note::

       When this function is used within a symbolic operation (e.g.
       :py:func:`drjit.while_loop()`), you *must* provide the ``seed``
       parameter.

       In the non-symbolic case, the seed parameter is internally made opaque
       via :py:func:`drjit.make_opaque` so that the use of this function does
       not interfere with kernel caching.

       In applications that require repeated generation of random variates
       (e.g., in a symbolic loop), is more efficient to directly work with the
       underlying random number generator (e.g., :py:class:`drjit.cuda.PCG32`)
       instead of using the high-level :py:func:`drjit.rand` interface.

    Args:
        source (type[ArrayT]): A Dr.Jit tensor or array type.

        shape (int | tuple[int, ...]): The target shape

        seed (int | None): A seed value used to initialize the random number generator.
          If no value is provided, a global seed value is used (and then subsequently
          incremented). Refer to :py:func:`drjit.seed()`.

        version (int): Optional parameter to target a specific implementation
          of this function in the case of future changes.

    Returns:
        ArrayT: The generated array of random variates.
    """

    global _rand_seed

    if isinstance(shape, int):
        shape = (shape, )

    # Resolve details about the array type
    is_jit = is_jit_v(dtype)
    is_tensor = is_tensor_v(dtype)
    value_tp = leaf_t(dtype)
    seed_tp = uint64_array_t(value_tp)
    rng_tp = _sys.modules[value_tp.__module__].PCG32

    # Compute an opaque seed value
    if flag(JitFlag.SymbolicScope):
        if seed is None:
            raise Exception("drjit.rand(): when used within a symbolic "
                            "operation, you *must* provide the 'seed' "
                            "parameter'")
        seed_v = seed_tp(seed)
    else:
        seed_v = seed_tp(_rand_seed if seed is None else seed)
        make_opaque(seed_v)

    if is_tensor:
        size = prod(shape)
    else:
        size = shape[-1]

    # Construct a suitably sized PCG32 instance
    if version == 1:
        if is_jit:
            rng = rng_tp(size, seed_v)
            leaf_tp = value_tp
        else:
            rng = rng_tp(1, seed_v[0])
            leaf_tp = float
        func = getattr(rng, _func_name)
    else:
        raise Exception("drjit.rand(): unsupported 'version' specified!")

    if depth_v(dtype) <= 1:
        # Default case: tensors, 1D arrays
        if is_jit:
            value = func(leaf_tp)
        else:
            value = value_tp(func(leaf_tp) for _ in range(size))
    else:
        # Complex case: vectors, matrices, etc.
        value = empty(dtype, shape)

        def fill(v):
            if depth_v(v) == 1:
                if is_jit:
                    return func(leaf_tp)
                else:
                    return value_tp(func(leaf_tp) for _ in range(size))

            for i in range(len(v)):
                v[i] = fill(v[i])

        fill(value)

    if seed is None:
        _rand_seed += 1

    if is_tensor:
        return dtype(value, shape)
    else:
        return value

def normal(dtype: Type[ArrayT],
           shape: Union[int, Tuple[int, ...]],
           *,
           seed: Union[int, AnyArray, None] = None,
           version: int = 1) -> ArrayT:
    """
    Return a Dr.Jit array or tensor containing pseudorandom variates
    following a standard normal distribution

    Please refer to :py:func:`drjit.rand()`, the interfaces of these
    two functions are identical.
    """

    return rand(
        dtype=dtype,
        shape=shape,
        seed=seed,
        version=version,
        _func_name='next_float_normal'
    )

def binary_search(start, end, pred):
    '''
    Perform a binary search over a range given a predicate ``pred``, which
    monotonically decreases over this range (i.e. max one ``True`` -> ``False``
    transition).

    Given a (scalar) ``start`` and ``end`` index of a range, this function
    evaluates a predicate ``floor(log2(end-start) + 1)`` times with index
    values on the interval [start, end] (inclusive) to find the first index
    that no longer satisfies it. Note that the template parameter ``Index`` is
    automatically inferred from the supplied predicate. Specifically, the
    predicate takes an index array as input argument. When ``pred`` is ``False``
    for all entries, the function returns ``start``, and when it is ``True`` for
    all cases, it returns ``end``.

    The following code example shows a typical use case: ``data`` contains a
    sorted list of floating point numbers, and the goal is to map floating
    point entries of ``x`` to the first index ``j`` such that ``data[j] >= threshold``
    (and all of this of course in parallel for each vector element).

    .. code-block::

        dtype = dr.llvm.Float
        data = dtype(...)
        threshold = dtype(...)

        index = dr.binary_search(
            0, len(data) - 1,
            lambda index: dr.gather(dtype, data, index) < threshold
        )

    Args:
        start (int): Starting index for the search range
        end (int): Ending index for the search range
        pred (function): The predicate function to be evaluated

    Returns:
        Index array resulting from the binary search
    '''
    assert isinstance(start, int) and isinstance(end, int)

    iterations = log2i(end - start) + 1 if start < end else 0

    for _ in range(iterations):
        middle = (start + end) >> 1

        cond = pred(middle)
        start = select(cond, minimum(middle + 1, end), start)
        end = select(cond, end, middle)

    return start


def assert_true(
    cond,
    fmt: Optional[str] = None,
    *args,
    tb_depth: int = 3,
    tb_skip: int = 0,
    **kwargs,
):
    """
    Generate an assertion failure message when any of the entries in ``cond``
    are ``False``.

    This function resembles the built-in ``assert`` keyword in that it raises
    an ``AssertionError`` when the condition ``cond`` is ``False``.

    In contrast to the built-in keyword, it also works when ``cond`` is an
    array of boolean values. In this case, the function raises an exception
    when *any* entry of ``cond`` is ``False``.

    The function accepts an optional format string along with positional and
    keyword arguments, and it processes them like :py:func:`drjit.print`. When
    only a subset of the entries of ``cond`` is ``False``, the function reduces
    the generated output to only include the associated entries.

    .. code-block:: python

       >>> x = Float(1, -4, -2, 3)
       >>> dr.assert_true(x >= 0, 'Found negative values: {}', x)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
         File "drjit/__init__.py", line 1327, in assert_true
           raise AssertionError(msg)
       AssertionError: Assertion failure: Found negative values: [-4, -2]

    This function also works when some of the function inputs are *symbolic*.
    In this case, the check is delayed and potential failures will be reported
    asynchronously. In this case, :py:func:`drjit.assert_true` generates output
    on ``sys.stderr`` instead of raising an exception, as the original
    execution context no longer exists at that point.

    Assertion checks carry a performance cost, hence they are disabled by
    default. To enable them, set the JIT flag :py:attr:`dr.JitFlag.Debug`.

    Args:
        cond (bool | drjit.ArrayBase): The condition used to trigger the
          assertion. This should be a scalar Python boolean or a 1D boolean
          array.

        fmt (str): An optional format string that will be appended to
          the error message. It can reference positional or keyword
          arguments specified via ``*args`` and ``**kwargs``.

        *args (tuple): Optional variable-length positional arguments referenced
          by ``fmt``, see :py:func:`drjit.print` for details on this.

        tb_depth (int): Depth of the backtrace that should be appended to the
          assertion message. This only applies to cases some of the inputs are
          symbolic, and printing of the error message must be delayed.

        tb_skip (int): The first ``tb_skip`` entries of the backtrace will be
          removed. This only applies to cases some of the inputs are symbolic,
          and printing of the error message must be delayed. This is helpful when
          the assertion check is called from a helper function that should not be
          shown.

        **kwargs (dict): Optional variable-length keyword arguments referenced
          by ``fmt``, see :py:func:`drjit.print` for details on this.
    """

    if not flag(JitFlag.Debug):
        return
    if cond is True or (not detail.any_symbolic(cond) and all(cond)):
        return

    import traceback, types

    active = not cond if isinstance(cond, bool) else ~cond

    if detail.any_symbolic((active, args, kwargs)):
        tb_frame = _sys._getframe(tb_skip + 1)
        tb = types.TracebackType(tb_next=None,
                                 tb_frame=tb_frame,
                                 tb_lasti=tb_frame.f_lasti,
                                 tb_lineno=tb_frame.f_lineno)

        tb_msg = "".join(traceback.format_tb(tb, limit=tb_depth))

        # Note: this is not a regular print statement -- it maps to 'drjit.print'
        print(
            f"Assertion failure" + ((': ' + fmt) if fmt else '!') + "\n{tb_msg}",
            *args,
            tb_msg=tb_msg,
            active=active,
            file=_sys.stderr,
            **kwargs
        )

    else:
        # Note: this is not a regular format statement -- it maps to 'drjit.format'
        msg = format(
            f"Assertion failure" + ((': ' + fmt) if fmt else '!'),
            *args,
            active=active,
            **kwargs
        )

        raise AssertionError(msg)


def assert_false(
    cond,
    fmt: Optional[str] = None,
    *args,
    tb_depth: int = 3,
    tb_skip: int = 0,
    **kwargs,
):
    """
    Equivalent to :py:func:`assert_true` with a flipped condition ``cond``.
    Please refer to the documentation of this function for further details.
    """
    return assert_true(
        not cond if isinstance(cond, bool) else ~cond,
        fmt,
        *args,
        tb_depth=tb_depth,
        tb_skip=tb_skip+1,
        **kwargs,
    )

def assert_equal(
    arg0,
    arg1,
    fmt: Optional[str] = None,
    *args,
    limit: int = 3,
    tb_skip: int = 0,
    **kwargs,
):
    """
    Equivalent to :py:func:`assert_true` with the condition ``arg0==arg1``.
    Please refer to the documentation of this function for further details.
    """
    return assert_true(
        arg0 == arg1,
        fmt,
        *args,
        limit=limit,
        tb_skip=tb_skip+1,
        **kwargs,
    )


newaxis = None

del overload, Optional
