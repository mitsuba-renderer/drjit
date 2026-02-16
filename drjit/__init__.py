from __future__ import annotations

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
        from typing_extensions import overload, Optional, Type, Tuple, List, Sequence, Union, Literal, Callable, TypeVar
    except ImportError:
        raise RuntimeError(
            "Dr.Jit requires the 'typing_extensions' package on Python <3.11")
else:
    from typing import overload, Optional, Type, Tuple, List, Sequence, Union, Literal, Callable, TypeVar

from .ast import syntax, hint
from .interop import wrap
from . import random
import warnings as _warnings

from functools import wraps


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

    return fma(b, t, fma(a, -t, a))

def relative_grad(x: dr.ArrayBase):
    """
    Create a factor with primal value ``1`` that injects a *relative*
    first-order derivative with respect to ``x``.

    For nonzero ``x``, the returned value has the same first-order derivative as

        x / detach(x)

    so gradient contributions depend on the *relative change* ``dx/x`` rather
    than the absolute scale of ``x``.

    The implementation handles zero-valued inputs gracefully.
    """
    grad_source = select(x != 0, x * detach(rcp(x)), 0)
    return replace_grad(1, grad_source)


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
        raise Exception('drjit.transform_compose(): scale has invalid shape!')

    if len(t) != 3:
        raise Exception('drjit.transform_compose(): translation has invalid shape!')

    m = _sys.modules[s.__module__]
    Matrix3f = replace_type_t(m.Matrix3f, type_v(s))
    Matrix4f = replace_type_t(m.Matrix4f, type_v(s))

    m33 = Matrix3f(quat_to_matrix(Matrix3f, q) @ s)

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

def sphdir(theta, phi):
    '''
    Spherical coordinate parameterization of the unit sphere

    Args:
        theta (float | drjit.ArrayBase): Elevation angle in radians, measured
            from the positive Z axis. Valid range is [0, π].

        phi (float | drjit.ArrayBase): Azimuth angle in radians, measured from
            the positive X axis in the XY plane. Valid range is [0, 2π].

    Returns:
        drjit.ArrayBase: A 3D unit direction vector corresponding to the input
        spherical coordinates. The result is a 3-component array with unit length.
    '''
    st, ct = sincos(theta)
    sp, cp = sincos(phi)

    import sys
    tp = type(theta)
    m = sys.modules[tp.__module__]
    Array3f = replace_type_t(m.Array3f, type_v(theta))

    return Array3f(cp * st, sp * st, ct)

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
           from scipy.special import sph_harm_y
           theta, phi = np.arccos(d.z), np.arctan2(d.y, d.x)
           r = []
           for l in range(order + 1):
               for m in range(-l, l + 1):
                   Y = sph_harm(l, abs(m), theta, phi)
                   if m > 0:
                       Y = np.sqrt(2) * Y.real
                   elif m < 0:
                       Y = np.sqrt(2) * Y.imag
                   r.append(Y.real)
           return r

    The Mathematica equivalent of a specific entry is given by:

    .. code-block:: wolfram

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
        raise RuntimeError("sh_eval(): order must be in [0, 9]")
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
    filter: Union[Literal["box", "linear", "hamming", "cubic", "lanczos", "gaussian"], Callable[[float], float]] = "cubic",
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

    - ``"cubic"``: use cubic filter kernel that queries :math:`4^n`
      neighbors to reconstruct each output sample when upsampling. Produces
      high-quality results. This is the default.

    - ``"lanczos"``: use a windowed Lanczos filter that queries :math:`6^n`
      neighbors to reconstruct each output sample when upsampling. This is the
      best filter for smooth signals, but also the costliest. The Lanczos
      filter is susceptible to ringing when the input array contains
      discontinuities.

    - ``"gaussian"``: use a Gaussian filter that queries :math:4^n` neighbors
      to reconstruct each output sample when upsampling. The kernel has a
      standard deviation of 0.5 and is truncated after 4 standard deviations.
      This filter is mainly useful when intending to blur a signal.

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

def convolve(
    source: ArrayT,
    filter: Union[Literal["box", "linear", "hamming", "cubic", "lanczos", "gaussian"], Callable[[float], float]],
    filter_radius: float,
    axis: Union[int, Tuple[int, ...], None] = None
) -> ArrayT:
    """
    Convolve one or more axes of an input array/tensor with a 1D filter

    This function filters one or more axes of a Dr.Jit array or tensor, for
    example to convolve an image with a 2D Gaussian filter to blur spatial
    detail.

    .. code-block:: python

       image: TensorXf = ...  # a RGB image

       blured_image = dr.convolve(
           image,
           filter='gaussian',
           filter_radius=10
       )

    The filter weights are renormalized to reduce edge effects near the
    boundary of the array.

    The function supports a set of provided filters, and custom filters
    can also be specified. This works analogously to the :py:func:`resample`
    function, please refer to its documentation for detail.

    Args:
        source (dr.ArrayBase): The Dr.Jit tensor or 1D array to be resampled.

        filter (str | Callable[[float], float])
          The desired reconstruction filter, see the above text for an overview.
          Alternatively, a custom reconstruction filter function can also be
          specified.

        filter_radius (float)
          The radius of the continous function to be used in the convolution.

        axis (int | tuple[int, ...] | ... | None): The axis or set of axes
          along which to convolve. The default argument ``axis=None`` causes all
          axes to be convolved. Negative values count from the last dimension.

    Returns:
        drjit.ArrayBase: The resampled output array. Its type matches ``source``.
    """

    shape = source.shape
    strides = _compute_strides(shape)
    ndim = len(shape)
    tp = type(source)
    value = source.array

    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis, )

    for i in axis:
        if i < 0:
            i = ndim + i
        res = shape[i]

        # Cache resampler in case it can be reused
        key = (res, res, filter, filter_radius)

        resampler = _resample_cache.get(key, None)
        if resampler is None:
            resampler = detail.Resampler(
                source_res=res,
                target_res=res,
                filter=filter,
                filter_radius=filter_radius,
                convolve=True
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


def _normalize_axis_tuple(t: Union[int, Tuple[int, ...]], ndim: int, name: str) -> List[int]:
    if isinstance(t, int):
        t = (t, )
    axes = []
    for i in t:
        if i < 0:
            i += ndim
        if i < 0 or i >= ndim:
            raise RuntimeError(f"'{name}' axis is out of bounds")
        axes.append(i)
    if len(set(axes)) != len(axes):
        raise RuntimeError(f"'{name}' contains repeated axes")
    return axes


def moveaxis(arg: ArrayBase, /, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]):
    """
    Move one or more axes of an input tensor to another position.

    Dimensions of that are not explicitly moved remain in their original order
    and appear at the positions not specified in the destination. Negative axis
    values count backwards from the end.
    """

    if not is_tensor_v(arg):
        raise TypeError("drjit.moveaxis(): expects a tensor instance as input!")

    shape_in = arg.shape
    ndim = len(shape_in)
    source_l = _normalize_axis_tuple(source, ndim, 'source')
    destination_l = _normalize_axis_tuple(destination, ndim, 'destination')

    if len(source_l) != len(destination_l):
        raise ValueError("'source' and 'destination` must have the "
                         "same number of elements")

    # Determine the final axis order (based on NumPy)
    order = [n for n in range(ndim) if n not in source_l]
    for dest, src in sorted(zip(destination_l, source_l)):
        order.insert(dest, src)

    shape_out = tuple(shape_in[i] for i in order)
    strides_in = _compute_strides(shape_in)
    strides_out = _compute_strides(shape_out)

    last_axis = 0
    for i, j in enumerate(order):
        if i != j:
            last_axis = i + 1

    arr = arg.array
    index_out = arange(uint32_array_t(arr), prod(shape_in))
    index_in = 0

    for i in range(last_axis):
        pos = index_out // strides_out[i]
        index_out -= pos * strides_out[i]
        index_in += pos * strides_in[order[i]]

    index_in += index_out

    return type(arg)(gather(type(arr), arr, index_in, mode=ReduceMode.Permute), shape_out)


def take(value: ArrayT, index: Union[int, ArrayBase], axis: int = 0) -> ArrayT:
    """
    Select values from a tensor along a specified axis using an index or index array.

    This function evaluates ``value[..., index, ...]`` where ``index`` is
    applied at position ``axis``. The output tensor has one fewer dimension
    than the input.

    Args:
        value (drjit.ArrayBase): Input tensor

        index (Union[int, drjit.ArrayBase]): Integer or 1D integer array.

        axis (int): Axis along which to select values. Negative values count
            from the end. The default is 0.

    Returns:
        drjit.ArrayBase: Output tensor with shape equal to the input shape
        minus the indexed axis dimension. The dtype matches the input tensor.
    """

    shape = value.shape
    ndim = len(shape)

    # Handle negative axis
    if axis < 0:
        axis += ndim

    if not is_tensor_v(value):
        raise TypeError("drjit.take(): expects a tensor instance as input!")

    if axis < 0 or axis >= ndim:
        raise RuntimeError(f"drjit.take(): tensor axis {axis} is out of bounds for tensor with {ndim} dimensions!")

    # Get array types
    array = value.array if is_tensor_v(value) else value
    Array = type(array)
    Index = uint32_array_t(Array)

    # Compute new shape
    if is_array_v(index) and len(index) > 1:
        index = Index(index)
        index_dim = index.shape
        index_len = index_dim[0]
    else:
        index_dim = ()
        index_len = 1

    new_shape = shape[:axis] + index_dim + shape[axis+1:]

    # Compute total size and stride after the axis
    total = prod(new_shape) if new_shape else 1
    stride_after = prod(shape[axis+1:]) if axis < ndim - 1 else 1

    result_idx = arange(Index, total)

    if index_len != 1:
        selected_index = index[(result_idx // stride_after) % index_len]
    else:
        selected_index = Index(index)

    flat_idx = (result_idx % stride_after) + selected_index * stride_after

    # Compute flat index for gathering
    if axis > 0:
        full_stride = shape[axis] * stride_after
        before_axis_idx = result_idx // (stride_after * index_len)
        flat_idx += before_axis_idx * full_stride

    return type(value)(gather(Array, array, flat_idx), new_shape)


def take_interp(value: ArrayT, pos: Union[float, ArrayBase], axis: int = 0) -> ArrayT:
    """
    Select and interpolate values from a tensor along a specified axis using
    fractional indices.

    Similar to :py:func:`drjit.take`, but accepts fractional positions and
    performs linear interpolation between adjacent values along the specified
    axis. This is useful for smooth sampling from discrete data.

    Args:
        value (drjit.ArrayBase): Input tensor

        pos (Union[float, drjit.ArrayBase]): Python ``float`` or 1D float array.

        axis (int): Axis along which to interpolate values. Negative values
            count from the end. Default is 0.

    Returns:
        drjit.ArrayBase: Output tensor with shape equal to the input shape
        minus the indexed axis dimension. Values are linearly interpolated
        based on the fractional indices. The dtype matches the input tensor.
    """
    shape = value.shape
    ndim = len(shape)

    # Handle negative axis
    if axis < 0:
        axis += ndim

    if not is_tensor_v(value):
        raise TypeError("drjit.take_interp(): expects a tensor instance as input!")

    if axis < 0 or axis >= ndim:
        raise RuntimeError(f"drjit.take_interp(): tensor axis {axis} is out of bounds for tensor with {ndim} dimensions!")

    if shape[axis] < 2:
        raise RuntimeError(f"drjit.take_interp(): tensor axis {axis} has size {shape[axis]}, but must have at least 2 elements for interpolation!")

    # Get array types
    array = value.array if is_tensor_v(value) else value
    Array = type(array)
    Index = uint32_array_t(Array)
    Float = float_array_t(Array)

    index = clip(Index(pos), 0, shape[axis] - 2)

    # Compute new shape
    if is_array_v(index) and len(index) > 1:
        index = Index(index)
        index_dim = index.shape
        index_len = index_dim[0]
    else:
        index_dim = ()
        index_len = 1

    new_shape = shape[:axis] + index_dim + shape[axis+1:]

    # Compute total size and stride after the axis
    total = prod(new_shape) if new_shape else 1
    stride_after = prod(shape[axis+1:]) if axis < ndim - 1 else 1

    result_idx = arange(Index, total)

    if index_len != 1:
        selected_index = index[(result_idx // stride_after) % index_len]
    else:
        selected_index = Index(index)

    flat_idx = (result_idx % stride_after) + selected_index * stride_after

    # Compute interpolation weights
    w1 = Float(pos) - Float(index)
    w0 = 1.0 - w1

    # Compute flat index for gathering
    if axis > 0:
        full_stride = shape[axis] * stride_after
        before_axis_idx = result_idx // (stride_after * index_len)
        flat_idx += before_axis_idx * full_stride

    # Gather adjacent values and interpolate
    v0 = gather(Array, array, flat_idx)
    v1 = gather(Array, array, flat_idx + stride_after)

    return type(value)(fma(v0, w0, v1 * w1), new_shape)


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


def rng(seed: Union[ArrayBase, int] = 0, method='philox4x32', symbolic: bool = False) -> random.Generator:
    '''
    Return a seeded random number generator.

    This function returns a :py:class:`drjit.random.Generator` object. Note the
    following:

    - Differently seeded random number generators produce statistically
      independent streams of random variates.

    - ``seed`` can be a Python `int` or Dr.Jit :py:class:`UInt64
      <drjit.auto.UInt64>`-typed array. The default value ``0`` is used when no
      seed is specified, making the generator's behavior deterministic across runs.

    - Only ``method=philox4x32`` is supported at the moment. This returns a
      generator object wrapping the :py:class:`Philox4x32
      <drjit.auto.Philox4x32>` counter-based PRNG.

    - When ``symbolic=True`` is specified, the internal sampler state will
      never be explicitly evaluated. This is useful in cases where you wish
      to explicitly bake these constants into the generated program. Dr.Jit
      also detects when the sampler is used in a symbolic code block (e.g.,
      a symbolic loop) and automaticallky sets this flag in such a case.
    '''

    if method == 'philox4x32':
        return random.Philox4x32Generator(seed, symbolic=symbolic)
    else:
        raise RuntimeError("Only generator='philox4x32' is currently supported.")


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

# Represents the frozen function passed to the decorator without arguments
F = TypeVar("F")
# Represents the frozen function passed to the decorator with arguments
F2 = TypeVar("F2")


@overload
def freeze(
    f: None = None,
    *,
    state_fn: Optional[Callable],
    limit: Optional[int] = None,
    warn_after: int = 10,
    backend: Optional[JitBackend] = None,
    auto_opaque: bool = True,
    enabled: bool = True,
) -> Callable[[F], F]:
    ...


@overload
def freeze(
    f: F,
    *,
    state_fn: Optional[Callable] = None,
    limit: Optional[int] = None,
    warn_after: int = 10,
    backend: Optional[JitBackend] = None,
    auto_opaque: bool = True,
    enabled: bool = True,
) -> F:
    ...


def freeze(
    f: Optional[F] = None,
    *,
    state_fn: Optional[Callable] = None,
    limit: Optional[int] = None,
    warn_after: int = 10,
    backend: Optional[JitBackend] = None,
    auto_opaque: bool = True,
    enabled: bool = True,
) -> Union[F, Callable[[F2], F2]]:
    """
    Decorator to "freeze" functions, which improves efficiency by removing
    repeated JIT tracing overheads.

    In general, Dr.Jit traces computation and then compiles and launches kernels
    containing this trace (see the section on :ref:`evaluation <eval>` for
    details). While the compilation step can often be skipped via caching, the
    tracing cost can still be significant especially when repeatedly evaluating
    complex models, e.g., as part of an optimization loop.

    The :py:func:`@dr.freeze <drjit.freeze>` decorator adresses this problem by
    altogether removing the need to trace repeatedly. For example, consider the
    following decorated function:

    .. code-block:: python

       @dr.freeze
       def f(x, y, z):
          return ... # Complicated code involving the arguments

    Dr.Jit will trace the first call to the decorated function ``f()``, while
    collecting additional information regarding the nature of the function's inputs
    and regarding the CPU/GPU kernel launches representing the body of ``f()``.

    If the function is subsequently called with *compatible* arguments (more on
    this below), it will immediately launch the previously made CPU/GPU kernels
    without re-tracing, which can substantially improve performance.

    When :py:func:`@dr.freeze <drjit.freeze>` detects *incompatibilities* (e.g., ``x``
    having a different type compared to the previous call), it will conservatively
    re-trace the body and keep track of another potential input configuration.

    Frozen functions support arbitrary :ref:`PyTrees <pytrees>` as function
    arguments and return values.

    The following may trigger re-tracing:

    - Changes in the **type** of an argument or :ref:`PyTree <pytrees>` element.
    - Changes in the **length** of a container (``list``, ``tuple``, ``dict``).
    - Changes of **dictionary keys**  or **field names** of dataclasses.
    - Changes in the AD status (:py:func:`dr.grad_enabled() <drjit.grad_enabled>`) of a variable.
    - Changes of (non-PyTree) **Python objects**, as detected by mismatching ``hash()``
      or ``id()`` if they are not hashable.

    The following more technical conditions also trigger re-tracing:

    - A Dr.Jit variable changes from/to a **scalar** configuration (size ``1``).
    - The sets of variables of the same size change. In the example above, this
      would be the case if ``len(x) == len(y)`` in one call, and ``len(x) != len(y)``
      subsequently.
    - When Dr.Jit variables reference external memory (e.g. mapped NumPy arrays), the
      memory can be aligned or unaligned. A re-tracing step is needed when this
      status changes.

    These all correspond to situations where the generated kernel code may need to
    change, and the system conservatively re-traces to ensure correctness.

    Frozen functions support arguments with a different variable *width* (see
    :py:func:`dr.with() <drjit.width>`) without re-tracing, as long as the sets of
    variables of the same width stay consistent.

    Some constructions are problematic and should be avoided in frozen functions.

    - The function :py:func:`dr.width() <drjit.width>` returns an integer literal
      that may be merged into the generated code. If the frozen function is later
      rerun with differently-sized arguments, the executed kernels will still
      reference the old size. One exception to this rule are constructions like
      `dr.arange(UInt32, dr.width(a))`, where the result only implicitly depends on
      the width value.

    When calling a frozen function from within an outer frozen function, the content
    of the inner function will be executed and recorded by the outer function.
    No separate recording will be made for the inner function, and its ``n_recordings``
    count will not change. Calling the inner function separately from outside a
    frozen function will therefore require re-tracing for the provided inputs.

    **Advanced features**. The :py:func:`@dr.freeze <drjit.freeze>` decorator takes
    several optional parameters that are helpful in certain situations.

    - **Warning when re-tracing happens too often**: Incompatible arguments trigger
      re-tracing, which can mask issues where *accidentally* incompatible arguments
      keep :py:func:`@dr.freeze <drjit.freeze>` from producing the expected
      performance benefits.

      In such situations, it can be helpful to warn and identify changing
      parameters by name. This feature is enabled and set to ``10`` by default.

       .. code-block:: pycon

          >>> @dr.freeze(warn_after=1)
          >>> def f(x):
          ...     return x
          ...
          >>> f(Int(1))
          >>> f(Float(1))
          The frozen function has been recorded 2 times, this indicates a problem
          with how the frozen function is being called. For example, calling it
          with changing python values such as an index. For more information about
          which variables changed set the log level to ``LogLevel::Debug``.

    - **Limiting memory usage**. Storing kernels for many possible input
      configuration requires device memory, which can become problematic. Set the
      ``limit=`` parameter to enable a LRU cache. This is useful when calls to a
      function are mostly compatible but require occasional re-tracing.

    Args:
        limit (Optional[int]): An optional integer specifying the maximum number of
          stored configurations. Once this limit is reached, incompatible calls
          requiring re-tracing will cause the last used configuration to be dropped.

        warn_after (int): When the number of re-tracing steps exceeds this value,
          Dr.Jit will generate a warning that explains which variables changed
          between calls to the function.

        state_fn (Optional[Callable]): This optional callable can specify additional
          state to identifies the configuration. ``state_fn`` will be called with
          the same arguments as that of the decorated function. It should return a
          traversable object (e.g., a list or tuple) that is conceptually treated
          as if it was another input of the function.

        backend (Optional[JitBackend]): If no inputs are given when calling the
          frozen function, the backend used has to be specified using this argument.
          It must match the backend used for computation within the function.

        auto_opaque (bool): If this flag is set true and only literal values
          or their size changes between calls to the function, these variables
          will be marked and made opaque. This reduces the memory usage, traversal
          overhead, and can improve the performance of generated kernels.
          If the flag is set to false, all input variables will be made opaque.

        enabled (bool): If this flag is set to false, the function will not be
          frozen, and the call will be forwarded to the inner function.
    """

    limit = limit if limit is not None else -1
    backend = backend if backend is not None else JitBackend.Invalid

    def decorator(f):
        """
        Internal decorator, returned in ``dr.freeze`` was used with arguments.
        """
        import functools
        import inspect

        def inner(input: dict):
            """
            This inner function is the one that is actually frozen, and it calls
            the wrapped function. It receives the input such as args, kwargs and
            any additional input such as closures or state specified with the ``state``
            lambda, and makes its traversal possible.
            """
            args = input["args"]
            kwargs = input["kwargs"]
            return f(*args, **kwargs)

        class FrozenFunction:
            # If this bool is true, the function will be frozen, otherwise the
            # call will be forwarded to the inner function.
            enabled: bool

            def __init__(self, f) -> None:
                self.f = f
                self.frozen = detail.FrozenFunction(
                    inner, limit, warn_after, backend, auto_opaque
                )
                self.enabled = enabled

            def __call__(self, *args, **kwargs):
                if not self.enabled:
                    return self.f(*args, **kwargs)

                # Capture closure variables to detect when nonlocal symbols change.
                closure = inspect.getclosurevars(f)
                input = {
                    "globals": closure.globals,
                    "nonlocals": closure.nonlocals,
                    "args": args,
                    "kwargs": kwargs,
                }
                if state_fn is not None:
                    input["state_fn"] = state_fn(*args, **kwargs)

                return self.frozen(input)

            @property
            def n_recordings(self):
                """
                Represents the number of times the function was recorded. This
                includes occasions where it was recorded due to a dry-run failing.
                It does not necessarily correspond to the number of recordings
                currently cached see ``n_cached_recordings`` for that.
                """
                return self.frozen.n_recordings

            @property
            def n_cached_recordings(self):
                """
                Represents the number of recordings currently cached of the frozen
                function. If a recording fails in dry-run mode, it will not create
                a new recording, but replace the recording that was attemted to be
                replayed. The number of recordings can also be limited with
                the ``max_cache_size`` argument.
                """
                return self.frozen.n_cached_recordings

            def clear(self):
                """
                Clears the recordings of the frozen function, and resets the
                ``n_recordings`` counter. The reference to the function is still
                kept, and the frozen function can be called again to re-trace
                new recordings.
                """
                return self.frozen.clear()

            def __get__(self, obj, type=None):
                if obj is None:
                    return self
                else:
                    return FrozenMethod(self.f, self.frozen, obj)

        class FrozenMethod(FrozenFunction):
            """
            A FrozenMethod currying the object into the __call__ method.

            If the ``freeze`` decorator is applied to a method of some class, it has
            to call the internal frozen function with the ``self`` argument. To this
            end we implement the ``__get__`` method of the frozen function, to
            return a ``FrozenMethod``, which holds a reference to the object.
            The ``__call__`` method of the ``FrozenMethod`` then supplies the object
            in addition to the arguments to the internal function.
            """
            def __init__(self, f, frozen, obj) -> None:
                self.f = f
                self.obj = obj
                self.frozen = frozen
                self.enabled = enabled

            def __call__(self, *args, **kwargs):
                if not self.enabled:
                    return self.f(self.obj, *args, **kwargs)

                # Capture closure variables to detect when nonlocal symbols change.
                closure = inspect.getclosurevars(self.f)
                input = {
                        "globals": closure.globals,
                        "nonlocals": closure.nonlocals,
                        "args": [self.obj, *args],
                        "kwargs": kwargs,
                    }
                if state_fn is not None:
                    input["state_fn"] = state_fn(self.obj, *args, **kwargs)
                return self.frozen(input)

        return functools.wraps(f)(FrozenFunction(f))

    if f is not None:
        return decorator(f)
    else:
        return decorator


del F
del F2

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

_clip = clip

def srgb_to_linear(x: ArrayT, clip: bool = True) -> ArrayT:
    """
    Convert sRGB gamma-corrected values to linear intensity values.

    Applies the inverse sRGB transfer function (gamma expansion) to convert
    gamma-encoded sRGB values to linear RGB values. The sRGB transfer function
    uses a piecewise curve with a linear segment near black for numerical stability.

    When `clip=True` (default), input values are clamped to [0, 1] before conversion.
    When `clip=False`, the transformation preserves the sign and applies the sRGB
    curve to the absolute value, enabling round-trip conversion of out-of-gamut
    colors. This is useful for wide-gamut workflows where colors may temporarily
    exceed the [0, 1] range during processing.

    The transformation is defined as:

    .. math::

        f(x) = \\begin{cases}
            \\frac{x}{12.92} & \\text{if } x \\leq 0.04045 \\\\
            \\left(\\frac{x + 0.055}{1.055}\\right)^{2.4} & \\text{if } x > 0.04045
        \\end{cases}

    When ``clip=False``, the transformation becomes:

    .. math::

        f(x) = \\text{sign}(x) \\cdot f(|x|)

    Args:
        x (ArrayT): Dr.Jit array containing sRGB gamma-corrected values.
                   Typically in range [0, 1] for in-gamut colors.
        clip (bool): If True, clamp input values to [0, 1]. If False, preserve
                    sign and apply transformation to absolute values. Default: True.

    Returns:
        ArrayT: Linear intensity values. When clip=True, output is in [0, 1].
                When clip=False, output preserves the sign of input values.
    """

    if clip:
        x = _clip(x, 0, 1)
        s = 1
    else:
        s = sign(x)
        x = abs(x)


    return s * select(
        x < 0.04045,
        x / 12.92,
        fma(x, 1 / 1.055, 0.055 / 1.055) ** 2.4
    )

def linear_to_srgb(x: ArrayT, clip: bool = True) -> ArrayT:
    """
    Convert linear intensity values to sRGB gamma-corrected values.

    Applies the sRGB transfer function (gamma compression) to convert linear RGB
    values to gamma-encoded sRGB values suitable for display or storage. The sRGB
    transfer function uses a piecewise curve with a linear segment near black for
    numerical stability.

    When `clip=True` (default), input values are clamped to [0, 1] before conversion.
    When `clip=False`, the transformation preserves the sign and applies the sRGB
    curve to the absolute value, enabling handling of out-of-gamut colors in
    wide-gamut workflows.

    The transformation is defined as:

    .. math::

        f(x) = \\begin{cases}
            12.92 \\cdot x & \\text{if } x \\leq 0.0031308 \\\\
            1.055 \\cdot x^{1/2.4} - 0.055 & \\text{if } x > 0.0031308
        \\end{cases}

    When ``clip=False``, the transformation becomes:

    .. math::

        f(x) = \\text{sign}(x) \\cdot f(|x|)

    Args:
        x (ArrayT): Dr.Jit array containing linear intensity values.
                   Typically in range [0, 1] for in-gamut colors.
        clip (bool): If True, clamp input values to [0, 1]. If False, preserve
                    sign and apply transformation to absolute values. Default: True.

    Returns:
        ArrayT: sRGB gamma-corrected values. When clip=True, output is in [0, 1].
                When clip=False, output preserves the sign of input values.
    """

    if clip:
        x = _clip(x, 0, 1)
        s = 1
    else:
        s = sign(x)
        x = abs(x)

    return s * select(
        x < 0.0031308,
        x * 12.92,
        fma(1.055, x ** (1.0 / 2.4), -0.055)
    )

_SRGB_TO_LMS = [
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
]

_CBRT_LMS_TO_OKLAB = [
    0.2104542553,  0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, +0.4505937099,
    0.0259040371,  0.7827717662, -0.8086757660,
]

_OKLAB_TO_CBRT_LMS = [
    1.0,  0.3963377774,  0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
]

_LMS_TO_SRGB = [
     4.0767416621, -3.3077115913,  0.2309699292,
    -1.2684380046,  2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147,  1.7076147010
]

def linear_srgb_to_oklab(value: ArrayT) -> ArrayT:
    """
    Convert colors from linear sRGB to Oklab color space.

    Oklab is a perceptual color space designed for image processing that provides
    better perceptual uniformity than CIELAB or HSV. The L, a, b coordinates are
    perceptually orthogonal, enabling independent manipulation of lightness, green-red
    axis, and blue-yellow axis without perceived changes in the other dimensions.
    Oklab produces smooth color transitions and accurately predicts human perception
    of hue and chroma.

    For more details, see: https://bottosson.github.io/posts/oklab/

    This function supports Dr.Jit tensors and arrays with RGB or RGBA data.
    For tensors, the color channels must be in the trailing dimension.
    For arrays, the color channels must be in the leading dimension.
    Alpha channels are preserved unchanged in RGBA inputs.

    The function expects **linear sRGB** values as input, not gamma-encoded
    sRGB. For typical gamma-encoded image data (e.g., from image files), first
    apply :py:func:`dr.srgb_to_linear() <drjit.srgb_to_linear>` to convert to
    linear sRGB before using this function.

    Args:
        value (dr.ArrayBase): Dr.Jit tensor or array containing **linear sRGB** colors.
                              For tensors: shape [..., 3] for RGB or [..., 4] for RGBA.
                              For arrays: shape [3, ...] for RGB or [4, ...] for RGBA.

    Returns:
        dr.ArrayBase: Colors converted to Oklab space with same shape as input.
                      L represents lightness, a represents green-red axis,
                      b represents blue-yellow axis.
    """

    if not is_array_v(value):
        raise Exception('linear_srgb_to_oklab(): expected a Jit tensor/array type as input!')

    Type = type(value)
    shape = value.shape

    if is_tensor_v(value):
        if shape[-1] != 3 and shape[-1] != 4:
            raise Exception('linear_srgb_to_oklab(): tensors must have trailing dimension 3 (RGB) or 4 (RGBA)')
        value = reshape(value, (-1, shape[-1]))
        Array = replace_shape_t(Type, (shape[-1], -1), 'array')
        value_2 = Array(value, flip_axes=True)
        value_2 = linear_srgb_to_oklab(value_2)
        return reshape(Type(value_2, flip_axes = True), shape)
    elif shape[0] == 4:
        value_2 = Type()
        value_2.xyz = linear_srgb_to_oklab(value.xyz)
        value_2.w = value.w
        return value_2
    elif shape[0] != 3:
        raise Exception('linear_srgb_to_oklab(): arrays must have leading dimension 3 (RGB) or 4 (RGBA)')

    Matrix = replace_shape_t(Type, (3, 3), 'matrix')
    lms = Matrix(_SRGB_TO_LMS) @ value
    return Matrix(_CBRT_LMS_TO_OKLAB) @ cbrt(lms)

def oklab_to_linear_srgb(value: ArrayT) -> ArrayT:
    """
    Convert colors from Oklab color space back to linear sRGB.

    This function performs the inverse transformation of linear_srgb_to_oklab,
    converting Oklab L, a, b coordinates back to linear sRGB values.

    This function supports Dr.Jit tensors and arrays with Lab or LabA data.
    For tensors, the color channels must be in the trailing dimension.
    For arrays, the color channels must be in the leading dimension.
    Alpha channels are preserved unchanged in LabA inputs.

    This function returns **linear sRGB** values, not gamma-encoded sRGB. To
    obtain gamma-encoded sRGB for typical image output (e.g., for display or
    saving to image files), apply :py:func:`dr.linear_to_srgb()
    <drjit.linear_to_srgb>` to the output of this function.

    Args:
        value (dr.ArrayBase): Dr.Jit tensor or array containing Oklab colors.
                              For tensors: shape [..., 3] for Lab or [..., 4] for LabA.
                              For arrays: shape [3, ...] for Lab or [4, ...] for LabA.

    Returns:
        dr.ArrayBase: Colors converted to **linear sRGB** space with same shape as input.
                      Values are in linear sRGB (not gamma-corrected).
    """

    if not is_array_v(value):
        raise Exception('oklab_to_linear_srgb(): expected a Jit tensor/array type as input!')

    Type = type(value)
    shape = value.shape

    if is_tensor_v(value):
        if shape[-1] != 3 and shape[-1] != 4:
            raise Exception('oklab_to_linear_srgb(): tensors must have trailing dimension 3 (Lab) or 4 (LabA)')
        value = reshape(value, (-1, shape[-1]))
        Array = replace_shape_t(Type, (shape[-1], -1), 'array')
        value_2 = Array(value, flip_axes=True)
        value_2 = oklab_to_linear_srgb(value_2)
        return reshape(Type(value_2, flip_axes=True), shape)
    elif shape[0] == 4:
        value_2 = Type()
        value_2.xyz = oklab_to_linear_srgb(value.xyz)
        value_2.w = value.w
        return value_2
    elif shape[0] != 3:
        raise Exception('oklab_to_linear_srgb(): arrays must have leading dimension 3 (Lab) or 4 (LabA)')

    Matrix = replace_shape_t(Type, (3, 3), 'matrix')

    lms = (Matrix(_OKLAB_TO_CBRT_LMS) @ value)**3
    return Matrix(_LMS_TO_SRGB) @ lms

def unit_angle(a, b):
    """
    Numerically well-behaved routine for computing the angle between two
    normalized 3D direction vectors

    This should be used wherever one is tempted to compute the angle via
    ``acos(dot(a, b))``. It yields significantly more accurate results when the
    angle is close to zero.

    By `Don Hatch <http://www.plunk.org/~hatch/rightway.php>`__.
    """

    dot_uv = dot(a, b)
    temp = 2 * asin(.5 * norm(b - mulsign(a, dot_uv)))
    return select(dot_uv >= 0, temp, pi - temp)

def func(
    f: Optional[F] = None,
    *,
    backend: Optional[JitBackend] = None
):
    """
    Decorator that prevents the body of the decorated function from being
    inlined into the caller, emitting it as a separate callable in the
    generated IR instead. This can significantly reduce compilation time for
    large programs where the decorated function is called from multiple sites.

    .. code-block:: python

       @dr.func
       def f(x):
           return x * 3

       @dr.func(backend=dr.JitBackend.LLVM)
       def g(x):
           return x * 3

    The decorated function must be *pure*: its return values should only
    depend on its input arguments. Accessing global Dr.Jit arrays or other
    external state from within the function can lead to errors.

    The backend is automatically inferred from the function arguments on the
    first call. Once detected, it is cached and reused for subsequent calls.
    If the function takes no Dr.Jit array arguments (directly or within a
    :ref:`PyTree <pytrees>`), the backend must be specified explicitly via the
    ``backend`` parameter.

    Args:
        f (Callable): The function to decorate.

        backend (Optional[JitBackend]): Backend to use for the generated
          function. If specified, this always takes priority over automatic
          detection. If not specified, the backend is inferred from the
          arguments on the first call.
    """
    backend = backend if backend is not None else JitBackend.Invalid

    def _detect_backend(*args, **kwargs):
        def check(a):
            b = backend_v(a)
            if b != JitBackend.Invalid:
                return b
            tp = type(a)
            if tp is list or tp is tuple:
                for v in a:
                    b = check(v)
                    if b != JitBackend.Invalid:
                        return b
            elif tp is dict:
                for v in a.values():
                    b = check(v)
                    if b != JitBackend.Invalid:
                        return b
            elif type(getattr(tp, 'DRJIT_STRUCT', None)) is dict:
                for k in tp.DRJIT_STRUCT:
                    b = check(getattr(a, k))
                    if b != JitBackend.Invalid:
                        return b
            elif hasattr(tp, '__dataclass_fields__'):
                for k in tp.__dataclass_fields__:
                    b = check(getattr(a, k))
                    if b != JitBackend.Invalid:
                        return b
            return JitBackend.Invalid

        for a in args:
            b = check(a)
            if b != JitBackend.Invalid:
                return b
        for a in kwargs.values():
            b = check(a)
            if b != JitBackend.Invalid:
                return b
        return JitBackend.Invalid


    def decorator(f):
        cached_backend = JitBackend.Invalid

        @wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal cached_backend
            if backend != JitBackend.Invalid:
                b = backend
            elif cached_backend != JitBackend.Invalid:
                b = cached_backend
            else:
                b = _detect_backend(*args, **kwargs)
                if b == JitBackend.Invalid:
                    raise RuntimeError(
                        "dr.func(): could not detect the backend from the "
                        "input arguments. Please specify it explicitly via "
                        "the 'backend' parameter.")
                cached_backend = b
            m = _sys.modules[f'drjit.{b.name.lower()}']
            tp = m.UInt32
            return switch(tp(0), [f], *args, label=f"{f.__name__}", **kwargs)

        return wrapper

    if f is not None:
        return decorator(f)
    return decorator


newaxis = None

from . import hashgrid as hashgrid
from . import nn as nn

del overload, Optional
