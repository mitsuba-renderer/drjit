from . import detail

with detail.scoped_rtld_deepbind():
    from . import drjit_ext

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


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    r'''
    Returns ``True`` if two arrays are element-wise equal within a given error
    tolerance.

    The function considers both absolute and relative error thresholds. Specifically
    **a** and **b** are considered equal if all elements satisfy

    .. math::
        |a - b| \le |b| \cdot \mathrm{rtol} + \mathrm{atol}.

    Args:
        a (object): A Dr.Jit array or other kind of numeric sequence type.
        b (object): A Dr.Jit array or other kind of numeric sequence type.
        rtol (float): A relative error threshold. The default is :math:`10^{-5}`.
        atol (float): An absolute error threshold. The default is :math:`10^{-8}`.
        equal_nan (bool): If **a** and **b** *both* contain a *NaN* (Not a Number) entry,
                          should they be considered equal? The default is ``False``.

    Returns:
        bool: The result of the comparison.
    '''

    if is_array_v(a) or is_array_v(b):
        # No derivative tracking in the following
        a, b = detach(a), detach(b)

        if is_array_v(a):
            diff = a - b
        else:
            diff = b - a

        a = type(diff)(a)
        b = type(diff)(b)

        cond = abs(diff) <= abs(b) * rtol + atol

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
        return abs(a - b) <= abs(b) * rtol + atol
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


def safe_sqrt(arg0, /):
    '''
    Safely evaluate the square root of the provided input avoiding domain errors.

    Negative inputs produce zero-valued output. When differentiated via AD,
    this function also avoids generating infinite derivatives at ``x=0``.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input
    '''
    result = sqrt(maximum(arg0, 0))
    if is_diff_v(arg0) and grad_enabled(arg0):
        alt = sqrt(maximum(arg0, epsilon(arg0)))
        result = replace_grad(result, alt)
    return result


def safe_asin(arg0, /):
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
    result = asin(clip(arg0, -1, 1))
    if is_diff_v(arg0) and grad_enabled(arg0):
        alt = asin(clip(arg0, -one_minus_epsilon(arg0), one_minus_epsilon(arg0)))
        result = replace_grad(result, alt)
    return result


def safe_acos(arg0, /):
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
    result = acos(clip(arg0, -1, 1))
    if is_diff_v(arg0) and grad_enabled(arg0):
        alt = acos(clip(arg0, -one_minus_epsilon(arg0), one_minus_epsilon(arg0)))
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

# --- Deprecated wrappers for old Dr.Jit reduction operations ----

def all_nested(a):
    import warnings
    warnings.warn("all_nested() is deprecated, please use all(a, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return all(a, axis=None)


def any_nested(a):
    import warnings
    warnings.warn("any_nested() is deprecated, please use any(a, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return any(a, axis=None)


def sum_nested(a):
    import warnings
    warnings.warn("sum_nested() is deprecated, please use sum(a, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return sum(a, axis=None)


def prod_nested(a):
    import warnings
    warnings.warn("prod_nested() is deprecated, please use prod(a, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return prod(a, axis=None)


def min_nested(a):
    import warnings
    warnings.warn("min_nested() is deprecated, please use min(a, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return min(a, axis=None)


def max_nested(a):
    import warnings
    warnings.warn("max_nested() is deprecated, please use max(a, axis=None)",
                  DeprecationWarning, stacklevel=2)
    return max(a, axis=None)


def cumsum(value):
    '''
    Compute an cumulative sum (aka. inclusive prefix sum) of the input array.

    This function wraps :cpp:func:`drjit.prefix_sum` and is implemented as

    .. code-block:: python

       return prefix_sum(value, exclusive=False)
    '''
    return prefix_sum(value, exclusive=False)


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

def epsilon(arg0, /):
    '''
    Returns the machine epsilon.

    The machine epsilon gives an upper bound on the relative approximation
    error due to rounding in floating point arithmetic.

    Args:
        arg0 (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The machine epsilon.
    '''
    vt = var_type_v(arg0)
    if vt == VarType.Float64:
        return float.fromhex('0x1p-53')
    elif vt == VarType.Float32:
        return float.fromhex('0x1p-24')
    elif vt == VarType.Float16:
        return float.fromhex('0x1p-11')
    else:
        raise TypeError("epsilon(): input is not a Dr.Jit array or array type!")


def one_minus_epsilon(arg0, /):
    '''
    Returns one minus the machine epsilon value.

    Args:
        arg0 (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: One minus the machine epsilon.
    '''
    vt = var_type_v(arg0)
    if vt == VarType.Float64:
        return float.fromhex('0x1.fffffffffffffp-1')
    elif vt == VarType.Float32:
        return float.fromhex('0x1.fffffep-1')
    elif vt == VarType.Float16:
        return float.fromhex('0x1.ffcp-1')
    else:
        raise TypeError("one_minus_epsilon(): input is not a Dr.Jit array or array type!")


def recip_overflow(arg0, /):
    '''
    Returns the reciprocal overflow threshold value.

    Any numbers equal to this threshold or a smaller value will overflow to
    infinity when reciprocated.

    Args:
        arg0 (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The reciprocal overflow threshold value.
    '''
    vt = var_type_v(arg0)
    if vt == VarType.Float64:
        return float.fromhex('0x1p-1024')
    elif vt == VarType.Float32:
        return float.fromhex('0x1p-128')
    elif vt == VarType.Float16:
        return float.fromhex('0x1p-16')
    else:
        raise TypeError("recip_overflow(): input is not a Dr.Jit array or array type!")


def smallest(arg0, /):
    '''
    Returns the smallest representable normalized floating point value.

    Args:
        arg0 (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The smallest representable normalized floating point value.
    '''
    vt = var_type_v(arg0)
    if vt == VarType.Float64:
        return float.fromhex('0x1p-1022')
    elif vt == VarType.Float32:
        return float.fromhex('0x1p-126')
    elif vt == VarType.Float16:
        return float.fromhex('0x1p-14')
    else:
        raise TypeError("smallest(): input is not a Dr.Jit array or array type!")

def largest(arg0, /):
    '''
    Returns the largest representable finite floating point value for `t`.

    Args:
        arg0 (object): Dr.Jit array or array type used to choose between
          an appropriate constant for half, single, or double precision.

    Returns:
        float: The largest representable finite floating point value.
    '''
    vt = var_type_v(arg0)
    if vt == VarType.Float64:
        return float.fromhex('0x1.fffffffffffffp+1023')
    elif vt == VarType.Float32:
        return float.fromhex('0x1.fffffep+127')
    elif vt == VarType.Float16:
        return float.fromhex('0x1.ffcp+15')
    else:
        raise TypeError("largest(): input is not a Dr.Jit array or array type!")
