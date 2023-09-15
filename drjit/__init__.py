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
