import sys
import os

if sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

if os.name == 'nt':
    # Specify DLL search path for windows (no rpath on this platform..)
    d = __file__
    for i in range(3):
        d = os.path.dirname(d)
    try: # try to use Python 3.8's DLL handling
        os.add_dll_directory(d)
    except AttributeError:  # otherwise use PATH
        os.environ['PATH'] += os.pathsep + d
    del d, i

del sys, os

# Native extension defining low-level arrays
import drjit.drjit_ext as drjit_ext  # noqa


def sqr(arg, /):
    return arg * arg


def isnan(arg, /):
    result = arg == arg
    if isinstance(result, bool):
        return not result
    else:
        return ~result


def isinf(arg, /):
    return abs(arg) == float('inf')


def isfinite(arg, /):
    return abs(arg) < float('inf')


def all_nested(arg, /):
    """
    Iterates :py:func:`all` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.
    """
    while True:
        arg_t = type(arg)
        arg = all(arg)
        if type(arg) is arg_t:
            break;
    return arg


def any_nested(arg, /):
    """
    Iterates :py:func:`any` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.
    """
    while True:
        arg_t = type(arg)
        arg = any(arg)
        if type(arg) is arg_t:
            break;
    return arg


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    r'''
    Returns ``True`` if two arrays are element-wise equal within a given error
    tolerance.

    The function considers both absolute and relative error thresholds. Specifically
    **a** and **b** are considered equal if all elements satisfy

    .. math::
        | a - b | \le b \cdot \mathrm{rtol} + \mathrm{atol}.

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

        if not is_float_v(a):
            a = float_array_t(type(a))(a)
        if not is_float_v(b):
            b = float_array_t(type(b))(a)

        diff = abs(a - b)
        cond = diff <= abs(b) * rtol + atol

        if equal_nan:
            cond |= isnan(a) & isnan(b)

        return all_nested(cond)

    return False

def detach(a, preserve_type=False):
    if is_diff_v(a):
        if preserve_type:
            return type(a)(a.detach_())
        else:
            return a.detach_()
    elif is_struct_v(a):
        result = type(a)()
        for k in type(a).DRJIT_STRUCT.keys():
            setattr(result, k, detach(getattr(a, k), preserve_type=preserve_type))
        return result
    else:
        return a

