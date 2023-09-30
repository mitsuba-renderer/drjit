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

    This function wraps :py:func:`drjit.prefix_sum` and is implemented as

    .. code-block:: python

       def cumsum(value):
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


def reverse(value, axis:int=0):
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

# -------------------------------------------------------------------
#               Context manager for setting JIT flags
# -------------------------------------------------------------------

class scoped_set_flag:
    """
    This context manager can be used to selectively enable or disable a JIT
    compilation flag (:py:class:`drjit.JitFlag` member) for a given chunk of
    code.

    An example is shown below:

    .. code-block::

       with dr.scoped_set_flag(dr.JitFlag.VCallOptimize, False):
           # .. code coes here ..
    """
    def __init__(self, flag: JitFlag, value: bool = True):
        self.flag = flag
        self.value = value

    def __enter__(self):
        self.backup = flag(self.flag)
        set_flag(self.flag, self.value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_flag(self.flag, self.backup)

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

    You may also specify any number of Dr.Jit arrays, tensors, or :ref:`Pytrees
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
        *args (tuple): Arbitrary list of Dr.Jit arrays, tuples, or :ref:`Pytrees
          <pytrees>`. Elements of data structures that could not possibly be
          attached to the AD graph (e.g., Python scalars) are ignored.

        when (bool): Optional keyword argument that can be specified to turn the
          context manager into a no-op via ``when=False``. The default value is
          ``when=True``.
    """
    if not when:
        return detail.NoopContextManager()

    array_indices = []
    detail.collect_indices(
        args,
        array_indices
    )

    if len(args) > 0 and len(array_indices) == 0:
        array_indices = [0]

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

    You may also specify any number of Dr.Jit arrays, tensors, or :ref:`Pytrees
    <pytrees>`. In this case, the context manager behaves differently by
    enabling gradient tracking more selectively for the specified variables.

    .. code-block::

       with dr.suspend_grad():
           with dr.resume_grad(x):
               z = x + y  # 'z' will only track gradients arising from 'x'

    The :py:func:`suspend_grad` and :py:func:`resume_grad` context manager can
    be arbitrarily nested and suitably update the set of tracked variables.

    Args:
        *args (tuple): Arbitrary list of Dr.Jit arrays, tuples, or :ref:`Pytrees
          <pytrees>`. Elements of data structures that could not possibly be
          attached to the AD graph (e.g., Python scalars) are ignored.

        when (bool): Optional keyword argument that can be specified to turn the
          context manager into a no-op via ``when=False``. The default value is
          ``when=True``.
    """
    if not when:
        return detail.NoopContextManager()

    array_indices = []
    detail.collect_indices(
        args,
        array_indices
    )

    if len(args) > 0 and len(array_indices) == 0:
        array_indices = [0]

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
        return detail.NoopContextManager()

    return detail.ADContextManager(detail.ADScope.Isolate, [])
