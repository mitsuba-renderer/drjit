import math as _math
import drjit as _dr

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
_epsilon_32           = float.fromhex('0x1p-24')  # noqa
_epsilon_64           = float.fromhex('0x1p-53')  # noqa
_one_minus_epsilon_32 = float.fromhex('0x1.fffffep-1')  # noqa
_one_minus_epsilon_64 = float.fromhex('0x1.fffffffffffffp-1')  # noqa
_recip_overflow_32    = float.fromhex('0x1p-128')  # noqa
_recip_overflow_64    = float.fromhex('0x1p-1024')  # noqa
_smallest_32          = float.fromhex('0x1p-126')  # noqa
_smallest_64          = float.fromhex('0x1p-1022')  # noqa
_largest_32           = float.fromhex('0x1.fffffep+127')  # noqa
_largest_64           = float.fromhex('0x1.fffffffffffffp+1023')  # noqa


def epsilon(t):
    '''
    Returns the machine epsilon.

    The machine epsilon gives an upper bound on the relative approximation
    error due to rounding in floating point arithmetic.

    Args:
        t (type): Python or Dr.Jit type determining whether to consider 32 or 64
            bits floating point precision.

    Returns:
        float: machine epsilon
    '''
    double_precision = t is _dr.float64_array_t(t)
    return _epsilon_64 if double_precision else _epsilon_32


def one_minus_epsilon(t):
    '''
    Returns one minus machine epsilon value.

    Args:
        t (type): Python or Dr.Jit type determining whether to consider 32 or 64
            bits floating point precision.

    Returns:
        float: one minus machine epsilon value
    '''
    double_precision = t is _dr.float64_array_t(t)
    return _one_minus_epsilon_64 if double_precision else _one_minus_epsilon_32


def recip_overflow(t):
    '''
    Returns the reciprocal overflow threshold value.

    Any numbers below this threshold will overflow to infinity when a reciprocal
    is evaluated.

    Args:
        t (type): Python or Dr.Jit type determining whether to consider 32 or 64
            bits floating point precision.

    Returns:
        float: reciprocal overflow threshold value
    '''
    double_precision = t is _dr.float64_array_t(t)
    return _recip_overflow_64 if double_precision else _recip_overflow_32


def smallest(t):
    '''
    Returns the smallest normalized floating point value.

    Args:
        t (type): Python or Dr.Jit type determining whether to consider 32 or 64
            bits floating point precision.

    Returns:
        float: smallest normalized floating point value
    '''
    double_precision = t is _dr.float64_array_t(t)
    return _smallest_64 if double_precision else _smallest_32

def largest(t):
    '''
    Returns the largest normalized floating point value.

    Args:
        t (type): Python or Dr.Jit type determining whether to consider 32 or 64
            bits floating point precision.

    Returns:
        float: largest normalized floating point value
    '''
    double_precision = t is _dr.float64_array_t(t)
    return _largest_64 if double_precision else _largest_32
