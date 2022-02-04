import math as _math
import drjit as _drjit

# -------------------------------------------------------------------
#                      Mathematical constants
# -------------------------------------------------------------------

E                  = 2.71828182845904523536  # noqa
LogTwo             = 0.69314718055994530942  # noqa
InvLogTwo          = 1.44269504088896340736  # noqa

Pi                 = 3.14159265358979323846  # noqa
InvPi              = 0.31830988618379067154  # noqa
SqrtPi             = 1.77245385090551602793  # noqa
InvSqrtPi          = 0.56418958354775628695  # noqa

TwoPi              = 6.28318530717958647692  # noqa
InvTwoPi           = 0.15915494309189533577  # noqa
SqrtTwoPi          = 2.50662827463100050242  # noqa
InvSqrtTwoPi       = 0.39894228040143267794  # noqa

FourPi             = 12.5663706143591729539  # noqa
InvFourPi          = 0.07957747154594766788  # noqa
SqrtFourPi         = 3.54490770181103205460  # noqa
InvSqrtFourPi      = 0.28209479177387814347  # noqa

SqrtTwo            = 1.41421356237309504880  # noqa
InvSqrtTwo         = 0.70710678118654752440  # noqa

Infinity           = float('inf')  # noqa
NaN                = float('nan')  # noqa
_Epsilon32         = float.fromhex('0x1p-24')  # noqa
_Epsilon64         = float.fromhex('0x1p-53')  # noqa
_OneMinusEpsilon32 = float.fromhex('0x1.fffffep-1')  # noqa
_OneMinusEpsilon64 = float.fromhex('0x1.fffffffffffffp-1')  # noqa
_RecipOverflow32   = float.fromhex('0x1p-128')  # noqa
_RecipOverflow64   = float.fromhex('0x1p-1024')  # noqa
_Smallest32        = float.fromhex('0x1p-126')  # noqa
_Smallest64        = float.fromhex('0x1p-1022')  # noqa
_Largest32         = float.fromhex('0x1.fffffep+127')  # noqa
_Largest64         = float.fromhex('0x1.fffffffffffffp+1023')  # noqa
_f64               = _drjit.VarType.Float64


def Epsilon(t):
    double_precision = getattr(t, 'Type', _f64) == _f64
    return _Epsilon64 if double_precision else _Epsilon32


def OneMinusEpsilon(t):
    double_precision = getattr(t, 'Type', _f64) == _f64
    return _OneMinusEpsilon64 if double_precision else _OneMinusEpsilon32


def RecipOverflow(t):
    double_precision = getattr(t, 'Type', _f64) == _f64
    return _RecipOverflow64 if double_precision else _RecipOverflow32


def Smallest(t):
    double_precision = getattr(t, 'Type', _f64) == _f64
    return _Smallest64 if double_precision else _Smallest32

def Largest(t):
    double_precision = getattr(t, 'Type', _f64) == _f64
    return _Largest64 if double_precision else _Largest32
