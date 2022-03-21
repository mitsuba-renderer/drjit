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


def sqr(a):
    return a * a

def isnan(a):
    result = a == a
    if isinstance(result, bool):
        return not result
    else:
        return ~result

def isinf(a):
    return abs(a) == float('inf')

def isfinite(a):
    return abs(a) < float('inf')

#def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
#    Fast path for Dr.Jit arrays, avoid for special array types
#    due to their non-standard broadcasting behavior
#    if _ek.is_array_v(a) or _ek.is_array_v(b):
#        if _ek.is_diff_array_v(a):
#            a = _ek.detach(a)
#        if _ek.is_diff_array_v(b):
#            b = _ek.detach(b)
#
#        if _ek.is_array_v(a) and not _ek.is_floating_point_v(a):
#            a += 0.0
#        if _ek.is_array_v(b) and not _ek.is_floating_point_v(b):
#            b += 0.0
#
#        diff = abs(a - b)
#        a = type(diff)(a)
#        b = type(diff)(b)
#
#        shape = 1
#        if _ek.is_tensor_v(diff):
#            shape = diff.shape
#        cond = diff <= abs(b) * rtol + _ek.full(type(diff), atol, shape)
#        if _ek.is_floating_point_v(a):
#            cond |= _ek.eq(a, b)  # plus/minus infinity
#        if equal_nan:
#            cond |= _ek.isnan(a) & _ek.isnan(b)
#        return _ek.all_nested(cond)
#
#    def safe_len(x):
#        try:
#            return len(x)
#        except TypeError:
#            return 0
#
#    def safe_getitem(x, xl, i):
#        return x[i if xl > 1 else 0] if xl > 0 else x
#
#    la, lb = safe_len(a), safe_len(b)
#    size = max(la, lb)
#
#    if la != size and la > 1 or lb != size and lb > 1:
#        raise Exception("allclose(): size mismatch (%i vs %i)!" % (la, lb))
#    elif size == 0:
#        if equal_nan and _math.isnan(a) and _math.isnan(b):
#            return True
#        return abs(a - b) <= abs(b) * rtol + atol
#    else:
#        for i in range(size):
#            ia = safe_getitem(a, la, i)
#            ib = safe_getitem(b, lb, i)
#            if not allclose(ia, ib, rtol, atol, equal_nan):
#                return False
#        return True
