from enoki import Dynamic, Exception
import enoki as _ek


def _check1(a0):
    s0 = len(a0)
    ar = a0.empty_(s0 if a0.Size == Dynamic else 0)
    return (s0, ar)


def _check2(a0, a1):
    """Validate the inputs of a binary generic array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if a0.Size == Dynamic else 0)
    return (s0, s1, ar, sr)


def _check2_inplace(a0, a1):
    """Validate the inputs of a binary generic array operation (in-place)"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if s0 != sr or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    return (s0, s1, sr)


def _check2_bitop(a0, a1):
    """Validate the inputs of a binary bit manipulation array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1) and type(a1) is not a0.MaskType:  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if a0.Size == Dynamic else 0)
    return (s0, s1, ar, sr)


def _check2_bitop_inplace(a0, a1):
    """Validate the inputs of a binary bit manipulation array operation
       (in-place)"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if s0 != sr or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1) and type(a1) is not a0.MaskType:  # noqa
        raise Exception("Type mismatch!")
    return (s0, s1, sr)


def _check2_mask(a0, a1):
    """Validate the inputs of a binary mask-producing array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.MaskType.empty_(sr if a0.Size == Dynamic else 0)
    return (s0, s1, ar, sr)


def _check3(a0, a1, a2):
    """Validate the inputs of a ternary generic array operation"""
    s0, s1, s2 = len(a0), len(a1), len(a2)
    sr = max(s0, s1, s2)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1) or \
       (s2 != sr and s2 != 1):
        raise Exception("Incompatible argument sizes: %i, %i, and %i"
                        % (s0, s1, s2))
    elif type(a0) is not type(a1) or type(a2) is not type(a2):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if a0.Size == Dynamic else 0)
    return (s0, s1, s2, ar, sr)


# -------------------------------------------------------------------
#                        Vertical operations
# -------------------------------------------------------------------


def neg_(a0):
    s0, ar = _check1(a0)
    if not a0.IsArithmetic:
        raise Exception("neg(): requires arithmetic operands!")
    for i in range(s0):
        ar[i] = -a0[i]
    return ar


def not_(a0):
    s0, ar = _check1(a0)
    if a0.IsFloat:
        raise Exception("not(): requires an integral or mask operand!")
    if type(a0.Value) is bool:
        for i in range(s0):
            ar[i] = not a0[i]
    else:
        for i in range(s0):
            ar[i] = ~a0[i]
    return ar


def add_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("add(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] + a1[i if s1 > 1 else 0]
    return ar


def iadd_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("iadd(): requires arithmetic operands!")
    for i in range(sr):
        a0[i] += a1[i if s1 > 1 else 0]
    return a0


def sub_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("sub(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] - a1[i if s1 > 1 else 0]
    return ar


def isub_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("isub(): requires arithmetic operands!")
    for i in range(sr):
        a0[i] -= a1[i if s1 > 1 else 0]
    return a0


def mul_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("mul(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] * a1[i if s1 > 1 else 0]
    return ar


def imul_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("imul(): requires arithmetic operands!")
    for i in range(sr):
        a0[i] *= a1[i if s1 > 1 else 0]
    return a0


def truediv_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Enoki integer arrays.")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] / a1[i if s1 > 1 else 0]
    return ar


def itruediv_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//=\" for "
                        "Enoki integer arrays.")
    for i in range(sr):
        a0[i] /= a1[i if s1 > 1 else 0]
    return a0


def floordiv_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Enoki floating point arrays.")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] // a1[i if s1 > 1 else 0]
    return ar


def ifloordiv_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsIntegral:
        raise Exception("Use the floor division operator \"/=\" for "
                        "Enoki floating point arrays.")
    for i in range(sr):
        a0[i] //= a1[i if s1 > 1 else 0]
    return a0


def mod_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsIntegral:
        raise Exception("mod(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] % a1[i if s1 > 1 else 0]
    return ar


def imod_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsIntegral:
        raise Exception("imod(): requires arithmetic operands!")
    for i in range(sr):
        a0[i] %= a1[i if s1 > 1 else 0]
    return a0


def and_(a0, a1):
    if a0.Depth == 1 and a0.IsFloat:
        a0i = _ek.reinterpret_array(_ek.uint_array_t(type(a0)), a0)
        a1i = _ek.reinterpret_array(_ek.uint_array_t(type(a1)), a1) \
            if a1.IsFloat else a1
        return _ek.reinterpret_array(type(a0), a0i.and_(a1i))
    else:
        s0, s1, ar, sr = _check2_bitop(a0, a1)
        for i in range(sr):
            ar[i] = a0[i if s0 > 1 else 0] & a1[i if s1 > 1 else 0]
        return ar


def iand_(a0, a1):
    s0, s1, sr = _check2_bitop_inplace(a0, a1)
    for i in range(sr):
        a0[i] &= a1[i if s1 > 1 else 0]
    return a0


def or_(a0, a1):
    s0, s1, ar, sr = _check2_bitop(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] | a1[i if s1 > 1 else 0]
    return ar


def ior_(a0, a1):
    s0, s1, sr = _check2_bitop_inplace(a0, a1)
    for i in range(sr):
        a0[i] |= a1[i if s1 > 1 else 0]
    return a0


def xor_(a0, a1):
    s0, s1, ar, sr = _check2_bitop(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] ^ a1[i if s1 > 1 else 0]
    return ar


def ixor_(a0, a1):
    s0, s1, sr = _check2_bitop_inplace(a0, a1)
    for i in range(sr):
        a0[i] ^= a1[i if s1 > 1 else 0]
    return a0


def sl_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsIntegral:
        raise Exception("sl(): requires integral operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] << a1[i if s1 > 1 else 0]
    return ar


def isl_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsIntegral:
        raise Exception("isl(): requires integral operands!")
    for i in range(sr):
        a0[i] <<= a1[i if s1 > 1 else 0]
    return a0


def sr_(a0, a1):
    s0, s1, ar, sr = _check2(a0, a1)
    if not a0.IsIntegral:
        raise Exception("sr(): requires integral operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] >> a1[i if s1 > 1 else 0]
    return ar


def isr_(a0, a1):
    s0, s1, sr = _check2_inplace(a0, a1)
    if not a0.IsIntegral:
        raise Exception("isr(): requires integral operands!")
    for i in range(sr):
        a0[i] >>= a1[i if s1 > 1 else 0]
    return a0


def lt_(a0, a1):
    s0, s1, ar, sr = _check2_mask(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("lt(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] < a1[i if s1 > 1 else 0]
    return ar


def le_(a0, a1):
    s0, s1, ar, sr = _check2_mask(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("le(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] <= a1[i if s1 > 1 else 0]
    return ar


def gt_(a0, a1):
    s0, s1, ar, sr = _check2_mask(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("gt(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] > a1[i if s1 > 1 else 0]
    return ar


def ge_(a0, a1):
    s0, s1, ar, sr = _check2_mask(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("ge(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] >= a1[i if s1 > 1 else 0]
    return ar


def eq_(a0, a1):
    s0, s1, ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = _ek.eq(a0[i if s0 > 1 else 0], a1[i if s1 > 1 else 0])
    return ar


def neq_(a0, a1):
    s0, s1, ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = _ek.neq(a0[i if s0 > 1 else 0], a1[i if s1 > 1 else 0])
    return ar


def sqrt_(a0):
    s0, ar = _check1(a0)
    if not a0.IsFloat:
        raise Exception("sqrt(): requires floating point operands!")
    for i in range(s0):
        ar[i] = _ek.sqrt(a0[i])
    return ar


def abs_(a0):
    s0, ar = _check1(a0)
    if not a0.IsArithmetic:
        raise Exception("abs(): requires arithmetic operands!")
    for i in range(s0):
        ar[i] = _ek.abs(a0[i])
    return ar


def fmadd_(a0, a1, a2):
    s0, s1, s2, ar, sr = _check3(a0, a1, a2)
    if not a0.IsFloat:
        raise Exception("fmadd(): requires floating point operands!")
    for i in range(sr):
        ar[i] = _ek.fmadd(a0[i if s0 > 1 else 0],
                          a1[i if s1 > 1 else 0],
                          a2[i if s2 > 1 else 0])
    return ar

# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def all_(a0):
    size = len(a0)
    if size == 0:
        raise Exception("all(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = value & a0[i]
    return value


def any_(a0):
    size = len(a0)
    if size == 0:
        raise Exception("any(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = value | a0[i]
    return value


def hsum_(a0):
    size = len(a0)
    if size == 0:
        raise Exception("hsum(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = value + a0[i]
    return value


def hprod_(a0):
    size = len(a0)
    if size == 0:
        raise Exception("hprod(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = value * a0[i]
    return value


def hmin_(a0):
    size = len(a0)
    if size == 0:
        raise Exception("hmin(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = _ek.min(value, a0[i])
    return value


def hmax_(a0):
    size = len(a0)
    if size == 0:
        raise Exception("hmax(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = _ek.max(value, a0[i])
    return value
