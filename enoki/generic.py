from enoki import Dynamic, Exception
import enoki as _ek


def _check(a0, a1):
    """Validate the inputs of a binary generic array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if s0 == Dynamic else 0)
    return (s0, s1, ar, sr)


def _check_inplace(a0, a1):
    """Validate the inputs of a binary generic array operation (in-place)"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if s0 != sr or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    return (s0, s1, sr)


def _check_bitop(a0, a1):
    """Validate the inputs of a binary bit manipulation array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1) and type(a1) is not a0.MaskType:  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if s0 == Dynamic else 0)
    return (s0, s1, ar, sr)


def _check_bitop_inplace(a0, a1):
    """Validate the inputs of a binary bit manipulation array operation
       (in-place)"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if s0 != sr or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1) and type(a1) is not a0.MaskType:  # noqa
        raise Exception("Type mismatch!")
    return (s0, s1, sr)


def _check_mask(a0, a1):
    """Validate the inputs of a binary mask-producing array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.MaskType.empty_(sr if s0 == Dynamic else 0)
    return (s0, s1, ar, sr)


def add_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("add(): requires an arithmetic type!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] + a1[i if s1 > 1 else 0]
    return ar


def iadd_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("iadd(): requires an arithmetic type!")
    for i in range(sr):
        a0[i] += a1[i if s1 > 1 else 0]
    return a0


def sub_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("sub(): requires an arithmetic type!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] - a1[i if s1 > 1 else 0]
    return ar


def isub_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("isub(): requires an arithmetic type!")
    for i in range(sr):
        a0[i] -= a1[i if s1 > 1 else 0]
    return a0


def mul_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("mul(): requires an arithmetic type!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] * a1[i if s1 > 1 else 0]
    return ar


def imul_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("imul(): requires an arithmetic type!")
    for i in range(sr):
        a0[i] *= a1[i if s1 > 1 else 0]
    return a0


def truediv_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Enoki integer arrays.")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] / a1[i if s1 > 1 else 0]
    return ar


def itruediv_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//=\" for "
                        "Enoki integer arrays.")
    for i in range(sr):
        a0[i] /= a1[i if s1 > 1 else 0]
    return a0


def floordiv_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    if not a0.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Enoki floating point arrays.")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] // a1[i if s1 > 1 else 0]
    return ar


def ifloordiv_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    if not a0.IsIntegral:
        raise Exception("Use the floor division operator \"/=\" for "
                        "Enoki floating point arrays.")
    for i in range(sr):
        a0[i] //= a1[i if s1 > 1 else 0]
    return a0


def mod_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    if not a0.IsIntegral:
        raise Exception("mod(): requires an arithmetic type!")
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] % a1[i if s1 > 1 else 0]
    return ar


def imod_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    if not a0.IsIntegral:
        raise Exception("imod(): requires an arithmetic type!")
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
        s0, s1, ar, sr = _check_bitop(a0, a1)
        for i in range(sr):
            ar[i] = a0[i if s0 > 1 else 0] & a1[i if s1 > 1 else 0]
        return ar


def iand_(a0, a1):
    s0, s1, sr = _check_bitop_inplace(a0, a1)
    for i in range(sr):
        a0[i] &= a1[i if s1 > 1 else 0]
    return a0


def or_(a0, a1):
    s0, s1, ar, sr = _check_bitop(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] | a1[i if s1 > 1 else 0]
    return ar


def ior_(a0, a1):
    s0, s1, sr = _check_bitop_inplace(a0, a1)
    for i in range(sr):
        a0[i] |= a1[i if s1 > 1 else 0]
    return a0


def xor_(a0, a1):
    s0, s1, ar, sr = _check_bitop(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] ^ a1[i if s1 > 1 else 0]
    return ar


def ixor_(a0, a1):
    s0, s1, sr = _check_bitop_inplace(a0, a1)
    for i in range(sr):
        a0[i] ^= a1[i if s1 > 1 else 0]
    return a0


def sl_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] << a1[i if s1 > 1 else 0]
    return ar


def isl_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    for i in range(sr):
        a0[i] <<= a1[i if s1 > 1 else 0]
    return a0


def sr_(a0, a1):
    s0, s1, ar, sr = _check(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] << a1[i if s1 > 1 else 0]
    return ar


def isr_(a0, a1):
    s0, s1, sr = _check_inplace(a0, a1)
    for i in range(sr):
        a0[i] >>= a1[i if s1 > 1 else 0]
    return a0


def lt_(a0, a1):
    s0, s1, ar, sr = _check_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] < a1[i if s1 > 1 else 0]
    return a0


def le_(a0, a1):
    s0, s1, ar, sr = _check_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] <= a1[i if s1 > 1 else 0]
    return a0


def gt_(a0, a1):
    s0, s1, ar, sr = _check_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] > a1[i if s1 > 1 else 0]
    return a0


def ge_(a0, a1):
    s0, s1, ar, sr = _check_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i if s0 > 1 else 0] >= a1[i if s1 > 1 else 0]
    return a0


def eq_(a0, a1):
    s0, s1, ar, sr = _check_mask(a0, a1)
    for i in range(sr):
        ar[i] = _ek.eq(a0[i if s0 > 1 else 0], a1[i if s1 > 1 else 0])
    return a0


def neq_(a0, a1):
    s0, s1, ar, sr = _check_mask(a0, a1)
    for i in range(sr):
        ar[i] = _ek.neq(a0[i if s0 > 1 else 0], a1[i if s1 > 1 else 0])
    return a0
