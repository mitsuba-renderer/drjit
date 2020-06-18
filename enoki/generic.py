from enoki import Dynamic, Exception, VarType
import enoki as _ek


def _check1(a0):
    s0 = len(a0)
    ar = a0.empty_(s0 if a0.Size == Dynamic else 0)
    return (ar, s0)


def _check2(a0, a1):
    """Validate the inputs of a binary generic array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if a0.Size == Dynamic else 0)
    return (ar, sr)


def _check2_inplace(a0, a1):
    """Validate the inputs of a binary generic array operation (in-place)"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if s0 != sr or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    return sr


def _check2_bitop(a0, a1):
    """Validate the inputs of a binary bit manipulation array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1) and type(a1) is not a0.MaskType:  # noqa
        raise Exception("Type mismatch!")
    ar = a0.empty_(sr if a0.Size == Dynamic else 0)
    return (ar, sr)


def _check2_bitop_inplace(a0, a1):
    """Validate the inputs of a binary bit manipulation array operation
       (in-place)"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if s0 != sr or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1) and type(a1) is not a0.MaskType:  # noqa
        raise Exception("Type mismatch!")
    return sr


def _check2_mask(a0, a1):
    """Validate the inputs of a binary mask-producing array operation"""
    s0, s1 = len(a0), len(a1)
    sr = max(s0, s1)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1):
        raise Exception("Incompatible argument sizes: %i and %i" % (s0, s1))
    elif type(a0) is not type(a1):  # noqa
        raise Exception("Type mismatch!")
    ar = a0.MaskType.empty_(sr if a0.Size == Dynamic else 0)
    return (ar, sr)


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
    return (ar, sr)


def _check3_select(a0, a1, a2):
    """Validate the inputs of a select() array operation"""
    s0, s1, s2 = len(a0), len(a1), len(a2)
    sr = max(s0, s1, s2)
    if (s0 != sr and s0 != 1) or (s1 != sr and s1 != 1) or \
       (s2 != sr and s2 != 1):
        raise Exception("Incompatible argument sizes: %i, %i, and %i"
                        % (s0, s1, s2))
    elif type(a1) is not type(a2) or type(a0) is not type(a1).MaskType:  # noqa
        raise Exception("Type mismatch!")
    ar = a1.empty_(sr if a0.Size == Dynamic else 0)
    return (ar, sr)


def _binary_op(a, b, fn):
    """
    Perform a bit-level operation 'fn' involving variables 'a' and 'b' in a way
    that works even when the operands are floating point variables
    """

    convert = isinstance(a, float) or isinstance(b, float)

    if convert:
        src, dst = VarType.Float64, VarType.Int64
        a = _ek.detail.reinterpret_scalar(a, src, dst)
        if isinstance(b, bool):
            b = -1 if b else 0
        else:
            b = _ek.detail.reinterpret_scalar(b, src, dst)

    c = fn(a, b)

    if convert:
        c = _ek.detail.reinterpret_scalar(c, dst, src)

    return c

# -------------------------------------------------------------------
#                        Vertical operations
# -------------------------------------------------------------------

def neg_(a0):
    if not a0.IsArithmetic:
        raise Exception("neg(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = -a0[i]
    return ar


def not_(a0):
    if a0.IsFloat:
        raise Exception("not(): requires an integral or mask operand!")
    ar, sr = _check1(a0)
    if type(a0.Value) is bool:
        for i in range(sr):
            ar[i] = not a0[i]
    else:
        for i in range(sr):
            ar[i] = ~a0[i]
    return ar


def add_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("add(): requires arithmetic operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] + a1[i]
    return ar


def iadd_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("iadd(): requires arithmetic operands!")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] += a1[i]
    return a0


def sub_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("sub(): requires arithmetic operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] - a1[i]
    return ar


def isub_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("isub(): requires arithmetic operands!")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] -= a1[i]
    return a0


def mul_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("mul(): requires arithmetic operands!")
    if isinstance(a1, int) or isinstance(a1, float):
        # Avoid type promotion in scalars multiplication, which would
        # be costly for special types (matrices, quaternions, etc.)
        sr = len(a0)
        ar = a0.empty_(sr if a0.Size == Dynamic else 0)
        for i in range(len(a0)):
            ar[i] = a0[i] * a1
        return ar
    elif a0.IsMatrix and a1.IsVector and a0.Size == a1.Size:
        ar = a0[0] * a1[0]
        for i in range(a1.Size):
            ar = _ek.fmadd(a0[i], a1[i], ar)
        return ar

    ar, sr = _check2(a0, a1)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = a0[i] * a1[i]
    elif a0.IsComplex:
        ar.real = _ek.fmsub(a0.real, a1.real, a0.imag * a1.imag)
        ar.imag = _ek.fmadd(a0.real, a1.imag, a0.imag * a1.real)
    elif a0.IsQuaternion:
        tbl = (4, 3, -2, 1, -3, 4, 1, 2, 2, -1, 4, 3, -1, -2, -3, 4)
        for i in range(4):
            accum = 0
            for j in range(4):
                idx = tbl[i*4 + j]
                value = a1[abs(idx) - 1]
                accum = _ek.fmadd(a0[j], value if idx > 0 else -value, accum)
            ar[i] = accum
    elif a0.IsMatrix:
        for j in range(a0.Size):
            accum = a0[0] * _ek.full(a0.Value, a1[0, j])
            for i in range(1, a0.Size):
                accum = _ek.fmadd(a0[i], _ek.full(a0.Value, a1[i, j]), accum)
            ar[j] = accum
    else:
        raise Exception("mul(): unsupported array type!")
    return ar


def imul_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("imul(): requires arithmetic operands!")
    sr = _check2_inplace(a0, a1)
    if not a0.IsSpecial:
        for i in range(sr):
            a0[i] *= a1[i]
    else:
        a0.assign_(a0 * a1)
    return a0


def truediv_(a0, a1):
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Enoki integer arrays.")
    if not a0.IsSpecial:
        ar, sr = _check2(a0, a1)
        for i in range(sr):
            ar[i] = a0[i] / a1[i]
        return ar
    elif a0.IsSpecial:
        return a0 * a1.rcp_()
    else:
        raise Exception("truediv(): unsupported array type!")


def itruediv_(a0, a1):
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//=\" for "
                        "Enoki integer arrays.")
    sr = _check2_inplace(a0, a1)
    if not a0.IsSpecial:
        for i in range(sr):
            a0[i] /= a1[i]
    else:
        a0.assign_(a0 / a1)

    return a0


def floordiv_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Enoki floating point arrays.")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] // a1[i]
    return ar


def ifloordiv_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("Use the floor division operator \"/=\" for "
                        "Enoki floating point arrays.")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] //= a1[i]
    return a0


def mod_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("mod(): requires arithmetic operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] % a1[i]
    return ar


def imod_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("imod(): requires arithmetic operands!")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] %= a1[i]
    return a0


def and_(a0, a1):
    ar, sr = _check2_bitop(a0, a1)

    if ar.Depth > 1:
        for i in range(sr):
            ar[i] = a0[i] & a1[i]
    else:
        for i in range(sr):
            ar[i] = _binary_op(a0[i], a1[i],
                               lambda a, b: a & b)

    return ar


def iand_(a0, a1):
    sr = _check2_bitop_inplace(a0, a1)
    if a0.Depth > 1:
        for i in range(sr):
            a0[i] &= a1[i]
    else:
        for i in range(sr):
            a0[i] = _binary_op(a0[i], a1[i],
                               lambda a, b: a & b)
    return a0


def or_(a0, a1):
    ar, sr = _check2_bitop(a0, a1)

    if ar.Depth > 1:
        for i in range(sr):
            ar[i] = a0[i] | a1[i]
    else:
        for i in range(sr):
            ar[i] = _binary_op(a0[i], a1[i],
                               lambda a, b: a | b)

    return ar


def ior_(a0, a1):
    sr = _check2_bitop_inplace(a0, a1)
    if a0.Depth > 1:
        for i in range(sr):
            a0[i] |= a1[i]
    else:
        for i in range(sr):
            a0[i] = _binary_op(a0[i], a1[i],
                               lambda a, b: a | b)
    return a0


def xor_(a0, a1):
    ar, sr = _check2_bitop(a0, a1)

    if ar.Depth > 1:
        for i in range(sr):
            ar[i] = a0[i] ^ a1[i]
    else:
        for i in range(sr):
            ar[i] = _binary_op(a0[i], a1[i],
                               lambda a, b: a ^ b)

    return ar


def ixor_(a0, a1):
    sr = _check2_bitop_inplace(a0, a1)
    if a0.Depth > 1:
        for i in range(sr):
            a0[i] ^= a1[i]
    else:
        for i in range(sr):
            a0[i] = _binary_op(a0[i], a1[i],
                               lambda a, b: a ^ b)
    return a0


def sl_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("sl(): requires integral operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] << a1[i]
    return ar


def isl_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("isl(): requires integral operands!")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] <<= a1[i]
    return a0


def sr_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("sr(): requires integral operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] >> a1[i]
    return ar


def isr_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("isr(): requires integral operands!")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] >>= a1[i]
    return a0


def lt_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("lt(): requires arithmetic operands!")
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] < a1[i]
    return ar


def le_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("le(): requires arithmetic operands!")
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] <= a1[i]
    return ar


def gt_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("gt(): requires arithmetic operands!")
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] > a1[i]
    return ar


def ge_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("ge(): requires arithmetic operands!")
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] >= a1[i]
    return ar


def eq_(a0, a1):
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = _ek.eq(a0[i], a1[i])
    return ar


def neq_(a0, a1):
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = _ek.neq(a0[i], a1[i])
    return ar


def sqrt_(a0):
    if not a0.IsFloat:
        raise Exception("sqrt(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.sqrt(a0[i])
    elif a0.IsComplex:
        n = abs(a0)
        m = a0.real >= 0
        zero = _ek.eq(n, 0)
        t1 = _ek.sqrt(.5 * (n + abs(a0.real)))
        t2 = .5 * a0.imag / t1
        im = _ek.select(m, t2, _ek.copysign(t1, a0.imag))
        ar.real = _ek.select(m, t1, abs(t2))
        ar.imag = _ek.select(zero, 0, im)
    elif a0.IsQuaternion:
        ri = _ek.norm(a0.imag)
        cs = _ek.sqrt(a0.Complex(a0.real, ri))
        ar.imag = a0.imag * (_ek.rcp(ri) * cs.imag)
        ar.real = cs.real
    else:
        raise Exception("sqrt(): unsupported array type!")
    return ar


def rsqrt_(a0):
    if not a0.IsFloat:
        raise Exception("rsqrt(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.rsqrt(a0[i])
        return ar
    else:
        return _ek.rcp(_ek.sqrt(a0))


def rcp_(a0):
    if not a0.IsFloat:
        raise Exception("rcp(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.rcp(a0[i])
    elif a0.IsComplex or a0.IsQuaternion:
        return _ek.conj(a0) * _ek.rcp(_ek.squared_norm(a0))
    else:
        raise Exception('rcp(): unsupported array type!')
    return ar


def abs_(a0):
    if not a0.IsArithmetic:
        raise Exception("abs(): requires arithmetic operands!")
    if not a0.IsSpecial or a0.IsMatrix:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.abs(a0[i])
        return ar
    elif a0.IsSpecial:
        return _ek.norm(a0)
    else:
        raise Exception('abs(): unsupported array type!')


def mulhi_(a0, a1):
    ar, sr = _check2(a0, a1)
    if not a0.IsIntegral:
        raise Exception("mulhi(): requires integral operands!")
    for i in range(sr):
        ar[i] = _ek.mulhi(a0[i], a1[i])
    return ar


def max_(a0, a1):
    ar, sr = _check2(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("max(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = _ek.max(a0[i], a1[i])
    return ar


def min_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("min(): requires arithmetic operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = _ek.min(a0[i], a1[i])
    return ar


def fmadd_(a0, a1, a2):
    if not a0.IsFloat:
        raise Exception("fmadd(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check3(a0, a1, a2)
        for i in range(sr):
            ar[i] = _ek.fmadd(a0[i], a1[i], a2[i])
        return ar
    else:
        return a0 * a1 + a2


@staticmethod
def select_(a0, a1, a2):
    ar, sr = _check3_select(a0, a1, a2)
    for i in range(sr):
        ar[i] = _ek.select(a0[i], a1[i], a2[i])
    return ar

# -------------------------------------------------------------------
#       Vertical operations -- ad/JIT compilation-related
# -------------------------------------------------------------------


def label_(a):
    if a.IsJIT or a.IsDiff:
        return [v.label_() for v in a]
    return None


def set_label_(a, label):
    if a.IsJIT or a.IsDiff:
        if isinstance(label, tuple) or isinstance(label, list):
            if len(a) != len(label):
                raise Exception("Size mismatch!")
            for i, v in enumerate(a):
                v.set_label_(label[i])
        else:
            for i, v in enumerate(a):
                v.set_label_(label + "_%i" % i)


@property
def label(a):
    return a.label_()


@label.setter
def label(a, value):
    return a.set_label_(value)


def schedule(a0):
    if a0.IsJIT:
        if a0.Depth == 1:
            if a0.IsDiff:
                a0 = a0.detach_()
            _ek.detail.schedule(a0.index_())
        else:
            for i in range(len(a0)):
                a0[i].schedule()
    return a0


def eval(a0):
    if a0.IsJIT:
        schedule(a0)
        _ek.detail.eval()
    return a0

# -------------------------------------------------------------------
#           Vertical operations -- transcendental functions
# -------------------------------------------------------------------


def sin_(a0):
    if not a0.IsFloat:
        raise Exception("sin(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.sin(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.real)
        sh, ch = _ek.sincosh(a0.imag)
        ar.real = s * ch
        ar.imag = c * sh
    else:
        raise Exception("sin(): unsupported array type!")
    return ar


def cos_(a0):
    if not a0.IsFloat:
        raise Exception("cos(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.cos(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.real)
        sh, ch = _ek.sincosh(a0.imag)
        ar.real = c * ch
        ar.imag = -s * sh
    else:
        raise Exception("cos(): unsupported array type!")

    return ar


def sincos_(a0):
    if not a0.IsFloat:
        raise Exception("sincos(): requires floating point operands!")
    ar0, sr0 = _check1(a0)
    ar1 = a0.empty_(sr0 if a0.Size == Dynamic else 0)
    if not a0.IsSpecial:
        for i in range(sr0):
            result = _ek.sincos(a0[i])
            ar0[i] = result[0]
            ar1[i] = result[1]
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.real)
        sh, ch = _ek.sincosh(a0.imag)
        ar0.real = s * ch
        ar0.imag = c * sh
        ar1.real = c * ch
        ar1.imag = -s * sh
    else:
        raise Exception("sincos(): unsupported array type!")
    return ar0, ar1


def csc_(a0):
    if not a0.IsFloat:
        raise Exception("csc(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.csc(a0[i])
    else:
        return 1 / _ek.sin(a0)
    return ar


def sec_(a0):
    if not a0.IsFloat:
        raise Exception("sec(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.sec(a0[i])
    else:
        return 1 / _ek.cos(a0)
    return ar


def tan_(a0):
    if not a0.IsFloat:
        raise Exception("tan(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.tan(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0)
        return s / c
    else:
        raise Exception("tan(): unsupported array type!")
    return ar


def cot_(a0):
    if not a0.IsFloat:
        raise Exception("cot(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.cot(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0)
        return c / s
    else:
        raise Exception("cot(): unsupported array type!")
    return ar


def asin_(a0):
    if not a0.IsFloat:
        raise Exception("asin(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.asin(a0[i])
    elif a0.IsSpecial:
        tmp = _ek.log(type(a0)(-a0.imag, a0.real) + _ek.sqrt(1 - _ek.sqr(a0)))
        ar.real = tmp.imag
        ar.imag = -tmp.real
    else:
        raise Exception("asin(): unsupported array type!")
    return ar


def acos_(a0):
    if not a0.IsFloat:
        raise Exception("acos(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.acos(a0[i])
    elif a0.IsSpecial:
        tmp = _ek.sqrt(1 - _ek.sqr(a0))
        tmp = _ek.log(a0 + type(a0)(-tmp.imag, tmp.real))
        ar.real = tmp.imag
        ar.imag = -tmp.real
    else:
        raise Exception("acos(): unsupported array type!")
    return ar


def atan_(a0):
    if not a0.IsFloat:
        raise Exception("atan(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.atan(a0[i])
    elif a0.IsSpecial:
        im = type(a0)(0, 1)
        tmp = _ek.log((im - a0) / (im + a0))
        return type(a0)(tmp.imag * .5, -tmp.real * 0.5)
    else:
        raise Exception("atan(): unsupported array type!")
    return ar


def atan2_(a0, a1):
    if not a0.IsFloat:
        raise Exception("atan2(): requires floating point operands!")
    ar, sr = _check2(a0, a1)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.atan2(a0[i], a1[i])
    else:
        raise Exception("atan2(): unsupported array type!")
    return ar


def exp_(a0):
    if not a0.IsFloat:
        raise Exception("exp(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.exp(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.imag)
        exp_r = _ek.exp(a0.real)
        ar.real = exp_r * c
        ar.imag = exp_r * s
    elif a0.IsQuaternion:
        qi = a0.imag
        ri = _ek.norm(qi)
        exp_w = _ek.exp(a0.real)
        s, c = _ek.sincos(ri)
        ar.imag = qi * (s * exp_w / ri)
        ar.real = c * exp_w
    else:
        raise Exception("exp(): unsupported array type!")
    return ar


def exp2_(a0):
    if not a0.IsFloat:
        raise Exception("exp2(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.exp2(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.imag * _ek.LogTwo)
        exp_r = _ek.exp2(a0.real)
        ar.real = exp_r * c
        ar.imag = exp_r * s
    else:
        raise Exception("exp2(): unsupported array type!")
    return ar


def log_(a0):
    if not a0.IsFloat:
        raise Exception("log(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.log(a0[i])
    elif a0.IsComplex:
        ar.real = .5 * _ek.log(_ek.squared_norm(a0))
        ar.imag = _ek.arg(a0)
    elif a0.IsQuaternion:
        qi_n = _ek.normalize(a0.imag)
        rq = _ek.norm(a0)
        acos_rq = _ek.acos(a0.real / rq)
        log_rq = _ek.log(rq)
        ar.imag = qi_n * acos_rq
        ar.real = log_rq
    else:
        raise Exception("log(): unsupported array type!")
    return ar


def log2_(a0):
    if not a0.IsFloat:
        raise Exception("log2(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.log2(a0[i])
    elif a0.IsComplex:
        ar.real = .5 * _ek.log2(_ek.squared_norm(a0))
        ar.imag = _ek.arg(a0) * _ek.InvLogTwo
    else:
        raise Exception("log2(): unsupported array type!")
    return ar


def pow_(a0, a1):
    if not a0.IsFloat:
        raise Exception("pow(): requires floating point operands!")
    if not a0.IsSpecial:
        if isinstance(a1, int) or isinstance(a1, float):
            ar, sr = _check1(a0)
            for i in range(sr):
                ar[i] = _ek.pow(a0[i], a1)
        else:
            ar, sr = _check2(a0, a1)
            for i in range(sr):
                ar[i] = _ek.pow(a0[i], a1[i])
    else:
        return _ek.exp(_ek.log(a0) * a1)
    return ar


def sinh_(a0):
    if not a0.IsFloat:
        raise Exception("sinh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.sinh(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.imag)
        sh, ch = _ek.sincosh(a0.real)
        ar.real = sh * c
        ar.imag = ch * s
    else:
        raise Exception("sinh(): unsupported array type!")
    return ar


def cosh_(a0):
    if not a0.IsFloat:
        raise Exception("cosh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.cosh(a0[i])
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.imag)
        sh, ch = _ek.sincosh(a0.real)
        ar.real = ch * c
        ar.imag = sh * s
    else:
        raise Exception("cosh(): unsupported array type!")
    return ar


def sincosh_(a0):
    if not a0.IsFloat:
        raise Exception("sincosh(): requires floating point operands!")
    ar0, sr0 = _check1(a0)
    ar1 = a0.empty_(sr0 if a0.Size == Dynamic else 0)
    if not a0.IsSpecial:
        for i in range(sr0):
            result = _ek.sincosh(a0[i])
            ar0[i] = result[0]
            ar1[i] = result[1]
    elif a0.IsComplex:
        s, c = _ek.sincos(a0.imag)
        sh, ch = _ek.sincosh(a0.real)
        ar0.real = sh * c
        ar0.imag = ch * s
        ar1.real = ch * c
        ar1.imag = sh * s
    else:
        raise Exception("sincosh(): unsupported array type!")
    return ar0, ar1


def asinh_(a0):
    if not a0.IsFloat:
        raise Exception("asinh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.asinh(a0[i])
    elif a0.IsComplex:
        return _ek.log(a0 + _ek.sqrt(_ek.sqr(a0) + 1))
    else:
        raise Exception("asinh(): unsupported array type!")
    return ar


def acosh_(a0):
    if not a0.IsFloat:
        raise Exception("acosh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.acosh(a0[i])
    elif a0.IsComplex:
        return 2 * _ek.log(_ek.sqrt(.5 * (a0 + 1)) + _ek.sqrt(.5 * (a0 - 1)))
    else:
        raise Exception("acosh(): unsupported array type!")
    return ar


def atanh_(a0):
    if not a0.IsFloat:
        raise Exception("atanh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.atanh(a0[i])
    elif a0.IsComplex:
        return _ek.log((1 + a0) / (1 - a0)) * .5
    else:
        raise Exception("atanh(): unsupported array type!")
    return ar


def cbrt_(a0):
    if not a0.IsFloat:
        raise Exception("cbrt(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _ek.cbrt(a0[i])
    else:
        raise Exception("cbrt(): unsupported array type!")
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


def dot_(a0, a1):
    size = len(a0)
    if size == 0:
        raise Exception("dot(): zero-sized array!")
    if size != len(a1):
        raise Exception("dot(): incompatible array sizes!")
    if type(a0) is not type(a1):
        raise Exception("Type mismatch!")

    value = a0[0] * a1[0]
    if a0.IsFloat:
        for i in range(1, size):
            value = _ek.fmadd(a0[i], a1[i], value)
    else:
        for i in range(1, size):
            value += a0[i] * a1[i]
    return value


# -------------------------------------------------------------------
#                     Automatic differentiation
# -------------------------------------------------------------------


def detach_(a):
    if not a.IsDiff:
        return a

    t = _ek.nondiff_array_t(type(a))
    result = t.empty_(len(a) if a.Size == Dynamic else 0)
    for i in range(len(a)):
        result[i] = a[i].detach_()
    return result


def grad_(a):
    if not a.IsDiff:
        return None

    t = _ek.nondiff_array_t(type(a))
    result = t.empty_(len(a) if a.Size == Dynamic else 0)
    for i in range(len(a)):
        g = a[i].grad_()
        if g is None:
            return None
        result[i] = g
    return result


def set_grad_enabled_(a, value):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")
    for i in range(len(a)):
        a[i] = a[i].set_grad_enabled_(value)


def set_grad_(a, grad):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")

    s = len(a)
    for i in range(s):
        a[i].set_grad_(grad[i])


def enqueue_(a):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")
    for i in range(len(a)):
        a[i].enqueue_()


def migrate_(a, type_):
    if not a.IsJIT:
        raise Exception("Expected a JIT array type!")
    for i in range(len(a)):
        a[i] = a[i].migrate_(type_)


def index_(a):
    if not a.IsJIT:
        raise Exception("Expected a JIT array type!")
    return tuple(v.index_() for v in a)


# -------------------------------------------------------------------
#                      Initialization operations
# -------------------------------------------------------------------

def assign_(self, other):
    if self is other:
        return
    elif len(self) != len(other):
        raise Exception("assign_(): size mismatch!")
    else:
        for i in range(len(self)):
            self[i] = other[i]


def broadcast_(self, value):
    if not self.IsSpecial:
        for i in range(len(self)):
            self.set_entry_(i, value)
    elif self.IsComplex:
        self.set_entry_(0, value)
        self.set_entry_(1, 0)
    elif self.IsQuaternion:
        for i in range(3):
            self.set_entry_(i, 0)
        self.set_entry_(3, value)
    elif self.IsMatrix:
        t = self.Value
        for i in range(len(self)):
            c = _ek.zero(t)
            c.set_entry_(i, t.Value(value))
            self.set_entry_(i, c)
    else:
        raise Exception("broadcast_(): don't know how to handle this type!")


@classmethod
def empty_(cls, size):
    result = cls()
    if cls.Size == Dynamic:
        result.init_(size)
    elif cls.IsDynamic:
        for i in range(len(result)):
            result.set_entry_(i, _ek.empty(cls.Value, size))
    return result


@classmethod
def zero_(cls, size=1):
    result = cls()
    if cls.Size == Dynamic:
        result.init_(size)
        for i in range(size):
            result.set_entry_(i, 0)
    else:
        for i in range(cls.Size):
            result.set_entry_(i, _ek.zero(cls.Value, size))
    return result


@classmethod
def full_(cls, value, size, eval):
    result = cls()
    if cls.Size == Dynamic:
        result.init_(size)
        for i in range(size):
            result.set_entry_(i, value)
    else:
        if _ek.array_depth_v(value) != cls.Depth - 1:
            value = _ek.full(cls.Value, value, size, eval)

        for i in range(cls.Size):
            result.set_entry_(i, value)
    return result


@classmethod
def linspace_(cls, min, max, size=1):
    result = cls.empty_(size)
    step = (max - min) / (len(result) - 1)
    if cls.IsFloat:
        for i in range(len(result)):
            result[i] = min + step * i
    else:
        for i in range(len(result)):
            result[i] = _ek.fmadd(step, i, min)
    return result


@classmethod
def arange_(cls, start, end, step):
    size = (end - start + step - (1 if step > 0 else -1)) / step
    result = cls.empty_(size)
    for i in range(len(result)):
        result[i] = start + step*i
    return result


@classmethod
def gather_(cls, source, index, mask):
    assert source.Depth == 1
    sr = max(len(index), len(mask))
    result = cls.empty_(sr if cls.Size == Dynamic else 0)
    for i in range(sr):
        result[i] = _ek.gather(cls.Value, source, index[i], mask[i])
    return result


def scatter_(self, target, index, mask):
    assert target.Depth == 1
    sr = max(len(self), len(index), len(mask))
    for i in range(sr):
        _ek.scatter(target, self[i], index[i], mask[i])


def scatter_add_(self, target, index, mask):
    assert target.Depth == 1
    sr = max(len(self), len(index), len(mask))
    for i in range(sr):
        _ek.scatter_add(target, self[i], index[i], mask[i])


# -------------------------------------------------------------------
#    Interoperability with other frameworks (NumPy, PyTorch, Jax)
# -------------------------------------------------------------------


def export_(a, migrate_to_host, version):
    shape = _ek.shape(a)
    ndim = len(shape)
    shape = tuple(reversed(shape))

    if not a.IsJIT:
        # Array is already contiguous in memory -- document its structure

        # Fortran-style strides
        temp, strides = a.Type.Size, [0] * ndim
        for i in range(ndim):
            strides[i] = temp
            temp *= shape[i]

        return {
            'shape': shape,
            'strides': tuple(strides),
            'typestr': '<' + a.Type.NumPy,
            'data': (a.data_(), False),
            'version': version,
            'device': -1
        }
    else:
        # JIT array -- requires some extra processing. Cache the
        # result in case this function is called multiple times

        cache = _ek.detail.get_cache(a)
        b = _ek.detach(a) if a.IsDiff else a
        key = (b.index_(), migrate_to_host, version)
        record = cache.get(key, None)
        if record is not None:
            return record

        b = _ek.ravel(b)
        if b is a:
            b = type(a)(b)

        if b.IsCUDA and migrate_to_host:
            b.migrate_(_ek.AllocType.Host)

        # C-style strides
        temp, strides = a.Type.Size, [0] * ndim
        for i in reversed(range(ndim)):
            strides[i] = temp
            temp *= shape[i]

        record = {
            'shape': shape,
            'strides': tuple(strides),
            'typestr': a.Type.NumPy,
            'data': (b.data_(), False),
            'version': version,
            'device': _ek.device(b)
        }

        cache[key] = record
        _ek.eval()
        _ek.sync_stream()
        return record


@property
def op_array_interface(a):
    return a.export_(migrate_to_host=True, version=3)


@property
def op_cuda_array_interface(a):
    if not a.IsCUDA:
        raise Exception("__cuda_array_interface__: only CUDA "
                        "arrays are supported!")
    return a.export_(migrate_to_host=False, version=2)


def dlpack(a):
    struct = a.export_(migrate_to_host=False, version=2)
    isize = a.Type.Size
    strides = tuple(k // isize for k in struct['strides'])
    return _ek.detail.to_dlpack(
        owner=a,
        data=struct['data'][0],
        type=a.Type,
        device=struct['device'],
        shape=struct['shape'],
        strides=strides
    )


def torch(a):
    from torch.utils.dlpack import from_dlpack
    return from_dlpack(a.dlpack())


def numpy(a):
    import numpy
    arr = numpy.array(a, copy=False)
    if a.IsComplex:
        arr = arr = numpy.ascontiguousarray(arr)
        if arr.dtype == numpy.float32:
            return arr.view(numpy.complex64)[..., 0]
        elif arr.dtype == numpy.float64:
            return arr.view(numpy.complex128)[..., 0]
        else:
            raise Exception("Unsupported dtype for complex conversion!")
    return arr


def jax(a):
    from jax.dlpack import from_dlpack
    return from_dlpack(a.dlpack())


def tf(a):
    from tensorflow.experimental.dlpack import from_dlpack
    return from_dlpack(a.dlpack())
