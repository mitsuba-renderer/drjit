from drjit import Dynamic, Exception, VarType
import drjit as _dr
import weakref as _wr


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
        a = _dr.detail.reinterpret_scalar(a, src, dst)
        if isinstance(b, bool):
            b = -1 if b else 0
        else:
            b = _dr.detail.reinterpret_scalar(b, src, dst)

    c = fn(a, b)

    if convert:
        c = _dr.detail.reinterpret_scalar(c, dst, src)

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
    if a0.Value is bool:
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

    if _dr.depth_v(a1) < _dr.depth_v(a0):
        # Avoid type promotion in scalars multiplication, which would
        # be costly for special types (matrices, quaternions, etc.)
        sr = len(a0)
        ar = a0.empty_(sr if a0.Size == Dynamic else 0)
        for i in range(len(a0)):
            ar[i] = a0[i] * a1
        return ar

    ar, sr = _check2(a0, a1)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = a0[i] * a1[i]
    elif a0.IsComplex:
        ar.real = _dr.fma(a0.real, a1.real, -a0.imag * a1.imag)
        ar.imag = _dr.fma(a0.real, a1.imag, a0.imag * a1.real)
    elif a0.IsQuaternion:
        tbl = (4, 3, -2, 1, -3, 4, 1, 2, 2, -1, 4, 3, -1, -2, -3, 4)
        for i in range(4):
            accum = 0
            for j in range(4):
                idx = tbl[i*4 + j]
                value = a1[abs(idx) - 1]
                accum = _dr.fma(a0[j], value if idx > 0 else -value, accum)
            ar[i] = accum
    elif a0.IsMatrix:
        raise Exception("mul(): please use the matrix multiplication operator '@' instead.")
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
        a0.assign(a0 * a1)
    return a0


def matmul_(a0, a1):
    if not (a0.Size == a1.Size and (a0.IsMatrix or a0.IsVector) \
                               and (a1.IsMatrix or a1.IsVector)):
        raise Exception("matmul(): unsupported operand shape!")

    if a0.IsVector and a1.IsVector:
        return _dr.dot(a0, a1)
    elif a0.IsMatrix and a1.IsVector:
        ar = a0[0] * a1[0]
        for i in range(1, a1.Size):
            ar = _dr.fma(a0[i], a1[i], ar)
        return ar
    elif a0.IsVector and a1.IsMatrix:
        ar = a1.Value()
        for i in range(a1.Size):
            ar[i] = _dr.dot(a0, a1[i])
        return ar
    else: # matrix @ matrix
        ar, sr = _check2(a0, a1)
        for j in range(a0.Size):
            accum = a0[0] * _dr.full(a0.Value, a1[0, j])
            for i in range(1, a0.Size):
                accum = _dr.fma(a0[i], _dr.full(a0.Value, a1[i, j]), accum)
            ar[j] = accum
        return ar

def truediv_(a0, a1):
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Dr.Jit integer arrays.")
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
                        "Dr.Jit integer arrays.")
    sr = _check2_inplace(a0, a1)
    if not a0.IsSpecial:
        for i in range(sr):
            a0[i] /= a1[i]
    else:
        a0.assign(a0 / a1)

    return a0


def floordiv_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Dr.Jit floating point arrays.")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] // a1[i]
    return ar


def ifloordiv_(a0, a1):
    if not a0.IsIntegral:
        raise Exception("Use the floor division operator \"/=\" for "
                        "Dr.Jit floating point arrays.")
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
        ar[i] = _dr.eq(a0[i], a1[i])
    return ar


def neq_(a0, a1):
    ar, sr = _check2_mask(a0, a1)
    for i in range(sr):
        ar[i] = _dr.neq(a0[i], a1[i])
    return ar


def sqrt_(a0):
    if not a0.IsFloat:
        raise Exception("sqrt(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.sqrt(a0[i])
    elif a0.IsComplex:
        n = abs(a0)
        m = a0.real >= 0
        zero = _dr.eq(n, 0)
        t1 = _dr.sqrt(.5 * (n + abs(a0.real)))
        t2 = .5 * a0.imag / t1
        im = _dr.select(m, t2, _dr.copysign(t1, a0.imag))
        ar.real = _dr.select(m, t1, abs(t2))
        ar.imag = _dr.select(zero, 0, im)
    elif a0.IsQuaternion:
        ri = _dr.norm(a0.imag)
        cs = _dr.sqrt(a0.Complex(a0.real, ri))
        ar.imag = a0.imag * (_dr.rcp(ri) * cs.imag)
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
            ar[i] = _dr.rsqrt(a0[i])
        return ar
    else:
        return _dr.rcp(_dr.sqrt(a0))


def rcp_(a0):
    if not a0.IsFloat:
        raise Exception("rcp(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.rcp(a0[i])
    elif a0.IsComplex or a0.IsQuaternion:
        return _dr.conj(a0) * _dr.rcp(_dr.squared_norm(a0))
    else:
        raise Exception("rcp(): unsupported array type!")
    return ar


def abs_(a0):
    if not a0.IsArithmetic:
        raise Exception("abs(): requires arithmetic operands!")
    if not a0.IsSpecial or a0.IsMatrix:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _dr.abs(a0[i])
        return ar
    elif a0.IsSpecial:
        return _dr.norm(a0)
    else:
        raise Exception("abs(): unsupported array type!")


def mulhi_(a0, a1):
    ar, sr = _check2(a0, a1)
    if not a0.IsIntegral:
        raise Exception("mulhi(): requires integral operands!")
    for i in range(sr):
        ar[i] = _dr.mulhi(a0[i], a1[i])
    return ar


def maximum_(a0, a1):
    ar, sr = _check2(a0, a1)
    if not a0.IsArithmetic:
        raise Exception("maximum(): requires arithmetic operands!")
    for i in range(sr):
        ar[i] = _dr.maximum(a0[i], a1[i])
    return ar


def minimum_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("minimum(): requires arithmetic operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = _dr.minimum(a0[i], a1[i])
    return ar


def fma_(a0, a1, a2):
    if not a0.IsFloat:
        raise Exception("fma(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check3(a0, a1, a2)
        for i in range(sr):
            ar[i] = _dr.fma(a0[i], a1[i], a2[i])
        return ar
    else:
        return a0 * a1 + a2


def tzcnt_(a0):
    if not a0.IsIntegral:
        raise Exception("tzcnt(): requires integral operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.tzcnt(a0[i])
    return ar


def lzcnt_(a0):
    if not a0.IsIntegral:
        raise Exception("lzcnt(): requires integral operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.lzcnt(a0[i])
    return ar


def popcnt_(a0):
    if not a0.IsIntegral:
        raise Exception("popcnt(): requires integral operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.popcnt(a0[i])
    return ar


def floor_(a0):
    if not a0.IsArithmetic:
        raise Exception("floor(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.floor(a0[i])
    return ar


def ceil_(a0):
    if not a0.IsArithmetic:
        raise Exception("ceil(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.ceil(a0[i])
    return ar


def round_(a0):
    if not a0.IsArithmetic:
        raise Exception("round(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.round(a0[i])
    return ar


def trunc_(a0):
    if not a0.IsArithmetic:
        raise Exception("trunc(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.trunc(a0[i])
    return ar


@staticmethod
def select_(a0, a1, a2):
    ar, sr = _check3_select(a0, a1, a2)
    for i in range(sr):
        ar[i] = _dr.select(a0[i], a1[i], a2[i])
    return ar

# -------------------------------------------------------------------
#       Vertical operations -- ad/JIT compilation-related
# -------------------------------------------------------------------


def label_(a):
    if a.IsTensor:
        return a.array.label_()
    if a.IsJIT or a.IsDiff:
        return [v.label_() for v in a]
    return None


def set_label_(a, label):
    if a.IsTensor:
        return a.array.set_label_(label)
    if a.IsJIT or a.IsDiff:
        if isinstance(label, tuple) or isinstance(label, list):
            if len(a) != len(label):
                raise Exception("Size mismatch!")
            for i, v in enumerate(a):
                v.set_label_(label[i])
        else:
            for i in range(len(a)):
                a[i].set_label_(label + "_%i" % i)


@property
def label(a):
    return a.label_()


@label.setter
def label(a, value):
    return a.set_label_(value)


def schedule_(a0):
    result = False
    if a0.IsTensor:
        result |= a0.array.schedule_()
    elif a0.IsJIT:
        if a0.Depth == 1:
            if a0.IsDiff:
                a0 = a0.detach_()
            result |= _dr.detail.schedule(a0.index)
        else:
            for i in range(len(a0)):
                result |= a0[i].schedule_()
    return result


# -------------------------------------------------------------------
#           Vertical operations -- transcendental functions
# -------------------------------------------------------------------


def sin_(a0):
    if not a0.IsFloat:
        raise Exception("sin(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.sin(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.real)
        sh, ch = _dr.sincosh(a0.imag)
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
            ar[i] = _dr.cos(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.real)
        sh, ch = _dr.sincosh(a0.imag)
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
            result = _dr.sincos(a0[i])
            ar0[i] = result[0]
            ar1[i] = result[1]
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.real)
        sh, ch = _dr.sincosh(a0.imag)
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
            ar[i] = _dr.csc(a0[i])
    else:
        return 1 / _dr.sin(a0)
    return ar


def sec_(a0):
    if not a0.IsFloat:
        raise Exception("sec(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _dr.sec(a0[i])
    else:
        return 1 / _dr.cos(a0)
    return ar


def tan_(a0):
    if not a0.IsFloat:
        raise Exception("tan(): requires floating point operands!")
    if not a0.IsSpecial:
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _dr.tan(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0)
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
            ar[i] = _dr.cot(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0)
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
            ar[i] = _dr.asin(a0[i])
    elif a0.IsSpecial:
        tmp = _dr.log(type(a0)(-a0.imag, a0.real) + _dr.sqrt(1 - _dr.sqr(a0)))
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
            ar[i] = _dr.acos(a0[i])
    elif a0.IsSpecial:
        tmp = _dr.sqrt(1 - _dr.sqr(a0))
        tmp = _dr.log(a0 + type(a0)(-tmp.imag, tmp.real))
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
            ar[i] = _dr.atan(a0[i])
    elif a0.IsSpecial:
        im = type(a0)(0, 1)
        tmp = _dr.log((im - a0) / (im + a0))
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
            ar[i] = _dr.atan2(a0[i], a1[i])
    else:
        raise Exception("atan2(): unsupported array type!")
    return ar


def exp_(a0):
    if not a0.IsFloat:
        raise Exception("exp(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.exp(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.imag)
        exp_r = _dr.exp(a0.real)
        ar.real = exp_r * c
        ar.imag = exp_r * s
    elif a0.IsQuaternion:
        qi = a0.imag
        ri = _dr.norm(qi)
        exp_w = _dr.exp(a0.real)
        s, c = _dr.sincos(ri)
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
            ar[i] = _dr.exp2(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.imag * _dr.log_two)
        exp_r = _dr.exp2(a0.real)
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
            ar[i] = _dr.log(a0[i])
    elif a0.IsComplex:
        ar.real = .5 * _dr.log(_dr.squared_norm(a0))
        ar.imag = _dr.arg(a0)
    elif a0.IsQuaternion:
        qi_n = _dr.normalize(a0.imag)
        rq = _dr.norm(a0)
        acos_rq = _dr.acos(a0.real / rq)
        log_rq = _dr.log(rq)
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
            ar[i] = _dr.log2(a0[i])
    elif a0.IsComplex:
        ar.real = .5 * _dr.log2(_dr.squared_norm(a0))
        ar.imag = _dr.arg(a0) * _dr.inv_log_two
    else:
        raise Exception("log2(): unsupported array type!")
    return ar


def power_(a0, a1):
    if not a0.IsFloat:
        raise Exception("power(): requires floating point operands!")
    if not a0.IsSpecial:
        if isinstance(a1, int) or isinstance(a1, float):
            ar, sr = _check1(a0)
            for i in range(sr):
                ar[i] = _dr.power(a0[i], a1)
        else:
            ar, sr = _check2(a0, a1)
            for i in range(sr):
                ar[i] = _dr.power(a0[i], a1[i])
    else:
        return _dr.exp(_dr.log(a0) * a1)
    return ar


def sinh_(a0):
    if not a0.IsFloat:
        raise Exception("sinh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.sinh(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.imag)
        sh, ch = _dr.sincosh(a0.real)
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
            ar[i] = _dr.cosh(a0[i])
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.imag)
        sh, ch = _dr.sincosh(a0.real)
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
            result = _dr.sincosh(a0[i])
            ar0[i] = result[0]
            ar1[i] = result[1]
    elif a0.IsComplex:
        s, c = _dr.sincos(a0.imag)
        sh, ch = _dr.sincosh(a0.real)
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
            ar[i] = _dr.asinh(a0[i])
    elif a0.IsComplex:
        return _dr.log(a0 + _dr.sqrt(_dr.sqr(a0) + 1))
    else:
        raise Exception("asinh(): unsupported array type!")
    return ar


def acosh_(a0):
    if not a0.IsFloat:
        raise Exception("acosh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.acosh(a0[i])
    elif a0.IsComplex:
        return 2 * _dr.log(_dr.sqrt(.5 * (a0 + 1)) + _dr.sqrt(.5 * (a0 - 1)))
    else:
        raise Exception("acosh(): unsupported array type!")
    return ar


def atanh_(a0):
    if not a0.IsFloat:
        raise Exception("atanh(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.atanh(a0[i])
    elif a0.IsComplex:
        return _dr.log((1 + a0) / (1 - a0)) * .5
    else:
        raise Exception("atanh(): unsupported array type!")
    return ar


def cbrt_(a0):
    if not a0.IsFloat:
        raise Exception("cbrt(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.cbrt(a0[i])
    else:
        raise Exception("cbrt(): unsupported array type!")
    return ar


def erf_(a0):
    if not a0.IsFloat:
        raise Exception("erf(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.erf(a0[i])
    else:
        raise Exception("erf(): unsupported array type!")
    return ar


def erfinv_(a0):
    if not a0.IsFloat:
        raise Exception("erfinv(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.erfinv(a0[i])
    else:
        raise Exception("erfinv(): unsupported array type!")
    return ar


def lgamma_(a0):
    if not a0.IsFloat:
        raise Exception("lgamma(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.lgamma(a0[i])
    else:
        raise Exception("lgamma(): unsupported array type!")
    return ar


def tgamma_(a0):
    if not a0.IsFloat:
        raise Exception("tgamma(): requires floating point operands!")
    ar, sr = _check1(a0)
    if not a0.IsSpecial:
        for i in range(sr):
            ar[i] = _dr.tgamma(a0[i])
    else:
        raise Exception("tgamma(): unsupported array type!")
    return ar


# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def all_(a0):
    if a0.IsTensor:
        return a0.array.all_()
    size = len(a0)
    if size == 0:
        return True
    value = a0[0]
    for i in range(1, size):
        value = value & a0[i]
    return value


def any_(a0):
    if a0.IsTensor:
        return a0.array.any_()
    size = len(a0)
    if size == 0:
        return False
    value = a0[0]
    for i in range(1, size):
        value = value | a0[i]
    return value


def sum_(a0):
    if a0.IsTensor:
        return a0.array.sum_()
    size = len(a0)
    if size == 0:
        return 0
    value = a0[0]
    for i in range(1, size):
        value = value + a0[i]
    return value


def prod_(a0):
    if a0.IsTensor:
        return a0.array.prod_()
    size = len(a0)
    if size == 0:
        return 1
    value = a0[0]
    for i in range(1, size):
        value = value * a0[i]
    return value


def min_(a0):
    if a0.IsTensor:
        return a0.array.min_()
    size = len(a0)
    if size == 0:
        raise Exception("min(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = _dr.minimum(value, a0[i])
    return value


def max_(a0):
    if a0.IsTensor:
        return a0.array.max_()

    size = len(a0)
    if size == 0:
        raise Exception("max(): zero-sized array!")

    value = a0[0]
    for i in range(1, size):
        value = _dr.maximum(value, a0[i])
    return value


def dot_(a0, a1):
    size = len(a0)
    if size == 0:
        return 0
    if size != len(a1):
        raise Exception("dot(): incompatible array sizes!")
    if type(a0) is not type(a1):
        raise Exception("Type mismatch!")

    value = a0[0] * a1[0]
    if a0.IsFloat:
        for i in range(1, size):
            value = _dr.fma(a0[i], a1[i], value)
    else:
        for i in range(1, size):
            value += a0[i] * a1[i]
    return value


def block_sum_(a0, block_size):
    if not a0.IsArithmetic:
        raise Exception("block_sum(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _dr.block_sum(a0[i], block_size)
    return ar

# -------------------------------------------------------------------
#                     Automatic differentiation
# -------------------------------------------------------------------


def detach_(a):
    if not a.IsDiff:
        return a

    t = _dr.detached_t(type(a))

    if a.IsTensor:
        return t(a.array.detach_(), a.shape)
    else:
        result = t.empty_(len(a) if a.Size == Dynamic else 0)
        for i in range(len(a)):
            result[i] = a[i].detach_()
        return result


def grad_(a):
    if not a.IsDiff:
        return None

    t = _dr.detached_t(type(a))

    if a.IsTensor:
        return t(a.array.grad_(), a.shape)
    else:
        result = t.empty_(len(a) if a.Size == Dynamic else 0)
        for i in range(len(a)):
            result[i] = a[i].grad_()

    return result


def grad_enabled_(a):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")
    if a.IsTensor:
        return a.array.grad_enabled_()
    elif a.IsFloat:
        enabled = False
        for i in range(len(a)):
            # ek.Loop requires entry_ref_ here to avoid creating copies
            # of variables, whose entries are temporarily invalid
            enabled |= a.entry_ref_(i).grad_enabled_()
        return enabled
    else:
        return False


def set_grad_enabled_(a, value):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")
    if a.IsTensor:
        a.array.set_grad_enabled_(value)
    else:
        for i in range(len(a)):
            a.entry_ref_(i).set_grad_enabled_(value)


def set_grad_(a, grad):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")

    if a.IsTensor:
        if _dr.is_tensor_v(grad):
            a.array.set_grad_(grad.array)
        else:
            a.array.set_grad_(grad)
    else:
        s = len(a)
        for i in range(s):
            a[i].set_grad_(grad[i])


def accum_grad_(a, grad):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")

    if a.IsTensor:
        if _dr.is_tensor_v(grad):
            a.array.accum_grad_(grad.array)
        else:
            a.array.accum_grad_(grad)
    else:
        s = len(a)
        for i in range(s):
            a[i].accum_grad_(grad[i])


def enqueue_(a, mode):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")
    if a.IsTensor:
        a.array.enqueue_(mode)
    else:
        for i in range(len(a)):
            a[i].enqueue_(mode)


def migrate_(a, target):
    if not a.IsJIT:
        raise Exception("Expected a JIT array type!")
    t = type(a)
    if a.IsTensor:
        return t(a.array.migrate_(target), a.shape)
    result = t.empty_(len(a) if a.Size == Dynamic else 0)
    for i in range(len(a)):
        result[i] = a[i].migrate_(target)
    return result


def index_(a):
    if not a.IsJIT:
        raise Exception("Expected a JIT array type!")
    return tuple(v.index for v in a)


def index_ad_(a):
    if not a.IsDiff:
        raise Exception("Expected a differentiable array type!")
    return tuple(v.index_ad for v in a)


# -------------------------------------------------------------------
#                      Initialization operations
# -------------------------------------------------------------------

def assign(self, other):
    if self is other:
        return
    elif len(self) != len(other):
        raise Exception("assign(): size mismatch!")
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
            c = _dr.zeros(t)
            c.set_entry_(i, t.Value(value))
            self.set_entry_(i, c)
    else:
        raise Exception("broadcast_(): don't know how to handle this type!")


@classmethod
def empty_(cls, shape):
    if cls.IsTensor:
        shape = [shape] if type(shape) is int else shape
        return cls(_dr.empty(cls.Array, _dr.prod(shape)), shape)

    result = cls()
    if cls.Size == Dynamic:
        result.init_(shape)
    elif cls.IsDynamic:
        for i in range(len(result)):
            result.set_entry_(i, _dr.empty(cls.Value, shape))
    return result


@classmethod
def zero_(cls, shape=1):
    if cls.IsTensor:
        shape = [shape] if type(shape) is int else shape
        return cls(_dr.zeros(cls.Array, _dr.prod(shape)), shape)

    result = cls()
    if cls.Size == Dynamic:
        result.init_(shape)
        for i in range(shape):
            result.set_entry_(i, 0)
    else:
        for i in range(cls.Size):
            result.set_entry_(i, _dr.zeros(cls.Value, shape))
    return result


@classmethod
def full_(cls, value, shape):
    if cls.IsTensor:
        shape = [shape] if type(shape) is int else shape
        return cls(_dr.full(cls.Array, value, _dr.prod(shape)), shape)

    result = cls()
    if cls.Size == Dynamic:
        result.init_(shape)
        for i in range(shape):
            result.set_entry_(i, value)
    else:
        if _dr.depth_v(value) != cls.Depth - 1:
            value = _dr.full(cls.Value, value, shape)

        for i in range(cls.Size):
            result.set_entry_(i, value)
    return result


@classmethod
def linspace_(cls, min, max, size=1, endpoint=True):
    if cls.IsTensor:
        raise Exception("linspace_(): Tensor type not supported!")
    result = cls.empty_(size)
    step = (max - min) / (len(result) - (1 if endpoint and (size > 1) else 0))
    if cls.IsFloat:
        for i in range(len(result)):
            result[i] = min + step * i
    else:
        for i in range(len(result)):
            result[i] = _dr.fma(step, i, min)
    return result


@classmethod
def arange_(cls, start, end, step):
    if cls.IsTensor:
        raise Exception("arange_(): Tensor type not supported!")
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
        result[i] = _dr.gather(cls.Value, source, index[i], mask[i])
    return result


def scatter_(self, target, index, mask):
    assert target.Depth == 1
    sr = max(len(self), len(index), len(mask))
    for i in range(sr):
        _dr.scatter(target, self[i], index[i], mask[i])


def scatter_reduce_(self, op, target, index, mask):
    assert target.Depth == 1
    sr = max(len(self), len(index), len(mask))
    for i in range(sr):
        _dr.scatter_reduce(op, target, self[i], index[i], mask[i])


# -------------------------------------------------------------------
#    Interoperability with other frameworks (NumPy, PyTorch, Jax)
# -------------------------------------------------------------------


def export_(a, migrate_to_host, version, owner_supported=True):
    shape = _dr.shape(a)
    ndim = len(shape)
    if not a.IsTensor:
        shape = tuple(reversed(shape))

    if not a.IsJIT:
        # F-style strides
        temp, strides = a.Type.Size, [0] * ndim

        if a.IsTensor:
            # First dimension is the dynamic one, the rest should be in reversed order
            for i in reversed(range(1, ndim)):
                strides[i] = temp
                temp *= shape[i]
            strides[0] = temp
        else:
            # Dr.Jit represents 3D arrays as 4D to leverage SIMD instructions
            padding = 1 if a.IsScalar and a.IsMatrix and shape[0] == 3 else 0
            for i in range(ndim):
                strides[i] = temp
                temp *= shape[i] + padding

        # Array is already contiguous in memory -- document its structure
        return {
            "shape": shape,
            "strides": tuple(strides),
            "typestr": "<" + a.Type.NumPy,
            "data": (a.data_(), False),
            "version": version,
            "device": -1,
            "owner": a
        }
    else:
        # C-style strides
        temp, strides = a.Type.Size, [0] * ndim

        # First dimension is the dynamic one, the rest should be in reversed order
        for i in reversed(range(1, ndim)):
            strides[i if a.IsTensor else (ndim - i)] = temp
            temp *= shape[i]
        strides[0] = temp

        # JIT array -- requires extra transformations
        b = _dr.ravel(_dr.detach(a) if a.IsDiff else a)
        _dr.eval(b)

        if b.IsCUDA and migrate_to_host:
            if b is a:
                b = type(a)(b)
            b = b.migrate_(_dr.AllocType.Host)
            _dr.sync_thread()
        elif b.IsLLVM:
            _dr.sync_thread()

        if not owner_supported and a is not b:
            # If the caller cannot deal with the 'owner' field, use
            # a weak reference to keep 'b' alive while 'a' exists
            _wr.finalize(a, lambda arg: None, b)

        data_ptr = b.data_()
        if data_ptr == 0:
            # NumPy does not accept null pointers (even when the array is empty)
            data_ptr = 1

        record = {
            "shape": shape,
            "strides": tuple(strides),
            "typestr": "<" + a.Type.NumPy,
            "data": (data_ptr, False),
            "version": version,
            "device": _dr.device(b),
            "owner": b
        }

        return record


@property
def op_array_interface(a):
    try:
        return a.export_(migrate_to_host=True, version=3,
                         owner_supported=False)
    except BaseException as e:
        print(e)


@property
def op_cuda_array_interface(a):
    if not a.IsCUDA:
        raise Exception("__cuda_array_interface__: only CUDA "
                        "arrays are supported!")
    return a.export_(migrate_to_host=False, version=2)


def numpy(a):
    import numpy
    arr = numpy.array(a, copy=False)
    if a.IsComplex:
        arr = arr.ravel()
        if arr.dtype == numpy.float32:
            return arr.view(numpy.complex64)[...]
        elif arr.dtype == numpy.float64:
            return arr.view(numpy.complex128)[...]
        else:
            raise Exception("Unsupported dtype for complex conversion!")
    return arr


def op_dlpack(a):
    struct = a.export_(migrate_to_host=False, version=2)
    isize = a.Type.Size
    strides = tuple(k // isize for k in struct["strides"])
    return _dr.detail.to_dlpack(
        owner=struct["owner"],
        data=struct["data"][0],
        type=a.Type,
        device=struct["device"],
        shape=struct["shape"],
        strides=strides
    )


def torch(a):
    from torch.utils.dlpack import from_dlpack
    return from_dlpack(a.__dlpack__())


def jax(a):
    from jax.dlpack import from_dlpack
    from jax import devices
    if a.IsLLVM:
        try:
            # Not all Jax versions accept the 'backend' parameter
            return from_dlpack(a.__dlpack__(), backend=devices(backend="cpu")[0])
        except:
            pass
    return from_dlpack(a.__dlpack__())


def tf(a):
    from tensorflow.experimental.dlpack import from_dlpack
    from tensorflow import constant
    constant(0) # Dummy op to ensure that the Tensorflow context is initialized
    return from_dlpack(a.__dlpack__())


def op_iter(a):
    size = len(a)
    class array_iterator:
        def __init__(self):
            self.i = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.i >= size:
                raise StopIteration()
            self.i += 1
            return a[self.i-1]
    return array_iterator()