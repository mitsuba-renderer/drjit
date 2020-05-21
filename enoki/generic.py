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
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] * a1[i]
    return ar


def imul_(a0, a1):
    if not a0.IsArithmetic:
        raise Exception("imul(): requires arithmetic operands!")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] *= a1[i]
    return a0


def truediv_(a0, a1):
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Enoki integer arrays.")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = a0[i] / a1[i]
    return ar


def itruediv_(a0, a1):
    if not a0.IsFloat:
        raise Exception("Use the floor division operator \"//=\" for "
                        "Enoki integer arrays.")
    sr = _check2_inplace(a0, a1)
    for i in range(sr):
        a0[i] /= a1[i]
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
    for i in range(sr):
        ar[i] = _ek.sqrt(a0[i])
    return ar


def rsqrt_(a0):
    if not a0.IsFloat:
        raise Exception("rsqrt(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.rsqrt(a0[i])
    return ar


def rcp_(a0):
    if not a0.IsFloat:
        raise Exception("rcp(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.rcp(a0[i])
    return ar


def abs_(a0):
    if not a0.IsArithmetic:
        raise Exception("abs(): requires arithmetic operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.abs(a0[i])
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
    ar, sr = _check3(a0, a1, a2)
    for i in range(sr):
        ar[i] = _ek.fmadd(a0[i], a1[i], a2[i])
    return ar


@staticmethod
def select_(a0, a1, a2):
    ar, sr = _check3_select(a0, a1, a2)
    for i in range(sr):
        ar[i] = _ek.select(a0[i], a1[i], a2[i])
    return ar

# -------------------------------------------------------------------
#       Vertical operations -- autodiff/JIT compilation-related
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
    for i in range(sr):
        ar[i] = _ek.sin(a0[i])
    return ar


def cos_(a0):
    if not a0.IsFloat:
        raise Exception("cos(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.cos(a0[i])
    return ar


def sincos_(a0):
    if not a0.IsFloat:
        raise Exception("sincos(): requires floating point operands!")
    ar0, sr0 = _check1(a0)
    ar1 = a0.empty_(sr0 if a0.Size == Dynamic else 0)
    for i in range(sr0):
        result = _ek.sincos(a0[i])
        ar0[i] = result[0]
        ar1[i] = result[1]
    return ar0, ar1


def tan_(a0):
    if not a0.IsFloat:
        raise Exception("tan(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.tan(a0[i])
    return ar


def cot_(a0):
    if not a0.IsFloat:
        raise Exception("cot(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.cot(a0[i])
    return ar


def asin_(a0):
    if not a0.IsFloat:
        raise Exception("asin(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.asin(a0[i])
    return ar


def acos_(a0):
    if not a0.IsFloat:
        raise Exception("acos(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.acos(a0[i])
    return ar


def atan_(a0):
    if not a0.IsFloat:
        raise Exception("atan(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.atan(a0[i])
    return ar


def atan2_(a0, a1):
    if not a0.IsFloat:
        raise Exception("atan2(): requires floating point operands!")
    ar, sr = _check2(a0, a1)
    for i in range(sr):
        ar[i] = _ek.atan2(a0[i], a1[i])
    return ar


def exp_(a0):
    if not a0.IsFloat:
        raise Exception("exp(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.exp(a0[i])
    return ar


def exp2_(a0):
    if not a0.IsFloat:
        raise Exception("exp2(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.exp2(a0[i])
    return ar


def log_(a0):
    if not a0.IsFloat:
        raise Exception("log(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.log(a0[i])
    return ar


def log2_(a0):
    if not a0.IsFloat:
        raise Exception("log2(): requires floating point operands!")
    ar, sr = _check1(a0)
    for i in range(sr):
        ar[i] = _ek.log2(a0[i])
    return ar


def pow_(a0, a1):
    if not a0.IsFloat:
        raise Exception("pow(): requires floating point operands!")
    if isinstance(a1, int) or isinstance(a1, float):
        ar, sr = _check1(a0)
        for i in range(sr):
            ar[i] = _ek.pow(a0[i], a1)
    else:
        ar, sr = _check2(a0, a1)
        for i in range(sr):
            ar[i] = _ek.pow(a0[i], a1[i])
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


def et_grad_enabled_(a, value):
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


# -------------------------------------------------------------------
#                      Initialization operations
# -------------------------------------------------------------------


@classmethod
def zero_(cls, size=1):
    result = cls.empty_(size)
    for i in range(len(result)):
        result[i] = 0
    return result


@classmethod
def full_(cls, value, size=1):
    result = cls.empty_(size)
    for i in range(len(result)):
        result[i] = value
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
