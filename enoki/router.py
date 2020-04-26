import enoki as _ek
from enoki import ArrayBase, VarType, Exception, Dynamic
from enoki.detail import array_name as _array_name
from sys import modules as _modules
import math as _math
import builtins as _builtins

# -------------------------------------------------------------------
#                        Type promotion logic
# -------------------------------------------------------------------


def _var_is_enoki(a):
    return isinstance(a, ArrayBase)


def _var_type(a, preferred=VarType.Invalid):
    """
    Return the VarType of a given Enoki object or plain Python type. Return
    'preferred' when there is sufficient room for interpretation (e.g. when
    given an 'int').
    """
    if isinstance(a, ArrayBase):
        return a.Type
    elif isinstance(a, float):
        return VarType.Float32
    elif isinstance(a, bool):
        return VarType.Bool
    elif isinstance(a, int):
        ok = False

        if preferred is not VarType.Invalid:
            if preferred == VarType.UInt32:
                ok |= a >= 0 and a <= 0xFFFFFFFF
            elif preferred == VarType.Int32:
                ok |= a >= -0x7FFFFFFF and a <= 0x7FFFFFFF
            elif preferred == VarType.UInt64:
                ok |= a >= 0 and a <= 0xFFFFFFFFFFFFFFFF
            elif preferred == VarType.Int64:
                ok |= a >= -0x7FFFFFFFFFFFFFFF and a <= 0x7FFFFFFFFFFFFFFF

        if not ok:
            if a >= 0:
                if a <= 0xFFFFFFFF:
                    preferred = VarType.UInt32
                else:
                    preferred = VarType.UInt64
            else:
                if a >= -0x7FFFFFFF:
                    preferred = VarType.Int32
                else:
                    preferred = VarType.Int64

        return preferred
    else:
        raise Exception("var_type(): Unsupported type!")


def _var_promote(*args):
    """
    Given a list of Enoki arrays and scalars, determine the flavor and shape of
    the result array and broadcast/convert everything into this form.
    """
    n = len(args)
    vt = [None] * n
    base = None
    depth = 0

    for i, a in enumerate(args):
        vt[i] = _var_type(a)
        depth_i = getattr(a, 'Depth', 0)
        if depth_i > depth:
            base = a
            depth = depth_i

    if base is None:
        raise Exception("At least one of the input arguments "
                        "must be an Enoki array!")

    for i in range(n):
        j = (i + 1) % n
        if vt[i] != vt[j]:
            vt[i] = _var_type(args[i], vt[j])

    t = base.ReplaceScalar(max(vt))

    result = list(args)
    for i, a in enumerate(result):
        if type(a) is not t:
            result[i] = t(result[i])

    return result


def _var_promote_mask(a0, a1):
    """
    Like _var_promote(), but has a special case where 'a1' can be a mask.
    """
    vt0 = _var_type(a0)
    vt1 = _var_type(a1)

    if vt0 != vt1:
        vt0 = _var_type(a0, vt1)
        vt1 = _var_type(a1, vt0)

    if vt1 != VarType.Bool:
        vt0 = vt1 = max(vt0, vt1)

    base = a0 if getattr(a0, 'Depth', 0) >= getattr(a1, 'Depth', 0) else a1
    t0 = base.ReplaceScalar(vt0)
    t1 = base.ReplaceScalar(vt1)

    if type(a0) is not t0:
        a0 = t0(a0)
    if type(a1) is not t1:
        a1 = t1(a1)

    return a0, a1


def _replace_scalar(cls, vt):
    name = _array_name(vt, cls.Depth, cls.Size, cls.IsScalar)
    module = _modules.get(cls.__module__)
    return getattr(module, name)


ArrayBase.ReplaceScalar = classmethod(_replace_scalar)

# -------------------------------------------------------------------
#                      Miscellaneous operations
# -------------------------------------------------------------------


def shape(a):
    """
    Return the shape of an N-dimensional Enoki input array, or an empty list
    when the provided argument is not an Enoki array.
    """
    result = []
    while isinstance(a, ArrayBase):
        size = len(a)
        result.append(size)
        if size == 0:
            while True:
                a = a.Value
                if not issubclass(a, ArrayBase):
                    break
                result.append(0)
            break
        a = a[0]
    return result


def _ragged_impl(a, shape, i, ndim):
    """Implementation detail of ragged()"""
    if len(a) != shape[i]:
        return True

    if i + 1 != ndim:
        for j in range(shape[i]):
            if _ragged_impl(a[j], shape, i + 1, ndim):
                return True

    return False


def ragged(a):
    """
    Check if the Enoki array ``a`` has ragged entries (e.g. when ``len(a[0])
    != len(a[1])``). Enoki can work with such arrays, but they are a special
    case and unsupported by some operations (e.g. ``repr()``).
    """
    s = shape(a)
    ndim = len(s)
    if ndim == 0:
        return False
    return _ragged_impl(a, s, 0, ndim)


# By default, don't print full contents of arrays with more than 20 entries
_print_threshold = 20


def _repr_impl(self, shape, buf, *args):
    """Implementation detail of op_repr()"""
    k = len(shape) - len(args)
    if k == 0:
        buf.write(repr(self[args]))
    else:
        size = shape[k - 1]
        buf.write('[')
        i = 0
        while i < size:
            if size > _print_threshold and i == 5:
                buf.write('.. %i skipped ..' % (size - 10))
                i = size - 6
            else:
                _repr_impl(self, shape, buf, i, *args)

            if i + 1 < size:
                if k == 1:
                    buf.write(', ')
                else:
                    buf.write(',\n')
                    buf.write(' ' * (len(args) + 1))
            i += 1
        buf.write(']')


def print_threshold():
    return _print_threshold


def set_print_threshold(size):
    global _print_threshold
    _print_threshold = max(size, 11)


def op_repr(self):
    if len(self) == 0:
        return '[]'

    s = shape(self)
    if _ragged_impl(self, s, 0, len(s)):
        return "[ragged array]"
    else:
        import io
        buf = io.StringIO()
        _repr_impl(self, s, buf)
        return buf.getvalue()


def op_bool(self):
    raise Exception(
        "To convert an Enoki array into a boolean value, use a mask reduction "
        "operation such as enoki.all(), enoki.any(), enoki.none(). Special "
        "variants (enoki.all_nested(), etc.) are available for nested arrays.")


# Mainly for testcases: keep track of how often eval() is invoked.
_coeff_evals = 0


def op_getitem(self, index):
    global _coeff_evals
    if isinstance(index, tuple):
        for i in index:
            self = op_getitem(self, i)
        return self
    else:
        size = len(self)
        if index < 0:
            index = size + index
        if index >= 0 and index < size:
            _coeff_evals += 1
            return self.coeff(index)
        else:
            raise IndexError("Index %i exceeds the array "
                             "bounds %i!" % (index, size))


def op_setitem(self, index, value):
    global _coeff_evals
    if isinstance(index, tuple):
        for i in index[:-1]:
            self = op_getitem(self, i)
        op_setitem(self, index[-1], value)
    else:
        size = len(self)
        if index < 0:
            index = size + index
        if index >= 0 and index < size:
            _coeff_evals += 1
            self.set_coeff(index, value)
        else:
            raise IndexError("Index %i exceeds the array "
                             "bounds %i!" % (index, size))


def reinterpret_array(target_type, value,
                      vt_target=VarType.Invalid,
                      vt_value=VarType.Invalid):
    assert isinstance(target_type, type)

    if issubclass(target_type, ArrayBase):
        if hasattr(target_type, "reinterpret_array_"):
            return target_type.reinterpret_array_(value)
        else:
            result = target_type()
            if result.Size == Dynamic:
                result.init_(len(value))

            for i in range(len(value)):
                result[i] = reinterpret_array(target_type.Value, value[i],
                                              target_type.Type, value.Type)

            return result
    else:
        return _ek.detail.reinterpret_scalar(value, vt_value, vt_target)


# -------------------------------------------------------------------
#                        Vertical operations
# -------------------------------------------------------------------


def op_neg(a):
    return a.neg_()


def op_invert(a):
    return a.not_()


def op_add(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.add_(b)


def op_iadd(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.iadd_(b)


def op_sub(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sub_(b)


def op_isub(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.isub_(b)


def op_mul(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.mul_(b)


def op_imul(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.imul_(b)


def op_truediv(a, b):
    if isinstance(b, float) or isinstance(b, int):
        return a * (1.0 / b)

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.truediv_(b)


def op_itruediv(a, b):
    if isinstance(b, float) or isinstance(b, int):
        a *= 1.0 / b

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.itruediv_(b)


def op_floordiv(a, b):
    if isinstance(b, int):
        if b != 0 and b & (b - 1) == 0:
            return a >> int(_math.log2(b))

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.floordiv_(b)


def op_ifloordiv(a, b):
    if isinstance(b, int):
        if b != 0 and b & (b - 1) == 0:
            a >>= int(_math.log2(b))

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.ifloordiv_(b)


def op_mod(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.mod_(b)


def op_imod(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.imod_(b)


def op_and(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)

    return a.and_(b)


def op_iand(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)

    return a.iand_(b)


def op_or(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.or_(b)


def op_ior(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.ior_(b)


def op_xor(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.xor_(b)


def op_ixor(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.ixor_(b)


def op_lshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sl_(b)


def op_ilshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.isl_(b)


def op_rshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sr_(b)


def op_irshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.isr_(b)


def op_lt(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.lt_(b)


def op_le(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.le_(b)


def op_gt(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.gt_(b)


def op_ge(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.ge_(b)


def op_eq(a, b):
    return all_nested(eq(a, b))


def op_neq(a, b):
    return any_nested(neq(a, b))


def eq(a, b):
    if not isinstance(a, ArrayBase) and not isinstance(b, ArrayBase):
        return a == b
    else:
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.eq_(b)


def neq(a, b):
    if not isinstance(a, ArrayBase) and not isinstance(b, ArrayBase):
        return a != b
    else:
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.neq_(b)


def sqrt(a):
    if not isinstance(a, ArrayBase):
        return _math.sqrt(a)
    else:
        return a.sqrt_()


def abs(a):
    if not isinstance(a, ArrayBase):
        return _builtins.abs(a)
    else:
        return a.abs_()


def op_abs(a):
    if not isinstance(a, ArrayBase):
        return _builtins.abs(a)
    else:
        return a.abs_()


def max(a, b):
    if not isinstance(a, ArrayBase) and \
       not isinstance(b, ArrayBase):
        return _builtins.max(a, b)
    else:
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.max_(b)


def min(a, b):
    if not isinstance(a, ArrayBase) and \
       not isinstance(b, ArrayBase):
        return _builtins.min(a, b)
    else:
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.min_(b)


def fmadd(a, b, c):
    if not isinstance(a, ArrayBase) and \
       not isinstance(b, ArrayBase) and \
       not isinstance(c, ArrayBase):
        return _ek.detail.fmadd_scalar(a, b, c)
    else:
        if type(a) is not type(b) or type(b) is not type(c):
            a, b, c = _var_promote(a, b, c)
        return a.fmadd_(b, c)


def fmsub(a, b, c):
    return fmadd(a, b, -c)


def fnmadd(a, b, c):
    return fmadd(-a, b, c)


def fnmsub(a, b, c):
    return fmadd(-a, b, -c)

# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def all(a):
    if _var_type(a) == VarType.Bool:
        return a.all_() if _var_is_enoki(a) else a
    else:
        raise Exception("all(): input array must be a mask!")


def all_nested(a):
    while _var_is_enoki(a):
        a = all(a)
    return a


def any(a):
    if _var_type(a) == VarType.Bool:
        return a.any_() if _var_is_enoki(a) else a
    else:
        raise Exception("any(): input array must be a mask!")


def any_nested(a):
    while _var_is_enoki(a):
        a = all(a)
    return a


def none(a):
    if _var_type(a) != VarType.Bool:
        raise Exception("none(): input array must be a mask!")

    if _var_is_enoki(a):
        return ~any(a)
    else:
        return not a


def none_nested(a):
    return not any_nested(a)


def hsum(a):
    return a.hsum_() if _var_is_enoki(a) else a


def hsum_nested(a):
    while _var_is_enoki(a):
        a = hsum(a)
    return a


def hprod(a):
    return a.hprod_() if _var_is_enoki(a) else a


def hprod_nested(a):
    while _var_is_enoki(a):
        a = hprod(a)
    return a


def hmax(a):
    return a.hmax_() if _var_is_enoki(a) else a


def hmax_nested(a):
    while _var_is_enoki(a):
        a = hmax(a)
    return a


def hmin(a):
    return a.hmin_() if _var_is_enoki(a) else a


def hmin_nested(a):
    while _var_is_enoki(a):
        a = hmin(a)
    return a
