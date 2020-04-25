from enoki import ArrayBase, VarType, Exception
from enoki.detail import array_name as _array_name
from sys import modules as _modules

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
    n = len(args)
    vt = [None] * n
    base = None

    for i, a in enumerate(args):
        vt[i] = _var_type(a)
        if isinstance(a, ArrayBase):
            if base is None or a.Depth > base.Depth:
                base = a

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


def _replace_scalar(cls, vt):
    name = _array_name(vt, cls.Depth, cls.Size, 'scalar' in cls.__name__)
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
        if index < size:
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
        if index < size:
            _coeff_evals += 1
            self.set_coeff(index, value)
        else:
            raise IndexError("Index %i exceeds the array "
                             "bounds %i!" % (index, size))



# -------------------------------------------------------------------
#                        Vertical operations
# -------------------------------------------------------------------

def op_add(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.add_(b)


def op_sub(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sub_(b)


def op_mul(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.mul_(b)

# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def all(a):
    if _var_type(a) != VarType.Bool:
        raise Exception("all(): input array must be a mask!")

    if _var_is_enoki(a):
        return a.all_()
    else:
        return a


def any(a):
    if _var_type(a) != VarType.Bool:
        raise Exception("any(): input array must be a mask!")

    if _var_is_enoki(a):
        return a.any_()
    else:
        return a


def none(a):
    if _var_type(a) != VarType.Bool:
        raise Exception("none(): input array must be a mask!")

    if _var_is_enoki(a):
        return ~any(a)
    else:
        return not a
