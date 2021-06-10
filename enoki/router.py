import enoki as _ek
from enoki import ArrayBase, VarType, Exception, Dynamic
from enoki.detail import array_name as _array_name
from sys import modules as _modules
import math as _math
import builtins as _builtins
from collections.abc import Mapping as _Mapping

# -------------------------------------------------------------------
#                        Type promotion logic
# -------------------------------------------------------------------


def _var_is_enoki(a):
    return isinstance(a, ArrayBase)


def _var_type(a, preferred=VarType.Void):
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

        if preferred is not VarType.Void:
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
    elif type(a).__module__ == 'numpy':
        for t in [VarType.Float32, VarType.Float64, VarType.Int32,
                  VarType.Int64, VarType.UInt32, VarType.UInt64]:
            if t.NumPy == a.dtype:
                return t
    elif isinstance(a, tuple) or isinstance(a, list):
        return _builtins.max([_var_type(v, preferred) for v in a])
    elif isinstance(a, type(None)) or 'pybind11' in type(type(a)).__name__:
        return VarType.Pointer
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
    diff = False

    for i, a in enumerate(args):
        vt[i] = _var_type(a)
        depth_i = getattr(a, 'Depth', 0)
        diff |= getattr(a, 'IsDiff', False)
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

    t = type(base)
    vtm = _builtins.max(vt)
    if t.IsDiff != diff or t.Type != vtm:
        t = base.ReplaceScalar(vtm, diff)

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
    diff = getattr(a0, 'IsDiff', False) | getattr(a1, 'IsDiff', False)

    if vt0 != vt1:
        vt0 = _var_type(a0, vt1)
        vt1 = _var_type(a1, vt0)

    if vt1 != VarType.Bool:
        vt0 = vt1 = _builtins.max(vt0, vt1)

    base = a0 if getattr(a0, 'Depth', 0) >= getattr(a1, 'Depth', 0) else a1
    t0 = base.ReplaceScalar(vt0, diff)
    t1 = base.ReplaceScalar(vt1, diff)

    if type(a0) is not t0:
        a0 = t0(a0)
    if type(a1) is not t1:
        a1 = t1(a1)

    return a0, a1


def _var_promote_select(a0, a1, a2):
    """
    Like _var_promote(), but specially adapted to the select() operation
    """
    vt0 = _var_type(a0)
    vt1 = _var_type(a1)
    vt2 = _var_type(a2)
    diff = getattr(a0, 'IsDiff', False) | \
           getattr(a1, 'IsDiff', False) | \
           getattr(a2, 'IsDiff', False)

    if vt1 != vt2:
        vt1 = _var_type(a1, vt2)
        vt2 = _var_type(a2, vt1)

    if vt0 != VarType.Bool:
        raise Exception("select(): first argument must be a mask!")

    base = a0 if getattr(a0, 'Depth', 0) >= getattr(a1, 'Depth', 0) else a1
    base = base if getattr(base, 'Depth', 0) >= getattr(a2, 'Depth', 0) else a2

    t0 = base.ReplaceScalar(vt0, diff)
    t12 = base.ReplaceScalar(_builtins.max(vt1, vt2), diff)

    if type(a0) is not t0:
        a0 = t0(a0)
    if type(a1) is not t12:
        a1 = t12(a1)
    if type(a2) is not t12:
        a2 = t12(a2)

    return a0, a1, a2


def _replace_scalar(cls, vt, diff=None):
    name = _array_name(cls.Prefix, vt, cls.Shape, cls.IsScalar)
    modname = cls.__module__

    if not modname.startswith('enoki.'):
        if cls.IsCUDA:
            modname = "enoki.cuda"
        elif cls.IsLLVM:
            modname = "enoki.llvm"
        elif cls.IsPacket:
            modname = "enoki.packet"
        else:
            modname = "enoki.scalar"

        if cls.IsDiff:
            modname += '.ad'

    if diff is not None:
        is_diff = modname.endswith('.ad')
        if is_diff != diff:
            if diff:
                modname += '.ad'
            else:
                modname = modname[:-3]

    module = _modules.get(modname)
    return getattr(module, name)


ArrayBase.ReplaceScalar = classmethod(_replace_scalar)


# -------------------------------------------------------------------
#                      Miscellaneous operations
# -------------------------------------------------------------------


def _shape_impl(a, i, shape):
    if not isinstance(a, ArrayBase):
        return

    size = len(a)
    if i < len(shape):
        cur = shape[i]
        maxval = _builtins.max(cur, size)

        if maxval != size and size != 1:
            return False

        shape[i] = maxval
    else:
        shape.append(size)

    if issubclass(a.Value, ArrayBase):
        for j in range(size):
            if not _shape_impl(a[j], i + 1, shape):
                return False

    return True


def shape(a):
    """
    Return the shape of an N-dimensional Enoki input array, or an empty list
    when the provided argument is not an Enoki array.

    When the arrays is ragged, the implementation signals a failure by
    returning ``None``. A ragged array has entries of incompatible size, e.g.
    ``[[1, 2], [3, 4, 5]]``. Note that an scalar entries (e.g. ``[[1, 2],
    [3]]``) are acceptable, since broadcasting can effectively convert them to
    any size.

    """
    s = []
    if not _shape_impl(a, 0, s):
        return None
    else:
        return s


def width(value):
    if _ek.is_array_v(value):
        return shape(value)[-1]
    elif _ek.is_enoki_struct_v(value):
        result = 0
        for k in type(value).ENOKI_STRUCT.keys():
            result = max(result, width(getattr(value, k)))
        return result
    else:
        return 1


def resize(value, size):
    if _ek.array_depth_v(value) > 1:
        for i in range(value.Size):
            resize(value[i], size)
    elif _ek.is_jit_array_v(value):
        value.resize_(size)
    elif _ek.is_enoki_struct_v(value):
        for k in type(value).ENOKI_STRUCT.keys():
            resize(getattr(value, k), size)


def device(value=None):
    if value is None:
        return _ek.detail.device()
    elif _ek.array_depth_v(value) > 1:
        return device(value[0])
    elif _ek.is_diff_array_v(value):
        return device(_ek.detach(value))
    elif _ek.is_jit_array_v(value):
        return _ek.detail.device(value.index())
    else:
        return -1


# By default, don't print full contents of arrays with more than 20 entries
_print_threshold = 20


def _repr_impl(self, shape, buf, *idx):
    """Implementation detail of op_repr()"""
    k = len(shape) - len(idx)
    if k == 0:
        el = idx[0]

        if self.IsQuaternion:
            idx = (3 if el == 0 else (el - 1), *idx[1:])

        value = self
        for k in idx:
            value = value.entry_(k)
        value_str = repr(value)

        if (self.IsComplex or self.IsQuaternion) and el > 0:
            if value_str[0] == '-':
                value_str = '- ' + value_str[1:]
            else:
                value_str = '+ ' + value_str

            value_str += '_ijk'[el]

        buf.write(value_str)
    else:
        size = shape[k - 1]
        buf.write('[')
        i = 0
        while i < size:
            if size > _print_threshold and i == 5:
                buf.write('.. %i skipped ..' % (size - 10))
                i = size - 6
            else:
                _repr_impl(self, shape, buf, i, *idx)

            if i + 1 < size:
                if k == 1:
                    buf.write(' ' if self.IsComplex or self.IsQuaternion
                              else ', ')
                else:
                    buf.write(',\n')
                    buf.write(' ' * (len(idx) + 1))
            i += 1
        buf.write(']')


def print_threshold():
    return _print_threshold


def set_print_threshold(size):
    global _print_threshold
    _print_threshold = _builtins.max(size, 11)


def op_repr(self):
    if len(self) == 0:
        return '[]'

    s = shape(self)
    if s is None:
        return "[ragged array]"
    else:
        import io
        buf = io.StringIO()
        try:
            self.schedule_()
        except BaseException as e:  # noqa
            return "[backend issue: %s]" % str(e)
        _repr_impl(self, s, buf)
        return buf.getvalue()


def op_bool(self):
    raise Exception(
        "To convert an Enoki array into a boolean value, use a mask reduction "
        "operation such as enoki.all(), enoki.any(), enoki.none(). Special "
        "variants (enoki.all_nested(), etc.) are available for nested arrays.")


def op_len(self):
    return self.Size


# Mainly for testcases: keep track of how often entry() is invoked.
_entry_evals = 0


def op_getitem(self, index):
    global _entry_evals
    if isinstance(index, int):
        size = len(self)
        if size == 1:
            index = 0
        if index < 0:
            index = size + index

        if index >= 0 and index < size:
            _entry_evals += 1
            return self.entry_(index)
        else:
            raise IndexError("Tried to read from array index %i, which "
                             "exceeds its size (%i)!" % (index, size))
    elif isinstance(index, tuple):
        if self.IsMatrix:
            index = (index[1], index[0], *index[2:])
        for i in index:
            self = op_getitem(self, i)
        return self
    elif isinstance(index, _builtins.slice):
        if not self.Size == Dynamic:
            raise Exception("Indexing via slice is only allowed in the case of "
                            "dynamic arrays")
        indices = tuple(range(len(self)))[index]
        result = _ek.empty(type(self), len(indices))
        for i in range(len(indices)):
            result[i] = op_getitem(self, indices[i])
        return result
    elif _ek.is_mask_v(index):
        raise Exception("Indexing via masks is only allowed in the case of "
                        "assignments, e.g.: array[mask] = value")
    else:
        raise Exception("Invalid array index! (must be an integer or a tuple "
                        "of integers!)")


def op_setitem(self, index, value):
    if isinstance(index, int):
        size = len(self)
        if index < 0:
            index = size + index
        if index >= 0 and index < size:
            global _entry_evals
            _entry_evals += 1
            if not type(value) is self.Value:
                self.set_entry_(index, self.Value(value))
            else:
                self.set_entry_(index, value)
        else:
            raise IndexError("Tried to write to array index %i, which "
                             "exceeds its size (%i)!" % (index, size))
    elif isinstance(index, tuple):
        if len(index) > 1:
            if self.IsMatrix:
                index = (index[1], index[0], *index[2:])
            if isinstance(index[0], _builtins.slice):
                indices = tuple(range(len(self)))[index[0]]
                for i in range(len(indices)):
                    value2 = op_getitem(self, indices[i])
                    op_setitem(value2, index[1:], value)
                    op_setitem(self, indices[i], value2)
            else:
                value2 = op_getitem(self, index[0])
                op_setitem(value2, index[1:], value)
                op_setitem(self, index[0], value2)
        else:
            op_setitem(self, index[0], value)
    elif _ek.is_mask_v(index):
        self.assign(_ek.select(index, value, self))
    elif isinstance(index, _builtins.slice):
        indices = tuple(range(len(self)))[index]
        for i in range(len(indices)):
            if isinstance(value, (float, int, bool)):
                op_setitem(self, indices[i], value)
            else:
                op_setitem(self, indices[i], value[i])
    else:
        raise Exception("Invalid array index! (must be an integer or a tuple "
                        "of integers!)")
    return self


def reinterpret_array(target_type, value,
                      vt_target=VarType.Void,
                      vt_value=VarType.Void):
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
#                      Scatter/gather operations
# -------------------------------------------------------------------

def _broadcast_index(target_type, index):
    size = target_type.Size
    if _ek.array_size_v(index) <= 1 and size == Dynamic:
        return target_type(index)
    elif target_type.Depth > index.Depth:
        assert size != Dynamic
        index_scaled = index * size
        result = target_type()
        for i in range(size):
            result[i] = _broadcast_index(target_type.Value, index_scaled + i)
        return result
    else:
        return index


def gather(target_type, source, index, mask=True, permute=False):
    if not isinstance(target_type, type):
        raise Exception('gather(): Type expected as first argument')
    elif not issubclass(target_type, ArrayBase):
        if _ek.is_enoki_struct_v(target_type):
            if type(source) is not target_type:
                raise Exception('gather(): type mismatch involving custom data structure!')
            result = target_type()
            for k, v in target_type.ENOKI_STRUCT.items():
                setattr(result, k,
                        gather(v, getattr(source, k), index, mask, permute))
            return result
        else:
            assert isinstance(index, int) and isinstance(mask, bool)
            return source[index] if mask else 0
    else:
        if source.Depth != 1:
            if source.Size != target_type.Size:
                raise Exception("gather(): mismatched source/target configuration!")

            result = target_type()
            for i in range(target_type.Size):
                result[i] = gather(target_type.Value, source[i], index, mask, permute)
            return result
        else:
            index_type = _ek.uint32_array_t(target_type)
            if not isinstance(index, index_type):
                index = _broadcast_index(index_type, index)

            mask_type = index_type.MaskType
            if not isinstance(mask, mask_type):
                mask = mask_type(mask)

            return target_type.gather_(source, index, mask, permute)


def scatter(target, value, index, mask=True, permute=False):
    target_type = type(target)
    if not issubclass(target_type, ArrayBase):
        if _ek.is_enoki_struct_v(target_type):
            if type(value) is not target_type:
                raise Exception('scatter(): type mismatch involving custom data structure!')
            for k in target_type.ENOKI_STRUCT.keys():
                scatter(getattr(target, k), getattr(value, k),
                        index, mask, permute)
        else:
            assert isinstance(index, int) and isinstance(mask, bool)
            if mask:
                target[index] = value
    else:
        if target.Depth != 1:
            if _ek.array_size_v(target) != _ek.array_size_v(value):
                raise Exception("scatter(): mismatched source/target configuration!")

            for i in range(len(target)):
                scatter(target.entry_ref_(i), value[i], index, mask, permute)
        else:
            index_type = _ek.uint32_array_t(type(value))
            if not isinstance(index, index_type):
                index = _broadcast_index(index_type, index)

            mask_type = index_type.MaskType
            if not isinstance(mask, mask_type):
                mask = mask_type(mask)

            return value.scatter_(target, index, mask, permute)


def scatter_reduce(op, target, value, index, mask=True):
    target_type = type(target)
    if not issubclass(target_type, ArrayBase):
        if _ek.is_enoki_struct_v(target_type):
            if type(value) is not target_type:
                raise Exception('scatter_reduce(): type mismatch involving custom data structure!')
            for k in target_type.ENOKI_STRUCT.keys():
                scatter_reduce(op, getattr(target, k), getattr(value, k),
                               index, mask)
        else:
            assert isinstance(index, int) and isinstance(mask, bool)
            if mask:
                target[index] += value
    else:
        if target.Depth != 1:
            if _ek.array_size_v(target) != _ek.array_size_v(value):
                raise Exception("scatter_reduce(): mismatched source/target configuration!")

            for i in range(len(target)):
                scatter_reduce(op, target.entry_ref_(i), value[i], index, mask)
        else:
            index_type = _ek.uint32_array_t(type(value))
            if not isinstance(index, index_type):
                index = _broadcast_index(index_type, index)

            mask_type = index_type.MaskType
            if not isinstance(mask, mask_type):
                mask = mask_type(mask)

            return value.scatter_reduce_(op, target, index, mask)


def ravel(array):
    if not _var_is_enoki(array) or array.Depth == 1:
        return array

    s = shape(array)
    if s is None:
        raise Exception('ravel(): ragged arrays not permitted!')

    target_type = type(array)
    while target_type.Depth > 1:
        target_type = target_type.Value
    index_type = _ek.uint32_array_t(target_type)

    target = empty(target_type, hprod(s))
    scatter(target, array, arange(index_type, s[-1]))
    return target


def unravel(target_class, array):
    if not isinstance(array, ArrayBase) or array.Depth != 1:
        raise Exception('unravel(): array input must be a flat array!')
    elif not issubclass(target_class, ArrayBase) or target_class.Depth == 1:
        raise Exception("unravel(): expected a nested array as target type!")

    size = 1
    t = target_class
    while t.Size != Dynamic:
        size *= t.Size
        t = t.Value

    if len(array) % size != 0:
        raise Exception('unravel(): input array length must be '
                        'divisible by %i!' % size)

    indices = arange(_ek.uint32_array_t(type(array)), len(array) // size)
    return gather(target_class, array, indices)


def slice(value, index=-1, return_type=None):
    t = type(value)
    if _ek.array_depth_v(t) > 1 or issubclass(t, tuple) or issubclass(t, list):
        size = len(value)
        result = [None] * size
        for i in range(size):
            result[i] = _ek.slice(value[i], index)
        return result
    elif _ek.is_enoki_struct_v(a):
        if return_type == None:
            raise Exception('slice(): return type should be specified for enoki struct!')
        result = return_type()
        for k in type(value).ENOKI_STRUCT.keys():
            setattr(result, k, _ek.slice(getattr(value, k), index))
        return result
    elif _ek.is_dynamic_array_v(value):
        if index == -1:
            if _ek.width(value) > 1:
                raise Exception('slice(): variable contains more than a single entry!')
            index = 0
        return value.entry_(index)
    else:
        if index == 0:
            raise Exception('slice(): index out of bound!')
        return value


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


def op_radd(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return b.add_(a)


def op_iadd(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.iadd_(b)


def op_sub(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sub_(b)


def op_rsub(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return b.sub_(a)


def op_isub(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.isub_(b)


def op_mul(a, b):
    if type(a) is not type(b):
        scalar_multp = isinstance(b, int) or (a.IsFloat and isinstance(b, float))
        value_type_a = a.Value.Value if a.IsMatrix else a.Value
        value_type_b = None

        if _ek.is_array_v(b):
            value_type_b = b.Value.Value if b.IsMatrix else b.Value

        if scalar_multp or value_type_a is type(b):
            pass
        elif value_type_b is type(a):
            a, b = b, a
        else:
            a, b = _var_promote(a, b)

    return a.mul_(b)


def op_rmul(a, b):
    if type(a) is not type(b):
        if isinstance(b, int) or (a.IsFloat and isinstance(b, float)):
            pass
        else:
            a, b = _var_promote(a, b)
    return a.mul_(b)


def op_matmul(a, b):
    if a.IsMatrix and not (_ek.is_vector_v(b) or _ek.is_matrix_v(b)):
        return a * b
    else:
        return a.matmul_(b)


def op_imul(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.imul_(b)


def op_truediv(a, b):
    if not a.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Enoki integer arrays.")
    if getattr(b, 'Depth', 0) < a.Depth:
        return a * rcp(b)

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.truediv_(b)


def op_rtruediv(a, b):
    if (isinstance(b, float) or isinstance(b, int)) and b == 1:
        return rcp(a)

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return b.truediv_(a)


def op_itruediv(a, b):
    if not a.IsFloat:
        raise Exception("Use the floor division operator \"//\" for "
                        "Enoki integer arrays.")

    if isinstance(b, float) or isinstance(b, int):
        a *= 1.0 / b
        return a

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.itruediv_(b)


def op_floordiv(a, b):
    if not a.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Enoki floating point arrays.")

    if isinstance(b, int):
        if b == 0:
            raise Exception("Division by zero!")
        elif b == 1:
            return a

        # Division via multiplication by magic number + shift
        multiplier, shift = _ek.detail.idiv(a.Type, b)

        if a.IsSigned:
            q = mulhi(multiplier, a) + a
            q_sign = q >> (a.Type.Size * 8 - 1)
            q = q + (q_sign & ((1 << shift) - (1 if multiplier == 0 else 0)))
            sign = type(a)(-1 if b < 0 else 0)
            return ((q >> shift) ^ sign) - sign
        else:
            if multiplier == 0:
                return a >> (shift + 1)

            q = _ek.mulhi(multiplier, a)
            t = ((a - q) >> 1) + q
            return t >> shift

    if type(a) is not type(b):
        a, b = _var_promote(a, b)

    return a.floordiv_(b)


def op_ifloordiv(a, b):
    if not a.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Enoki floating point arrays.")

    if isinstance(b, int):
        a.assign(a // b)
        return a

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.ifloordiv_(b)


def op_rfloordiv(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return b.floordiv_(a)


def op_mod(a, b):
    if not a.IsIntegral:
        raise Exception("The modulo operator only supports integral arrays!")

    if isinstance(b, int):
        if not a.IsSigned and b != 0 and (b & (b - 1)) == 0:
            return a & (b - 1)
        else:
            return a - (a // b) * b

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.mod_(b)


def op_imod(a, b):
    if not a.IsIntegral:
        raise Exception("The modulo operator only supports integral arrays!")

    if isinstance(b, int):
        a.assign(a % b)
        return a

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.imod_(b)


def op_and(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)

    return a.and_(b)


def op_rand(a, b):
    if type(a) is not type(b) and type(a) is not _ek.mask_t(b):
        b, a = _var_promote_mask(b, a)
    return b.and_(a)


def op_iand(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)

    return a.iand_(b)


def and_(a, b):
    if type(a) is bool and type(b) is bool:
        return a and b
    else:
        return a & b


def op_or(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.or_(b)


def op_ror(a, b):
    if type(a) is not type(b) and type(a) is not _ek.mask_t(b):
        b, a = _var_promote_mask(b, a)
    return b.or_(a)


def op_ior(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.ior_(b)


def or_(a, b):
    if type(a) is bool and type(b) is bool:
        return a or b
    else:
        return a | b


def op_xor(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.xor_(b)


def op_rxor(a, b):
    if type(a) is not type(b) and type(a) is not _ek.mask_t(b):
        b, a = _var_promote_mask(b, a)
    return b.xor_(a)


def op_ixor(a, b):
    if type(a) is not type(b) and type(b) is not _ek.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.ixor_(b)


def op_lshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sl_(b)


def op_rlshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return b.sl_(a)


def op_ilshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.isl_(b)


def op_rshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.sr_(b)


def op_rrshift(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return b.sr_(a)


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
    if isinstance(a, ArrayBase) or isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.eq_(b)
    else:
        return a == b


def neq(a, b):
    if isinstance(a, ArrayBase) or isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.neq_(b)
    else:
        return a != b


def op_abs(a):
    if isinstance(a, ArrayBase):
        return a.abs_()
    else:
        return _builtins.abs(a)


def abs(a):
    if isinstance(a, ArrayBase):
        return a.abs_()
    else:
        return _builtins.abs(a)


def floor(a):
    if isinstance(a, ArrayBase):
        return a.floor_()
    else:
        return _math.floor(a)


def ceil(a):
    if isinstance(a, ArrayBase):
        return a.ceil_()
    else:
        return _math.ceil(a)


def round(a):
    if isinstance(a, ArrayBase):
        return a.round_()
    else:
        return _math.round(a)


def trunc(a):
    if isinstance(a, ArrayBase):
        return a.trunc_()
    else:
        return _math.trunc(a)


def mulhi(a, b):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.mulhi_(b)
    else:
        raise Exception("mulhi(): undefined for Python integers!")


def sqr(a):
    return a * a


def sqrt(a):
    if isinstance(a, ArrayBase):
        return a.sqrt_()
    else:
        return _math.sqrt(a)


def rcp(a):
    if isinstance(a, ArrayBase):
        return a.rcp_()
    else:
        return 1.0 / a


def rsqrt(a):
    if isinstance(a, ArrayBase):
        return a.rsqrt_()
    else:
        return 1.0 / _math.sqrt(a)


def max(a, b):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.max_(b)
    else:
        return _builtins.max(a, b)


def min(a, b):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.min_(b)
    else:
        return _builtins.min(a, b)


def fmadd(a, b, c):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase) or \
       isinstance(c, ArrayBase):
        if type(a) is not type(b) or type(b) is not type(c):
            a, b, c = _var_promote(a, b, c)
        return a.fmadd_(b, c)
    else:
        return _ek.detail.fmadd_scalar(a, b, c)


def fmsub(a, b, c):
    return fmadd(a, b, -c)


def fnmadd(a, b, c):
    return fmadd(-a, b, c)


def fnmsub(a, b, c):
    return fmadd(-a, b, -c)


def select(m, t, f):
    if isinstance(m, bool):
        return t if m else f
    type_t, type_f, type_m = type(t), type(f), type(m)

    if type_t is not type_f or type_m is not _ek.mask_t(t):
        if type_t is type_f and _ek.is_enoki_struct_v(type_t):
            result = type_t()
            for k in type_t.ENOKI_STRUCT.keys():
                setattr(result, k, select(m, getattr(t, k), getattr(f, k)))
            return result

        m, t, f = _var_promote_select(m, t, f)

    return type(t).select_(m, t, f)


def sign(a):
    t = type(a)
    return select(a >= 0, t(1), t(-1))


def copysign(a, b):
    a_a = abs(a)
    return select(b >= 0, a_a, -a_a)


def copysign_neg(a, b):
    a_a = abs(a)
    return select(b >= 0, a_a, -a_a)


def mulsign(a, b):
    return select(b >= 0, a, -a)


def mulsign_neg(a, b):
    return select(b >= 0, -a, a)


def isnan(a):
    if _ek.is_array_v(a):
        return ~eq(a, a)
    else:
        return not (a == a)


def isinf(a):
    return eq(abs(a), _ek.Infinity)


def isfinite(a):
    return abs(a) < _ek.Infinity


def lerp(a, b, t):
    return fmadd(b, t, fnmadd(a, t, a))


def clamp(value, min, max):
    return _ek.max(_ek.min(value, max), min)


def arg(value):
    if _ek.is_complex_v(value):
        return _ek.atan2(value.imag, value.real)
    else:
        return _ek.select(value >= 0, 0, -_ek.Pi)


def real(value):
    if _ek.is_complex_v(value):
        return value[0]
    elif _ek.is_quaternion_v(value):
        return value[3]
    else:
        return value


def imag(value):
    if _ek.is_complex_v(value):
        return value[1]
    elif _ek.is_quaternion_v(value):
        name = _ek.detail.array_name('Array', value.Type, (3), value.IsScalar)
        Array3f = getattr(_modules.get(value.__module__), name)
        return Array3f(value[0], value[1], value[2])
    else:
        return type(value)(0)


def tzcnt(a):
    if isinstance(a, ArrayBase):
        return a.tzcnt_()
    else:
        # The following assumes that 'a' is a 32 bit integer
        assert a >= 0 and a <= 0xFFFFFFFF
        result = 32
        while a & 0xFFFFFFFF:
            result -= 1
            a <<= 1
        return result


def lzcnt(a):
    if isinstance(a, ArrayBase):
        return a.lzcnt_()
    else:
        # The following assumes that 'a' is a 32 bit integer
        assert a >= 0 and a <= 0xFFFFFFFF
        result = 32
        while a:
            result -= 1
            a >>= 1
        return result


def popcnt(a):
    if isinstance(a, ArrayBase):
        return a.popcnt_()
    else:
        result = 0
        while a:
            result += a & 1
            a >>= 1
        return result


def log2i(a):
    if isinstance(a, ArrayBase):
        return (a.Type.Size * 8 - 1) - lzcnt(a)
    else:
        return 31 - lzcnt(a)


# -------------------------------------------------------------------
#   "Safe" functions that avoid domain errors due to rounding
# -------------------------------------------------------------------


def safe_sqrt(a):
    result = sqrt(max(a, 0))
    if _ek.is_diff_array_v(a) and _ek.grad_enabled(a):
        alt = sqrt(max(a, _ek.Epsilon(a)))
        result = _ek.replace_grad(result, alt)
    return result


def safe_cbrt(a):
    result = cbrt(max(a, 0))
    if _ek.is_diff_array_v(a) and _ek.grad_enabled(a):
        alt = cbrt(max(a, _ek.Epsilon(a)))
        result = _ek.replace_grad(result, alt)
    return result


def safe_asin(a):
    result = asin(clamp(a, -1, 1))
    if _ek.is_diff_array_v(a) and _ek.grad_enabled(a):
        alt = asin(clamp(a, -_ek.OneMinusEpsilon(a), _ek.OneMinusEpsilon(a)))
        result = _ek.replace_grad(result, alt)
    return result


def safe_acos(a):
    result = acos(clamp(a, -1, 1))
    if _ek.is_diff_array_v(a) and _ek.grad_enabled(a):
        alt = acos(clamp(a, -_ek.OneMinusEpsilon(a), _ek.OneMinusEpsilon(a)))
        result = _ek.replace_grad(result, alt)
    return result


# -------------------------------------------------------------------
#       Vertical operations -- AD/JIT compilation-related
# -------------------------------------------------------------------


def label(a):
    if isinstance(a, ArrayBase):
        return a.label_()
    else:
        return None


def set_label(a, label):
    if _ek.is_jit_array_v(a) or _ek.is_diff_array_v(a):
        a.set_label_(label)
    elif _ek.is_enoki_struct_v(a):
        for k in a.ENOKI_STRUCT.keys():
            set_label(getattr(a, k), label + "_" + k)


def schedule(*args):
    result = False
    for a in args:
        t = type(a)
        if issubclass(t, ArrayBase):
            result |= a.schedule_()
        elif _ek.is_enoki_struct_v(t):
            for k in t.ENOKI_STRUCT.keys():
                result |= schedule(getattr(a, k))
        elif issubclass(t, tuple) or issubclass(t, list):
            for v in a:
                result |= schedule(v)
        elif issubclass(t, _Mapping):
            for k, v in a.items():
                result |= schedule(v)
    return result


def eval(*args):
    if schedule(*args) or len(args) == 0:
        _ek.detail.eval()


def graphviz_str(arg):
    base = _ek.leaf_array_t(arg)

    if _ek.is_diff_array_v(base):
        return base.graphviz_()
    elif _ek.is_jit_array_v(base):
        return _ek.detail.graphviz()
    else:
        raise Exception('graphviz_str(): only variables registered with '
                        'the JIT (LLVM/CUDA) or AD backend are supported!')


def graphviz(arg):
    try:
        from graphviz import Source
        return Source(graphviz_str(arg))
    except ImportError:
        raise Exception('The "graphviz" Python package not available! Install '
                        'via "python -m pip install graphviz". Alternatively, '
                        'you can call enoki.graphviz_str() function to obtain '
                        'a string representation.')


def migrate(a, type_):
    if _ek.is_jit_array_v(a):
        return a.migrate_(type_)
    else:
        return a

# -------------------------------------------------------------------
#           Vertical operations -- transcendental functions
# -------------------------------------------------------------------


def sin(a):
    if isinstance(a, ArrayBase):
        return a.sin_()
    else:
        return _math.sin(a)


def cos(a):
    if isinstance(a, ArrayBase):
        return a.cos_()
    else:
        return _math.cos(a)


def sincos(a):
    if isinstance(a, ArrayBase):
        return a.sincos_()
    else:
        return (_math.sin(a), _math.cos(a))


def tan(a):
    if isinstance(a, ArrayBase):
        return a.tan_()
    else:
        return _math.tan(a)


def csc(a):
    if isinstance(a, ArrayBase):
        return a.csc_()
    else:
        return 1 / _math.sin(a)


def sec(a):
    if isinstance(a, ArrayBase):
        return a.sec_()
    else:
        return 1 / _math.sec(a)


def cot(a):
    if isinstance(a, ArrayBase):
        return a.cot_()
    else:
        return 1.0 / _math.tan(a)


def asin(a):
    if isinstance(a, ArrayBase):
        return a.asin_()
    else:
        return _math.asin(a)


def acos(a):
    if isinstance(a, ArrayBase):
        return a.acos_()
    else:
        return _math.acos(a)


def atan(a):
    if isinstance(a, ArrayBase):
        return a.atan_()
    else:
        return _math.atan(a)


def atan2(a, b):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.atan2_(b)
    else:
        return _math.atan2(a, b)


def exp(a):
    if isinstance(a, ArrayBase):
        return a.exp_()
    else:
        return _math.exp(a)


def exp2(a):
    if isinstance(a, ArrayBase):
        return a.exp2_()
    else:
        return _math.exp(a * _math.log(2.0))


def log(a):
    if isinstance(a, ArrayBase):
        return a.log_()
    else:
        return _math.log(a)


def log2(a):
    if isinstance(a, ArrayBase):
        return a.log2_()
    else:
        return _math.log2(a)


def pow(a, b):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b) and not \
           (isinstance(b, int) or isinstance(b, float)):
            a, b = _var_promote(a, b)
        return a.pow_(b)
    else:
        return _math.pow(a, b)


def op_pow(a, b):
    return pow(a, b)


def cbrt(a):
    if isinstance(a, ArrayBase):
        return a.cbrt_()
    else:
        return _math.pow(a, 1.0 / 3.0)


def erf(a):
    if isinstance(a, ArrayBase):
        return a.erf_()
    else:
        return _math.erf(a)


def erfinv(a):
    if isinstance(a, ArrayBase):
        return a.erfinv_()
    else:
        return _math.erfinv(a)


def lgamma(a):
    if isinstance(a, ArrayBase):
        return a.lgamma_()
    else:
        return _math.lgamma(a)


def tgamma(a):
    if isinstance(a, ArrayBase):
        return a.tgamma_()
    else:
        return _math.gamma(a)


def sinh(a):
    if isinstance(a, ArrayBase):
        return a.sinh_()
    else:
        return _math.sinh(a)


def cosh(a):
    if isinstance(a, ArrayBase):
        return a.cosh_()
    else:
        return _math.cosh(a)


def sincosh(a):
    if isinstance(a, ArrayBase):
        return a.sincosh_()
    else:
        return (_math.sinh(a), _math.cosh(a))


def tanh(a):
    if isinstance(a, ArrayBase):
        return a.tanh_()
    else:
        return _math.tanh(a)


def csch(a):
    return 1 / sinh(a)


def sech(a):
    return 1 / cosh(a)


def coth(a):
    return 1 / tanh(a)


def asinh(a):
    if isinstance(a, ArrayBase):
        return a.asinh_()
    else:
        return _math.asinh(a)


def acosh(a):
    if isinstance(a, ArrayBase):
        return a.acosh_()
    else:
        return _math.acosh(a)


def atanh(a):
    if isinstance(a, ArrayBase):
        return a.atanh_()
    else:
        return _math.atanh(a)


def rad_to_deg(a):
    return a * (180.0 / _ek.Pi)


def deg_to_rad(a):
    return a * (_ek.Pi / 180.0)


# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def shuffle(perm, value):
    if not _ek.is_array_v(value) or len(perm) != value.Size:
        raise Exception("shuffle(): incompatible input!")

    result = type(value)()
    for i, j in enumerate(perm):
        result[i] = value[j]

    return result


def compress(mask):
    if not _ek.is_mask_v(mask) or not mask.IsDynamic:
        raise Exception("compress(): incompatible input!")
    return mask.compress_()


def count(a):
    if _var_is_enoki(a):
        if a.Type != VarType.Bool:
            raise Exception("count(): input array must be a mask!")
        return a.count_()
    elif isinstance(a, bool):
        return 1 if a else 0
    elif _ek.is_iterable_v(a):
        result = 0
        for index, value in enumerate(a):
            if index == 0:
                result = ek.select(value, 0, 1)
            else:
                result = result + ek.select(value, 1, 0)
        return result
    else:
        raise Exception("count(): input must be a boolean or an "
                        "iterable containing masks!")


def all(a):
    if _var_is_enoki(a):
        if a.Type != VarType.Bool:
            raise Exception("all(): input array must be a mask!")
        return a.all_()
    elif isinstance(a, bool):
        return a
    elif _ek.is_iterable_v(a):
        result = True
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = result & value
        return result
    else:
        raise Exception("all(): input must be a boolean or an "
                        "iterable containing masks!")


def all_nested(a):
    while True:
        b = all(a)
        if b is a:
            break
        a = b
    return a


def all_or(value, a):
    assert isinstance(value, bool)
    if _ek.is_jit_array_v(a) and a.Depth == 1:
        return value
    else:
        return _ek.all(a)


def any(a):
    if _var_is_enoki(a):
        if a.Type != VarType.Bool:
            raise Exception("any(): input array must be a mask!")
        return a.any_()
    elif isinstance(a, bool):
        return a
    elif _ek.is_iterable_v(a):
        result = False
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = result | value
        return result
    else:
        raise Exception("any(): input must be a boolean or an "
                        "iterable containing masks!")


def any_nested(a):
    while True:
        b = any(a)
        if b is a:
            break
        a = b
    return a


def any_or(value, a):
    assert isinstance(value, bool)
    if _ek.is_jit_array_v(a) and a.Depth == 1:
        return value
    else:
        return _ek.any(a)


def none(a):
    b = any(a)
    return not b if isinstance(b, bool) else ~b


def none_nested(a):
    b = any_nested(a)
    return not b if isinstance(b, bool) else ~b


def none_or(value, a):
    assert isinstance(value, bool)
    if _ek.is_jit_array_v(a) and a.Depth == 1:
        return value
    else:
        return _ek.none(a)


def hsum(a):
    if _var_is_enoki(a):
        return a.hsum_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _ek.is_iterable_v(a):
        result = 0
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = result + value
        return result
    else:
        raise Exception("hsum(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def hsum_async(a):
    if _var_is_enoki(a) and hasattr(a, 'hsum_async_'):
        return a.hsum_async_()
    else:
        return type(a)([a.hsum_()])


def hsum_nested(a):
    while True:
        b = hsum(a)
        if b is a:
            break
        a = b
    return a


def hprod(a):
    if _var_is_enoki(a):
        return a.hprod_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _ek.is_iterable_v(a):
        result = 1
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = result * value
        return result
    else:
        raise Exception("hprod(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def hprod_async(a):
    if _var_is_enoki(a) and hasattr(a, 'hprod_async_'):
        return a.hprod_async_()
    else:
        return type(a)([a.hprod_()])


def hprod_nested(a):
    while True:
        b = hprod(a)
        if b is a:
            break
        a = b
    return a


def hmax(a):
    if _var_is_enoki(a):
        return a.hmax_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _ek.is_iterable_v(a):
        result = None
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = _ek.max(result, value)
        if result is None:
            raise Exception("hmax(): zero-sized array!")
        return result
    else:
        raise Exception("hmax(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def hmax_async(a):
    if _var_is_enoki(a) and hasattr(a, 'hmax_async_'):
        return a.hmax_async_()
    else:
        return type(a)([a.hmax_()])


def hmax_nested(a):
    while True:
        b = hmax(a)
        if b is a:
            break
        a = b
    return a


def hmin(a):
    if _var_is_enoki(a):
        return a.hmin_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _ek.is_iterable_v(a):
        result = None
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = _ek.min(result, value)
        if result is None:
            raise Exception("hmin(): zero-sized array!")
        return result
    else:
        raise Exception("hmin(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def hmin_async(a):
    if _var_is_enoki(a) and hasattr(a, 'hmin_async_'):
        return a.hmin_async_()
    else:
        return type(a)([a.hmin_()])


def hmin_nested(a):
    while True:
        b = hmin(a)
        if b is a:
            break
        a = b
    return a


def dot(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.dot_(b)


def dot_async(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)

    elif hasattr(a, 'dot_async_'):
        return a.dot_async_(b)
    else:
        return type(a)(a.dot_(b))


def abs_dot(a, b):
    return abs(dot(a, b))


def abs_dot_async(a, b):
    return abs(dot_async(a, b))


def squared_norm(a):
    return dot(a, a)


def norm(a):
    return sqrt(dot(a, a))


def normalize(a):
    return a * rsqrt(squared_norm(a))


def conj(a):
    if _ek.is_complex_v(a):
        return type(a)(a.real, -a.imag)
    elif _ek.is_quaternion_v(a):
        return type(a)(-a.x, -a.y, -a.z, a.w)
    else:
        return a


def hypot(a, b):
    a, b = abs(a), abs(b)
    maxval = _ek.max(a, b)
    minval = _ek.min(a, b)
    ratio = minval / maxval
    inf = _ek.Infinity

    return _ek.select(
        (a < inf) & (b < inf) & (ratio < inf),
        maxval * _ek.sqrt(_ek.fmadd(ratio, ratio, 1)),
        a + b
    )


def tile(array, count: int):
    if not _ek.is_array_v(array) or not isinstance(count, int):
        raise("tile(): invalid input types!")
    elif not array.IsDynamic:
        raise("tile(): first input argument must be a dynamic Enoki array!")

    size = len(array)
    t = type(array)

    if array.Depth > 1:
        result = t()

        if array.Size == Dynamic:
            result.init_(size)

        for i in range(size):
            result[i] = tile(array[i], count)

        return result
    else:
        index = _ek.arange(_ek.uint_array_t(t), size * count) % size
        return _ek.gather(t, array, index)


def repeat(array, count: int):
    if not _ek.is_array_v(array) or not isinstance(count, int):
        raise("tile(): invalid input types!")
    elif not array.IsDynamic:
        raise("tile(): first input argument must be a dynamic Enoki array!")

    size = len(array)
    t = type(array)

    if array.Depth > 1:
        result = t()

        if array.Size == Dynamic:
            result.init_(size)

        for i in range(size):
            result[i] = repeat(array[i], count)

        return result
    else:
        index = _ek.arange(_ek.uint_array_t(t), size * count) // count
        return _ek.gather(t, array, index)


def meshgrid(*args, indexing='xy'):
    if indexing != "ij" and indexing != "xy":
        raise Exception("meshgrid(): 'indexing' argument must equal"
                        " 'ij' or 'xy'!")

    if len(args) == 0:
        return ()
    elif len(args) == 1:
        return args[0]

    t = type(args[0])
    for v in args:
        if not _ek.is_dynamic_array_v(v) or \
           _ek.array_depth_v(v) != 1 or \
           type(v) is not t:
            raise Exception("meshgrid(): consistent 1D dynamic arrays expected!")

    size = _ek.hprod((len(v) for v in args))
    index_t = _ek.uint32_array_t(t)
    index = _ek.arange(index_t, size)

    result = []
    if indexing == "xy":
        args = reversed(args)

    for v in args:
        size //= len(v)
        index_v = index // size
        index = index - index_v * size
        result.append(_ek.gather(t, v, index_v))

    if indexing == "xy":
        result = reversed(result)

    return tuple(result)


def block_sum(value, block_size):
    if _ek.is_jit_array_v(value):
        return value.block_sum_(block_size)
    else:
        raise Exception("block_sum(): requires a JIT array!")


def binary_search(start, end, pred):
    assert isinstance(start, int) and isinstance(end, int)

    iterations = log2i(end - start) + 1 if start < end else 0

    for i in range(iterations):
        middle = (start + end) >> 1

        cond = pred(middle)
        start = _ek.select(cond, _ek.min(middle + 1, end), start)
        end = _ek.select(cond, end, middle)

    return start


# -------------------------------------------------------------------
#    Transformations, matrices, operations for 3D vector spaces
# -------------------------------------------------------------------

def cross(a, b):
    if _ek.array_size_v(a) != 3 or _ek.array_size_v(a) != 3:
        raise Exception("cross(): requires 3D input arrays!")

    ta, tb = type(a), type(b)

    return fmsub(ta(a.y, a.z, a.x), tb(b.z, b.x, b.y),
                 ta(a.z, a.x, a.y) * tb(b.y, b.z, b.x))


# -------------------------------------------------------------------
#                     Automatic differentiation
# -------------------------------------------------------------------


def detach(a, preserve_type=False):
    if _ek.is_diff_array_v(a):
        if preserve_type:
            return type(a)(a.detach_())
        else:
            return a.detach_()
    elif _ek.is_enoki_struct_v(a):
        result = type(a)()
        for k in type(a).ENOKI_STRUCT.keys():
            setattr(result, k, detach(getattr(a, k), preserve_type=preserve_type))
        return result
    else:
        return a


def grad(a):
    if _ek.is_diff_array_v(a):
        return a.grad_()
    elif _ek.is_enoki_struct_v(a):
        result = type(a)()
        for k in type(a).ENOKI_STRUCT.keys():
            setattr(result, k, grad(getattr(a, k)))
        return result
    elif isinstance(a, tuple) or isinstance(a, list):
        return type(a)([grad(v) for v in a])
    elif isinstance(a, _Mapping):
        return {k : grad(v) for k, v in a.items()}
    else:
        return _ek.zero(type(a))


def set_grad(a, value):
    if _ek.is_diff_array_v(a) and a.IsFloat:
        if _ek.is_diff_array_v(value):
            value = _ek.detach(value)

        t = _ek.detached_t(a)
        if type(value) is not t:
            value = t(value)

        a.set_grad_(value)
    elif _ek.is_enoki_struct_v(a):
        for k in type(a).ENOKI_STRUCT.keys():
            set_grad(getattr(a, k), value)


def accum_grad(a, value):
    if _ek.is_diff_array_v(a) and a.IsFloat:
        if _ek.is_diff_array_v(value):
            value = _ek.detach(value)

        t = _ek.detached_t(a)
        if type(value) is not t:
            value = t(value)

        a.accum_grad_(value)
    elif _ek.is_enoki_struct_v(a):
        for k in type(a).ENOKI_STRUCT.keys():
            accum_grad(getattr(a, k), value)


def grad_enabled(a):
    result = False
    if _ek.is_diff_array_v(a):
        result = a.grad_enabled_()
    elif _ek.is_enoki_struct_v(a):
        for k in type(a).ENOKI_STRUCT.keys():
            result |= grad_enabled(getattr(a, k))
    elif isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            result |= grad_enabled(v)
    elif isinstance(a, _Mapping):
        for k, v in a.items():
            result |= grad_enabled(v)
    return result


def set_grad_enabled(a, value):
    if _ek.is_diff_array_v(a) and a.IsFloat:
        a.set_grad_enabled_(value)
    elif _ek.is_enoki_struct_v(a):
        for k in type(a).ENOKI_STRUCT.keys():
            set_grad_enabled(getattr(a, k), value)
    elif isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            set_grad_enabled(v, value)
    elif isinstance(a, _Mapping):
        for k, v in a.items():
            set_grad_enabled(v, value)


def enable_grad(*args):
    for v in args:
        set_grad_enabled(v, True)


def disable_grad(*args):
    for v in args:
        set_grad_enabled(v, False)


def grad_suspended(a):
    result = False
    if _ek.is_diff_array_v(a):
        result = a.grad_suspended_()
    elif _ek.is_enoki_struct_v(a):
        for k in type(a).ENOKI_STRUCT.keys():
            result |= grad_suspended(getattr(a, k))
    elif isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            result |= grad_suspended(v)
    elif isinstance(a, _Mapping):
        for k, v in a.items():
            result |= grad_suspended(v)
    return result


def set_grad_suspended(a, value):
    if _ek.is_diff_array_v(a) and a.IsFloat:
        a.set_grad_suspended_(value)
    elif _ek.is_enoki_struct_v(a):
        for k in type(a).ENOKI_STRUCT.keys():
           set_grad_suspended(getattr(a, k), value)
    elif isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            set_grad_suspended(v, value)
    elif isinstance(a, _Mapping):
        for k, v in a.items():
            set_grad_suspended(v, value)


def suspend_grad(*args):
    for v in args:
        set_grad_suspended(v, True)


def resume_grad(*args):
    for v in args:
        set_grad_suspended(v, False)


def replace_grad(a, b):
    if type(a) is not type(b) or not _ek.is_diff_array_v(a):
        raise Exception("replace_grad(): unsupported input types!")

    if a.Depth > 1:
        size = _builtins.max(len(a), len(b))
        result = type(a)()
        if a.Size == Dynamic:
            result.init_(size)
        for i in range(size):
            result[i] = replace_grad(a[i], b[i])
        return result
    else:
        return type(a).create_(b.index_ad(), a.detach_())


def enqueue(*args):
    for a in args:
        if _ek.is_diff_array_v(a):
            a.enqueue_()


def traverse(t, reverse=True, retain_graph=False):
    if not _ek.is_diff_array_v(t):
        raise Exception('traverse(): expected a differentiable array type!')

    _ek.leaf_array_t(t).traverse_(reverse, retain_graph)


def ad_clear(a):
    t = _ek.leaf_array_t(a)
    if _ek.is_diff_array_v(t):
        t.ad_clear_()
    else:
        raise Exception("Expected a differentiable array type!")


def backward(a, retain_graph=False):
    if _ek.is_diff_array_v(a):
        if not grad_enabled(a):
            raise Exception("backward(): attempted to propagate derivatives "
                            "through a variable that is not registered with "
                            "the AD backend. Did you forget to call "
                            "enable_grad()?")
        set_grad(a, 1)
        a.enqueue_()
        traverse(type(a), reverse=True, retain_graph=retain_graph)
    else:
        raise Exception("Expected a differentiable array type!")


def forward(a, retain_graph=False):
    if _ek.is_diff_array_v(a):
        if not grad_enabled(a):
            raise Exception("forward(): attempted to propagate derivatives "
                            "through a variable that is not registered with "
                            "the AD backend. Did you forget to call "
                            "enable_grad()?")
        set_grad(a, 1)
        a.enqueue_()
        traverse(type(a), reverse=False, retain_graph=retain_graph)
    else:
        raise Exception("Expected a differentiable array type!")

# -------------------------------------------------------------------
#                      Initialization operations
# -------------------------------------------------------------------


def zero(type_, size=1):
    if not isinstance(type_, type):
        raise Exception('zero(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.zero_(size)
    elif _ek.is_enoki_struct_v(type_):
        result = type_()
        for k, v in type_.ENOKI_STRUCT.items():
            setattr(result, k, zero(v, size))
        if hasattr(type_, 'zero_'):
            result.zero_(size)
        return result
    elif not type_ in (int, float, complex, bool):
        return None
    else:
        return type_(0)


def empty(type_, size=1):
    if not isinstance(type_, type):
        raise Exception('empty(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.empty_(size)
    elif _ek.is_enoki_struct_v(type_):
        result = type_()
        for k, v in type_.ENOKI_STRUCT.items():
            setattr(result, k, empty(v, size))
        return result
    else:
        return type_(0)


def full(type_, value, size=1):
    if not isinstance(type_, type):
        raise Exception('full(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.full_(value, size)
    else:
        return type_(value)


def opaque(type_, value, size=1):
    if not isinstance(type_, type):
        raise Exception('opaque(): Type expected as first argument')
    if issubclass(type_, ArrayBase):
        return type_.opaque_(value, size)
    elif _ek.is_enoki_struct_v(type_):
        result = type_()
        for k, v in type_.ENOKI_STRUCT.items():
            setattr(result, k, opaque(v, getattr(value, k), size))
        return result
    else:
        return type_(value)


def make_opaque(*args):
    for a in args:
        t = type(a)
        if issubclass(t, ArrayBase):
            if _ek.array_depth_v(t) > 1:
                res = t()
                for i in range(a.Size):
                    tmp = a[i]
                    make_opaque(tmp)
                    res[i] = tmp
                a.assign(res)
            elif _ek.is_diff_array_v(t):
                make_opaque(a.detach_())
            elif _ek.is_jit_array_v(t):
                if not a.is_evaluated_():
                    a.assign(a.copy_())
                    a.data_()
        elif _ek.is_enoki_struct_v(t):
            for k in t.ENOKI_STRUCT.keys():
                make_opaque(getattr(a, k))
        elif issubclass(t, tuple) or issubclass(t, list):
            for v in a:
                make_opaque(v)
        elif issubclass(t, _Mapping):
            for k, v in a.items():
                make_opaque(v)


def linspace(type_, min, max, size=1, endpoint=True):
    if not isinstance(type_, type):
        raise Exception('linspace(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.linspace_(min, max, size, endpoint)
    else:
        return type_(min)


def arange(type_, start=None, end=None, step=1):
    if start is None:
        start = 0
        end = 1
    elif end is None:
        end = start
        start = 0

    if not isinstance(type_, type):
        raise Exception('arange(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.arange_(start, end, step)
    else:
        return type_(start)


def identity(type_, size=1):
    if _ek.is_special_v(type_):
        result = zero(type_, size)

        if type_.IsComplex or type_.IsQuaternion:
            result.real = identity(type_.Value, size)
        elif type_.IsMatrix:
            one = identity(type_.Value.Value, size)
            for i in range(type_.Size):
                result[i, i] = one
        return result
    elif _ek.is_array_v(type_):
        return full(type_, 1, size)
    else:
        return type_(1)

# -------------------------------------------------------------------
#                  Higher-level utility functions
# -------------------------------------------------------------------

def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    # Fast path for Enoki arrays, avoid for special array types
    # due to their non-standard broadcasting behavior
    if _ek.is_array_v(a) or _ek.is_array_v(b):
        if _ek.is_diff_array_v(a):
            a = _ek.detach(a)
        if _ek.is_diff_array_v(b):
            b = _ek.detach(b)

        if _ek.is_array_v(a) and not _ek.is_floating_point_v(a):
            a, _ = _var_promote(a, 1.0)
        if _ek.is_array_v(b) and not _ek.is_floating_point_v(b):
            b, _ = _var_promote(b, 1.0)

        if type(a) is not type(b):
            a, b = _var_promote(a, b)

        diff = abs(a - b)
        cond = diff <= abs(b) * rtol + _ek.full(type(diff), atol)
        if _ek.is_floating_point_v(a):
            cond |= _ek.eq(a, b)  # plus/minus infinity
        if equal_nan:
            cond |= _ek.isnan(a) & _ek.isnan(b)
        return _ek.all_nested(cond)

    def safe_len(x):
        try:
            return len(x)
        except TypeError:
            return 0

    def safe_getitem(x, xl, i):
        return x[i if xl > 1 else 0] if xl > 0 else x

    la, lb = safe_len(a), safe_len(b)
    size = max(la, lb)

    if la != size and la > 1 or lb != size and lb > 1:
        raise Exception("allclose(): size mismatch (%i vs %i)!" % (la, lb))
    elif size == 0:
        if equal_nan and _math.isnan(a) and _math.isnan(b):
            return True
        return abs(a - b) <= abs(b) * rtol + atol
    else:
        for i in range(size):
            ia = safe_getitem(a, la, i)
            ib = safe_getitem(b, lb, i)
            if not allclose(ia, ib, rtol, atol, equal_nan):
                return False
        return True


def printf_async(mask, fmt, *args):
    if not _ek.is_jit_array_v(mask) or not _ek.array_depth_v(mask) == 1 or not _ek.is_mask_v(mask):
        raise Exception("printf_async(): 'mask' argument must be boolean-valued depth-1 JIT array")
    indices = []
    for a in args:
        if not _ek.is_jit_array_v(a) or not _ek.array_depth_v(a) == 1:
            raise Exception("printf_async(): extra arguments must all be depth-1 JIT arrays")
        indices.append(a.index())
    _ek.detail.printf_async(mask.index(), fmt, indices)


# -------------------------------------------------------------------
#             Automatic differentation of custom fuctions
# -------------------------------------------------------------------

class CustomOp:
    def grad_out(self):
        return _ek.grad(self.output)

    def set_grad_out(self, value):
        _ek.accum_grad(self.output, value)

    def grad_in(self, name):
        if name not in self.inputs:
            raise Exception('Could not find input argument named \"%s\"!' % name)
        return _ek.grad(self.inputs[name])

    def set_grad_in(self, name, value):
        if name not in self.inputs:
            raise Exception('Could not find input argument named \"%s\"!' % name)
        _ek.accum_grad(self.inputs[name], value)

    def forward(self):
        raise Exception('CustomOp.forward(): not implemented')

    def backward(self):
        raise Exception('CustomOp.backward(): not implemented')

    def name(self):
        return "CustomOp[unnamed]"


def custom(cls, *args, **kwargs):
    # Extract indices of differentiable variables
    def diff_vars(o, indices):
        if _ek.array_depth_v(o) > 1 \
           or isinstance(o, list) \
           or isinstance(o, tuple):
            for v in o:
                diff_vars(v, indices)
        elif isinstance(o, _Mapping):
            for k, v in o.items():
                diff_vars(v, indices)
        elif _ek.is_diff_array_v(o) and _ek.grad_enabled(o):
            indices.append(o.index_ad())

    # Clear primal values of a differentiable array
    def clear_primal(o):
        if _ek.array_depth_v(o) > 1 \
           or isinstance(o, list) \
           or isinstance(o, tuple):
            return type(o)([clear_primal(v) for v in o])
        elif isinstance(o, _Mapping):
            return { k: clear_primal(v) for k, v in o.items() }
        elif _ek.is_diff_array_v(o):
            to = type(o)
            return to.create_(o.index_ad(), _ek.detached_t(to)())

    inst = cls()

    # Convert args to kwargs
    kwargs.update(zip(inst.eval.__code__.co_varnames[1:], args))

    output = inst.eval(**{ k: _ek.detach(v) for k, v in kwargs.items() })
    del args

    diff_vars_in = []
    diff_vars(kwargs, diff_vars_in)

    if len(diff_vars_in) > 0:
        output = _ek.diff_array_t(output)
        _ek.enable_grad(output)

        inst.inputs = clear_primal(kwargs)
        inst.output = clear_primal(output)

        diff_vars_out = []
        diff_vars(inst.output, diff_vars_out)

        if len(diff_vars_out) == 0:
            raise Exception("enoki.custom(): internal error!")

        Type = _ek.leaf_array_t(output)
        detail = _modules.get(Type.__module__ + ".detail")

        tmp_in, tmp_out = None, None

        if len(diff_vars_in) > 1:
            tmp_in = Type()
            _ek.enable_grad(tmp_in)
            _ek.set_label(tmp_in, inst.name() + "_in")
            for index in diff_vars_in:
                detail.ad_add_edge(index, tmp_in.index_ad())

        if len(diff_vars_out) > 1:
            tmp_out = Type()
            _ek.enable_grad(tmp_out)
            _ek.set_label(tmp_out, inst.name() + "_out")
            for index in diff_vars_out:
                detail.ad_add_edge(tmp_out.index_ad(), index)

        detail.ad_add_edge(
            diff_vars_in[0] if tmp_in is None else tmp_in.index_ad(),
            diff_vars_out[0] if tmp_out is None else tmp_out.index_ad(),
            inst
        )

    return output
