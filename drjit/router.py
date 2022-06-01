import drjit as _dr
from drjit import ArrayBase, VarType, Exception, Dynamic
from drjit.detail import array_name as _array_name
from sys import modules as _modules
import math as _math
import builtins as _builtins
from collections.abc import Mapping as _Mapping, \
                            Sequence as _Sequence

# -------------------------------------------------------------------
#                        Type promotion logic
# -------------------------------------------------------------------


def _var_is_drjit(a):
    return isinstance(a, ArrayBase)


def _var_type(a, preferred=VarType.Void):
    '''
    Return the VarType of a given Dr.Jit object or plain Python type. Return
    'preferred' when there is sufficient room for interpretation (e.g. when
    given an 'int').
    '''
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
        for t in [VarType.Float16, VarType.Float32, VarType.Float64,
                  VarType.Int32, VarType.Int64, VarType.UInt32, VarType.UInt64]:
            if t.NumPy == a.dtype:
                return t
    elif isinstance(a, _Sequence):
        return _builtins.max([_var_type(v, preferred) for v in a])
    elif isinstance(a, type(None)) or 'pybind11' in type(type(a)).__name__:
        return VarType.Pointer
    else:
        raise Exception("var_type(): Unsupported type!")


def _var_promote(*args):
    '''
    Given a list of Dr.Jit arrays and scalars, determine the flavor and shape of
    the result array and broadcast/convert everything into this form.
    '''
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
        elif getattr(a, 'IsTensor', False):
            base = a

    if base is None:
        raise Exception("At least one of the input arguments "
                        "must be an Dr.Jit array!")

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
    '''
    Like _var_promote(), but has a special case where 'a1' can be a mask.
    '''
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
    '''
    Like _var_promote(), but specially adapted to the select() operation
    '''
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

    if not modname.startswith('drjit.'):
        if cls.IsCUDA:
            modname = "drjit.cuda"
        elif cls.IsLLVM:
            modname = "drjit.llvm"
        elif cls.IsPacket:
            modname = "drjit.packet"
        else:
            modname = "drjit.scalar"

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


def shape(arg):
    '''
    shape(arg, /)
    Return a tuple describing dimension and shape of the provided Dr.Jit array
    or tensor.

    When the arrays is ragged, the implementation signals a failure by returning
    ``None``. A ragged array has entries of incompatible size, e.g. ``[[1, 2],
    [3, 4, 5]]``. Note that an scalar entries (e.g. ``[[1, 2], [3]]``) are
    acceptable, since broadcasting can effectively convert them to any size.

    The expressions ``drjit.shape(arg)`` and ``arg.shape`` are equivalent.

    Args:
        arg (drjit.ArrayBase): An arbitrary Dr.Jit array or tensor

    Returns:
        tuple | NoneType: A tuple describing the dimension and shape of the
        provided Dr.Jit input array or tensor. When the input array is *ragged*
        (i.e., when it contains components with mismatched sizes), the function
        returns ``None``.
    '''
    if _dr.is_tensor_v(arg):
        return arg.shape

    s = []
    if not _shape_impl(arg, 0, s):
        return None
    else:
        return s


def width(arg):
    r'''
    width(arg, /)
    Return the width of the provided dynamic Dr.Jit array, tensor, or
    :ref:`custom data structure <custom-struct>`.

    The function returns ``1`` if the input variable isn't a Dr.Jit array,
    tensor, or :ref:`custom data structure <custom-struct>`.

    Args:
        arg (drjit.ArrayBase): An arbitrary Dr.Jit array, tensor, or
          :ref:`custom data structure <custom-struct>`

    Returns:
        int: The dynamic width of the provided variable.
    '''
    if _dr.is_array_v(value):
        if _dr.is_tensor_v(value):
            return width(value.array)
        else:
            return shape(value)[-1]
    elif _dr.is_drjit_struct_v(value):
        result = 0
        for k in type(arg).DRJIT_STRUCT.keys():
            result = max(result, width(getattr(arg, k)))
        return result
    else:
        return 1


def resize(arg, size):
    r'''
    resize(arg, size)
    Resize in-place the provided Dr.Jit array, tensor, or
    :ref:`custom data structure <custom-struct>` to a new size.

    The provided variable must have a size of zero or one originally otherwise
    this function will fail.

    When the provided variable doesn't have a size of 1 and its size exactly
    matches ``size`` the function does nothing. Otherwise, it fails.

    Args:
        arg (drjit.ArrayBase): An arbitrary Dr.Jit array, tensor, or
          :ref:`custom data structure <custom-struct>` to be resized

        size (int): The new size
    '''
    if _dr.array_depth_v(arg) > 1:
        for i in range(arg.Size):
            resize(arg[i], size)
    elif _dr.is_jit_array_v(arg):
        arg.resize_(size)
    elif _dr.is_drjit_struct_v(arg):
        for k in type(arg).DRJIT_STRUCT.keys():
            resize(getattr(arg, k), size)


def device(value=None):
    if value is None:
        return _dr.detail.device()
    elif _dr.array_depth_v(value) > 1:
        return device(value[0])
    elif _dr.is_diff_array_v(value):
        return device(_dr.detach(value))
    elif _dr.is_jit_array_v(value):
        return _dr.detail.device(value.index())
    else:
        return -1


# By default, don't print full contents of arrays with more than 20 entries
_print_threshold = 20


def _repr_impl(self, shape, buf, *idx):
    '''Implementation detail of op_repr()'''
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
        if self.IsTensor:
            return f"{type(self).__name__}(shape={s})"

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
        "To convert an Dr.Jit array into a boolean value, use a mask reduction "
        "operation such as drjit.all(), drjit.any(), drjit.none(). Special "
        "variants (drjit.all_nested(), etc.) are available for nested arrays.")


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
            if self.Depth > 1:
                return self.entry_ref_(index)
            else:
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
        result = _dr.empty(type(self), len(indices))
        for i in range(len(indices)):
            result[i] = op_getitem(self, indices[i])
        return result
    elif _dr.is_mask_v(index):
        return type(self)(self)
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
    elif _dr.is_mask_v(index):
        self.assign(_dr.select(index, value, self))
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


def reinterpret_array_v(dtype, value):
    '''
    Reinterpret the provided Dr.Jit array/tensor into a specified type.

    Args:
        dtype (type): Target type for the reinterpretation.

        value (object): Dr.Jit array or tensor to reinterpret.

    Returns:
        object: Result of the conversion as described above.
    '''
    assert isinstance(dtype, type)

    def reinterpret_array(dtype, value, vt_target, vt_value):
        if issubclass(dtype, ArrayBase):
            if hasattr(dtype, "reinterpret_array_"):
                return dtype.reinterpret_array_(value)
            else:
                result = dtype()
                if result.Size == Dynamic:
                    result.init_(len(value))

                for i in range(len(value)):
                    result[i] = reinterpret_array(dtype.Value, value[i],
                                                       dtype.Type, value.Type)

                return result
        else:
            return _dr.detail.reinterpret_scalar(value, vt_value, vt_target)

    return reinterpret_array(dtype, value, _dr.VarType.Void,  _dr.VarType.Void)


# -------------------------------------------------------------------
#                      Scatter/gather operations
# -------------------------------------------------------------------

def _broadcast_index(target_type, index):
    size = target_type.Size
    if _dr.array_size_v(index) <= 1 and size == Dynamic:
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


def gather(dtype, source, index, active=True):
    '''
    Gather values from a flat array or nested data structure

    This function performs a *gather* (i.e., indirect memory read) from
    ``source`` at position ``index``. It expects a ``dtype`` argument and will
    return an instance of this type. The optional ``active`` argument can be
    used to disable some of the components, which is useful when not all indices
    are valid; the corresponding output will be zero in this case.

    This operation can be used in the following different ways:

    1. When ``dtype`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
    this operation implements a parallelized version of the Python array
    indexing expression ``source[index]`` with optional masking. Example:

    .. code-block::

        source = dr.cuda.Float([...])
        index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
        result = dr.gather(dtype=type(source), source=source, index=index)

    2. When ``dtype`` is a more complex type (e.g. a :ref:`custom source structure <custom-struct>`,
        nested Dr.Jit array, tuple, list, dictionary, etc.), the behavior depends:

    - When ``type(source)`` matches ``dtype``, the the gather operation threads
        through entries and invokes itself recursively. For example, the gather
        operation in

        .. code-block::

            result = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            result = dr.gather(dr.cuda.Array3f, source, index)

        is equivalent to

        .. code-block::

            result = dr.cuda.Array3f(
                dr.gather(dr.cuda.Float, source.x, index),
                dr.gather(dr.cuda.Float, source.y, index),
                dr.gather(dr.cuda.Float, source.z, index)
            )

    - Otherwise, the operation reconstructs the requested ``dtype`` from a flat
        ``source`` array, using C-style ordering with a suitably modified
        ``index``. For example, the gather below reads 3D vectors from a 1D
        array.


        .. code-block::

            source = dr.cuda.Float([...])
            index = dr.cuda.UInt([...])
            result = dr.gather(dr.cuda.Array3f, source, index)

        and is equivalent to

        .. code-block::

            result = dr.cuda.Vector3f(
                dr.gather(dr.cuda.Float, source, index*3 + 0),
                dr.gather(dr.cuda.Float, source, index*3 + 1),
                dr.gather(dr.cuda.Float, source, index*3 + 2)
            )

    .. danger::

        The indices provided to this operation are unchecked. Out-of-bounds
        reads are undefined behavior (if not disabled via the ``active``
        parameter) and may crash the application. Negative indices are not
        permitted.

    Args:
        dtype (type): The desired output type (typically equal to ``type(source)``,
          but other variations are possible as well, see the description above.)

        source (object): The object from which data should be read (typically a
          1D Dr.Jit array, but other variations are possible as well, see the
          description above.)

        index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
          :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
          specifying gather indices. Dr.Jit will attempt an implicit conversion
          if another type is provided. active

        (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
          :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
          specifying active components. Dr.Jit will attempt an implicit
          conversion if another type is provided. The default is `True`.

    Returns:
        object: An instance of type ``dtype`` containing the result of the gather operation.
    '''
    if not isinstance(dtype, type):
        raise Exception('gather(): Type expected as first argument')
    elif not issubclass(dtype, ArrayBase):
        if _dr.is_drjit_struct_v(dtype):
            if type(source) is not dtype:
                raise Exception('gather(): type mismatch involving custom data structure!')
            result = dtype()
            for k, v in dtype.DRJIT_STRUCT.items():
                setattr(result, k,
                        gather(v, getattr(source, k), index, active))
            return result
        else:
            assert isinstance(index, int) and isinstance(active, bool)
            return source[index] if active else 0
    else:
        if _dr.is_tensor_v(dtype) or _dr.is_tensor_v(source):
            raise Exception("gather(): Tensor type not supported! Should work "
                            "with the underlying array instead. (e.g. tensor.array)")
        if source.Depth != 1:
            if source.Size != dtype.Size:
                raise Exception("gather(): mismatched source/target configuration!")

            result = dtype()
            for i in range(dtype.Size):
                result[i] = gather(dtype.Value, source[i], index, active)
            return result
        else:
            index_type = _dr.uint32_array_t(dtype)
            if not isinstance(index, index_type):
                index = _broadcast_index(index_type, index)

            active_type = index_type.MaskType
            if not isinstance(active, active_type):
                active = active_type(active)

            return dtype.gather_(source, index, active)


def scatter(target, value, index, active=True):
    '''
    Scatter values into a flat array or nested data structure

    This operation performs a *scatter* (i.e., indirect memory write) of the
    ``value`` parameter to the ``target`` array at position ``index``. The optional
    ``active`` argument can be used to disable some of the individual write
    operations, which is useful when not all provided values or indices are valid.

    This operation can be used in the following different ways:

    1. When ``target`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
    this operation implements a parallelized version of the Python array
    indexing expression ``target[index] = value`` with optional masking. Example:

    .. code-block::

        target = dr.empty(dr.cuda.Float, 1024*1024)
        value = dr.cuda.Float([...])
        index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
        dr.scatter(target, value=value, index=index)

    2. When ``target`` is a more complex type (e.g. a :ref:`custom source structure
    <custom-struct>`, nested Dr.Jit array, tuple, list, dictionary, etc.), the
    behavior depends:

    - When ``target`` and ``value`` are of the same type, the scatter operation
        threads through entries and invokes itself recursively. For example, the
        scatter operation in

        .. code-block::

            target = dr.cuda.Array3f(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter(target, value, index)

        is equivalent to

        .. code-block::

            dr.scatter(target.x, value.x, index)
            dr.scatter(target.y, value.y, index)
            dr.scatter(target.z, value.z, index)

    - Otherwise, the operation flattens the ``value`` array and writes it using
        C-style ordering with a suitably modified ``index``. For example, the
        scatter below writes 3D vectors into a 1D array.

        .. code-block::

            target = dr.cuda.Float(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter(target, value, index)

        and is equivalent to

        .. code-block::

            dr.scatter(target, value.x, index*3 + 0)
            dr.scatter(target, value.y, index*3 + 1)
            dr.scatter(target, value.z, index*3 + 2)

    .. danger::

        The indices provided to this operation are unchecked. Out-of-bounds writes
        are undefined behavior (if not disabled via the ``active`` parameter) and
        may crash the application. Negative indices are not permitted.

        Dr.Jit makes no guarantees about the expected behavior when a scatter
        operation has *conflicts*, i.e., when a specific position is written
        multiple times by a single :py:func:`drjit.scatter()` operation.

    Args:
        target (object): The object into which data should be written (typically
          a 1D Dr.Jit array, but other variations are possible as well, see the
          description above.)

        value (object): The values to be written (typically of type ``type(target)``,
          but other variations are possible as well, see the description above.)
          Dr.Jit will attempt an implicit conversion if the the input is not an
          array type.

        index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
          :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
          specifying gather indices. Dr.Jit will attempt an implicit conversion
          if another type is provided.

        active (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
          :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
          specifying active components. Dr.Jit will attempt an implicit
          conversion if another type is provided. The default is `True`.
    '''
    target_type = type(target)
    if not issubclass(target_type, ArrayBase):
        if _dr.is_drjit_struct_v(target_type):
            if type(value) is not target_type:
                raise Exception('scatter(): type mismatch involving custom data structure!')
            for k in target_type.DRJIT_STRUCT.keys():
                scatter(getattr(target, k), getattr(value, k),
                        index, active)
        else:
            assert isinstance(index, int) and isinstance(active, bool)
            if active:
                target[index] = value
    else:
        if _dr.is_tensor_v(target) or _dr.is_tensor_v(value):
            raise Exception("scatter(): Tensor type not supported! Should work "
                            "with the underlying array instead. (e.g. tensor.array)")
        if target.Depth != 1:
            if _dr.array_size_v(target) != _dr.array_size_v(value):
                raise Exception("scatter(): mismatched source/target configuration!")

            for i in range(len(target)):
                scatter(target.entry_ref_(i), value[i], index, active)
        else:
            if not _dr.is_array_v(value):
                value = target_type(value)

            index_type = _dr.uint32_array_t(type(value))
            if not isinstance(index, index_type):
                index = _broadcast_index(index_type, index)

            active_type = index_type.MaskType
            if not isinstance(active, active_type):
                active = active_type(active)

            return value.scatter_(target, index, active)


def scatter_reduce(op, target, value, index, active=True):
    '''
    Perform a read-modify-write operation on a flat array or nested data structure

    This function performs a read-modify-write operation of the ``value``
    parameter to the ``target`` array at position ``index``. The optional
    ``active`` argument can be used to disable some of the individual write
    operations, which is useful when not all provided values or indices are valid.

    The operation to be applied is defined by tje ``op`` argument (see
    :py:class:`drjit.ReduceOp`).

    This operation can be used in the following different ways:

    1. When ``target`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
    this operation implements a parallelized version of the expression
    ``target[index] = op(value, target[index])`` with optional masking. Example:

    .. code-block::

        target = dr.cuda.Float([...])
        value = dr.cuda.Float([...])
        index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
        dr.scatter_reduce(dr.ReduceOp.Add, target, value=value, index=index)

    2. When ``target`` is a more complex type (e.g. a :ref:`custom source structure
    <custom-struct>`, nested Dr.Jit array, tuple, list, dictionary, etc.), then
    ``target`` and ``value`` must be of the same type. The scatter reduce operation
    threads through entries and invokes itself recursively. For example, the
    scatter operation in

        .. code-block::

            target = dr.cuda.Array3f(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter_reduce(dr.ReduceOp.Add, target, value, index)

        is equivalent to

        .. code-block::

            dr.scatter_reduce(dr.ReduceOp.Add, target.x, value.x, index)
            dr.scatter_reduce(dr.ReduceOp.Add, target.y, value.y, index)
            dr.scatter_reduce(dr.ReduceOp.Add, target.z, value.z, index)

    .. danger::

        The indices provided to this operation are unchecked. Out-of-bounds writes
        are undefined behavior (if not disabled via the ``active`` parameter) and
        may crash the application. Negative indices are not permitted.

    Args:
        op (drjit.ReduceOp): Operation to be perform in the reduction.
        target (object): The object into which data should be written (typically
          a 1D Dr.Jit array, but other variations are possible as well, see the
          description above.)

        value (object): The values to be written (typically of type ``type(target)``,
          but other variations are possible as well, see the description above.)
          Dr.Jit will attempt an implicit conversion if the the input is not an
          array type.

        index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
          :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
          specifying gather indices. Dr.Jit will attempt an implicit conversion
          if another type is provided.

        active (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
          :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
          specifying active components. Dr.Jit will attempt an implicit
          conversion if another type is provided. The default is `True`.
    '''
    target_type = type(target)
    if not issubclass(target_type, ArrayBase):
        if _dr.is_drjit_struct_v(target_type):
            if type(value) is not target_type:
                raise Exception('scatter_reduce(): type mismatch involving custom data structure!')
            for k in target_type.DRJIT_STRUCT.keys():
                scatter_reduce(op, getattr(target, k), getattr(value, k),
                               index, active)
        else:
            assert isinstance(index, int) and isinstance(active, bool)
            assert op == _dr.ReduceOp.Add
            if active:
                target[index] += value
    else:
        if _dr.is_tensor_v(target) or _dr.is_tensor_v(value):
            raise Exception("scatter_reduce(): Tensor type not supported! "
                            "Should work with the underlying array instead. (e.g. tensor.array)")
        if target.Depth != 1:
            if _dr.array_size_v(target) != _dr.array_size_v(value):
                raise Exception("scatter_reduce(): mismatched source/target configuration!")

            for i in range(len(target)):
                scatter_reduce(op, target.entry_ref_(i), value[i], index, active)
        else:
            if not _dr.is_array_v(value):
                value = target_type(value)

            index_type = _dr.uint32_array_t(type(value))
            if not isinstance(index, index_type):
                index = _broadcast_index(index_type, index)

            active_type = index_type.MaskType
            if not isinstance(active, active_type):
                active = active_type(active)

            return value.scatter_reduce_(op, target, index, active)


def ravel(array, order='A'):
    '''
    Convert the input into a contiguous flat array

    This operation takes a Dr.Jit array, typically with some static and some
    dynamic dimensions (e.g., :py:class:`drjit.cuda.Array3f` with shape `3xN`),
    and converts it into a flattened 1D dynamically sized array (e.g.,
    :py:class:`drjit.cuda.Float`) using either a C or Fortran-style ordering
    convention.

    It can also convert Dr.Jit tensors into a flat representation, though only
    C-style ordering is supported in this case.

    For example,

    .. code-block::

        x = dr.cuda.Array3f([1, 2], [3, 4], [5, 6])
        y = dr.ravel(x, order=...)

    will produce

    - ``[1, 3, 5, 2, 4, 6]`` with ``order='F'`` (the default for Dr.Jit arrays),
    which means that X/Y/Z components alternate.
    - ``[1, 2, 3, 4, 5, 6]`` with ``order='C'``, in which case all X coordinates
    are written as a contiguous block followed by the Y- and then Z-coordinates.

    .. danger::

        Currently C-style ordering is not implemented for tensor types.

    Args:
        array (drjit.ArrayBase): An arbitrary Dr.Jit array or tensor

        order (str): A single character indicating the index order. ``'F'``
          indicates column-major/Fortran-style ordering, in which case the first
          index changes at the highest frequency. The alternative ``'C'`` specifies
          row-major/C-style ordering, in which case the last index changes at the
          highest frequency. The default value ``'A'`` (automatic) will use F-style
          ordering for arrays and C-style ordering for tensors.

    Returns:
        object: A dynamic 1D array containing the flattened representation of
        ``array`` with the desired ordering. The type of the return value depends
        on the type of the input. When ``array`` is already contiguous/flattened,
        this function returns it without making a copy.
    '''
    if not _var_is_drjit(array):
        return array
    elif array.IsTensor:
        if order == 'C':
            raise Exception('ravel(): C-style ordering not implemented for tensors!')
        return array.array
    elif array.Depth == 1:
        return array

    s = shape(array)

    if array.IsSpecial and not array.IsMatrix:
        name = _dr.detail.array_name('Array', array.Type, s, array.IsScalar)
        t = getattr(_modules.get(array.__module__), name)
        array = t(array)

    if s is None:
        raise Exception('ravel(): ragged arrays not permitted!')

    target_type = type(array)
    while target_type.Depth > 1:
        target_type = target_type.Value
    index_type = _dr.uint32_array_t(target_type)

    target = empty(target_type, hprod(s))

    if order == 'A':
        order = 'C' if target_type.IsTensor else 'F'

    if order == 'F':
        scatter(target, array, arange(index_type, s[-1]))
    elif order == 'C':
        n = len(array)
        for i in range(n):
            scatter(target, array[i], arange(index_type, s[-1]) + s[-1] * i)
    else:
        raise Exception('ravel(): invalid order argument, must be \'A\', \'F\' or \'C\'!')

    return target


def unravel(dtype, array, order='F'):
    """
    Load a sequence of Dr.Jit vectors/matrices/etc. from a contiguous flat array

    This operation implements the inverse of :py:func:`drjit.ravel()`. In contrast
    to :py:func:`drjit.ravel()`, it requires one additional parameter (``dtype``)
    specifying type of the return value. For example,

    .. code-block::

        x = dr.cuda.Float([1, 2, 3, 4, 5, 6])
        y = dr.unravel(dr.cuda.Array3f, x, order=...)

    will produce an array of two 3D vectors with different contents depending
    on the indexing convention:

    - ``[1, 2, 3]`` and ``[4, 5, 6]`` when unraveled with ``order='F'`` (the default for Dr.Jit arrays), and
    - ``[1, 3, 5]`` and ``[2, 4, 6]`` when unraveled with ``order='C'``

    Args:
        dtype (type): An arbitrary Dr.Jit array type

        array (drjit.ArrayBase): A dynamically sized 1D Dr.Jit array instance
          that is compatible with ``dtype``. In other words, both must have the
          same underlying scalar type and be located imported in the same package
          (e.g., ``drjit.llvm.ad``).

        order (str): A single character indicating the index order. ``'F'`` (the
          default) indicates column-major/Fortran-style ordering, in which case
          the first index changes at the highest frequency. The alternative
          ``'C'`` specifies row-major/C-style ordering, in which case the last
          index changes at the highest frequency.


    Returns:
        object: An instance of type ``dtype`` containing the result of the unravel
        operation.
    """
    if not isinstance(array, ArrayBase) or array.Depth != 1:
        raise Exception('unravel(): array input must be a flat array!')
    elif not issubclass(dtype, ArrayBase) or dtype.Depth == 1:
        raise Exception("unravel(): expected a nested array as target type!")

    size = 1
    t = dtype
    while t.Size != Dynamic:
        size *= t.Size
        t = t.Value
    index_type = _dr.uint32_array_t(t)

    if len(array) % size != 0:
        raise Exception('unravel(): input array length must be '
                        'divisible by %i!' % size)

    n = len(array) // size
    if order == 'F':
        indices = arange(index_type, n)
        return gather(dtype, array, indices)
    elif order == 'C':
        result = dtype()
        for i in range(len(result)):
            indices = arange(index_type, n) + i * n
            result[i] = gather(dtype.Value, array, indices)
        return result
    else:
        raise Exception('unravel(): invalid order argument, must be \'F\' or \'C\'!')


def slice(value, index=None, return_type=None):
    """
    Slice a structure of arrays to return a single entry for a given index

    This function is the equivalent to ``__getitem__(index)`` for the *dynamic
    dimension* of a Dr.Jit array or :ref:`custom source structure <custom-struct>`.
    It can be used to access a single element out a structure of arrays for a
    given index.

    The returned object type will differ from the type of the input value as its
    *dynamic dimension* will be removed. For static arrays (e.g.
    :py:class:`drjit.cuda.Array3f`) the function will return a Python ``list``.
    For :ref:`custom source structure <custom-struct>` the returned type needs
    to be specified through the argument ``return_type``.

    Args:
        value (object): A dynamically sized 1D Dr.Jit array instance
          that is compatible with ``dtype``. In other words, both must have the
          same underlying scalar type and be located imported in the same package
          (e.g., ``drjit.llvm.ad``).

        index (int): Index of the entry to be returned in the structure of arrays.
          When not specified (or ``None``), the provided object must have a
          dynamic width of ``1`` and this function will *remove* the dynamic
          dimension to this object by casting it into the appropriate type.

        return_type (type): A return type must be specified when slicing through
          a :ref:`custom source structure <custom-struct>`. Otherwise set to ``None``.

    Returns:
        object: Single entry of the structure of arrays.
    """
    t = type(value)
    if _dr.array_depth_v(t) > 1 or issubclass(t, _Sequence):
        size = len(value)
        result = [None] * size
        for i in range(size):
            result[i] = _dr.slice(value[i], index)
        return result
    elif _dr.is_drjit_struct_v(value):
        if return_type == None:
            raise Exception('slice(): return type should be specified for drjit struct!')
        result = return_type()
        for k in type(value).DRJIT_STRUCT.keys():
            setattr(result, k, _dr.slice(getattr(value, k), index))
        return result
    elif _dr.is_dynamic_array_v(value):
        if index is None:
            if _dr.width(value) > 1:
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

        if _dr.is_array_v(b):
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
    if a.IsMatrix and not (_dr.is_vector_v(b) or _dr.is_matrix_v(b)):
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
                        "Dr.Jit integer arrays.")
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
                        "Dr.Jit integer arrays.")

    if isinstance(b, float) or isinstance(b, int):
        a *= 1.0 / b
        return a

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.itruediv_(b)


def op_floordiv(a, b):
    if not a.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Dr.Jit floating point arrays.")

    if isinstance(b, int):
        if b == 0:
            raise Exception("Division by zero!")
        elif b == 1:
            return a

        # Division via multiplication by magic number + shift
        multiplier, shift = _dr.detail.idiv(a.Type, b)

        if a.IsSigned:
            q = mulhi(multiplier, a) + a
            q_sign = q >> (a.Type.Size * 8 - 1)
            q = q + (q_sign & ((1 << shift) - (1 if multiplier == 0 else 0)))
            sign = type(a)(-1 if b < 0 else 0)
            return ((q >> shift) ^ sign) - sign
        else:
            if multiplier == 0:
                return a >> (shift + 1)

            q = _dr.mulhi(multiplier, a)
            t = ((a - q) >> 1) + q
            return t >> shift

    if type(a) is not type(b):
        a, b = _var_promote(a, b)

    return a.floordiv_(b)


def op_ifloordiv(a, b):
    if not a.IsIntegral:
        raise Exception("Use the true division operator \"/\" for "
                        "Dr.Jit floating point arrays.")

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
    if type(a) is not type(b) and type(b) is not _dr.mask_t(a):
        a, b = _var_promote_mask(a, b)

    return a.and_(b)


def op_rand(a, b):
    if type(a) is not type(b) and type(a) is not _dr.mask_t(b):
        b, a = _var_promote_mask(b, a)
    return b.and_(a)


def op_iand(a, b):
    if type(a) is not type(b) and type(b) is not _dr.mask_t(a):
        a, b = _var_promote_mask(a, b)

    return a.iand_(b)


def and_(a, b):
    if type(a) is bool and type(b) is bool:
        return a and b
    else:
        return a & b


def op_or(a, b):
    if type(a) is not type(b) and type(b) is not _dr.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.or_(b)


def op_ror(a, b):
    if type(a) is not type(b) and type(a) is not _dr.mask_t(b):
        b, a = _var_promote_mask(b, a)
    return b.or_(a)


def op_ior(a, b):
    if type(a) is not type(b) and type(b) is not _dr.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.ior_(b)


def or_(a, b):
    if type(a) is bool and type(b) is bool:
        return a or b
    else:
        return a | b


def op_xor(a, b):
    if type(a) is not type(b) and type(b) is not _dr.mask_t(a):
        a, b = _var_promote_mask(a, b)
    return a.xor_(b)


def op_rxor(a, b):
    if type(a) is not type(b) and type(a) is not _dr.mask_t(b):
        b, a = _var_promote_mask(b, a)
    return b.xor_(a)


def op_ixor(a, b):
    if type(a) is not type(b) and type(b) is not _dr.mask_t(a):
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
        return round(a)


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
        return _dr.detail.fmadd_scalar(a, b, c)


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

    if type_t is not type_f or type_m is not _dr.mask_t(t):
        if type_t is type_f and _dr.is_drjit_struct_v(type_t):
            result = type_t()
            for k in type_t.DRJIT_STRUCT.keys():
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
    if _dr.is_array_v(a):
        return ~eq(a, a)
    else:
        return not (a == a)


def isinf(a):
    return eq(abs(a), _dr.inf)


def isfinite(a):
    return abs(a) < _dr.inf


def lerp(a, b, t):
    return fmadd(b, t, fnmadd(a, t, a))


def clamp(value, min, max):
    return _dr.max(_dr.min(value, max), min)


def arg(value):
    if _dr.is_complex_v(value):
        return _dr.atan2(value.imag, value.real)
    else:
        return _dr.select(value >= 0, 0, -_dr.pi)


def real(value):
    if _dr.is_complex_v(value):
        return value[0]
    elif _dr.is_quaternion_v(value):
        return value[3]
    else:
        return value


def imag(value):
    if _dr.is_complex_v(value):
        return value[1]
    elif _dr.is_quaternion_v(value):
        name = _dr.detail.array_name('Array', value.Type, (3), value.IsScalar)
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
    '''
    Safely evaluate the square root of the provided input avoiding domain errors.

    Negative inputs produce a ``0.0`` output value.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input
    '''
    result = sqrt(max(a, 0))
    if _dr.is_diff_array_v(a) and _dr.grad_enabled(a):
        alt = sqrt(max(a, _dr.epsilon(a)))
        result = replace_grad(result, alt)
    return result


def safe_asin(a):
    '''
    Safe wrapper around :py:func:`drjit.asin` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arcsine approximation
    '''
    result = asin(clamp(a, -1, 1))
    if _dr.is_diff_array_v(a) and _dr.grad_enabled(a):
        alt = asin(clamp(a, -_dr.one_minus_epsilon(a), _dr.one_minus_epsilon(a)))
        result = replace_grad(result, alt)
    return result


def safe_acos(a):
    '''
    Safe wrapper around :py:func:`drjit.acos` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arccosine approximation
    '''
    result = acos(clamp(a, -1, 1))
    if _dr.is_diff_array_v(a) and _dr.grad_enabled(a):
        alt = acos(clamp(a, -_dr.one_minus_epsilon(a), _dr.one_minus_epsilon(a)))
        result = replace_grad(result, alt)
    return result


# -------------------------------------------------------------------
#       Vertical operations -- AD/JIT compilation-related
# -------------------------------------------------------------------


def label(a):
    if isinstance(a, ArrayBase):
        return a.label_()
    else:
        return None


def set_label(*args, **kwargs):
    n_args, n_kwargs = len(args), len(kwargs)
    if (n_kwargs and n_args) or (n_args and n_args != 2):
        raise Exception('set_label(): invalid input arguments')

    if n_args:
        a, label = args
        if _dr.is_jit_array_v(a) or _dr.is_diff_array_v(a):
            a.set_label_(label)
        elif isinstance(a, _Mapping):
            for k, v in a.items():
                set_label(v, label + "_" + k)
        elif _dr.is_drjit_struct_v(a):
            for k in a.DRJIT_STRUCT.keys():
                set_label(getattr(a, k), label + "_" + k)
    elif n_kwargs:
        for k, v in kwargs.items():
            set_label(v, k)


def schedule(*args):
    result = False
    for a in args:
        t = type(a)
        if issubclass(t, ArrayBase):
            result |= a.schedule_()
        elif _dr.is_drjit_struct_v(t):
            for k in t.DRJIT_STRUCT.keys():
                result |= schedule(getattr(a, k))
        elif issubclass(t, _Sequence):
            for v in a:
                result |= schedule(v)
        elif issubclass(t, _Mapping):
            for k, v in a.items():
                result |= schedule(v)
    return result


def eval(*args):
    if schedule(*args) or len(args) == 0:
        _dr.detail.eval()


def graphviz_str(arg):
    base = _dr.leaf_array_t(arg)

    if _dr.is_diff_array_v(base):
        return base.graphviz_()
    elif _dr.is_jit_array_v(base):
        return _dr.detail.graphviz()
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
                        'you can call drjit.graphviz_str() function to obtain '
                        'a string representation.')


def migrate(a, type_):
    if _dr.is_jit_array_v(a):
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
        return 1 / _math.cos(a)


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
    if _dr.is_tensor_v(a):
        return type(a)(a.array.pow_(b if not _dr.is_tensor_v(b) else b.array), a.shape)
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
        raise Exception("erfinv(): only implemented for drjit types!")


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
    return a * (180.0 / _dr.pi)


def deg_to_rad(a):
    return a * (_dr.pi / 180.0)


# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def shuffle(perm, value):
    if not _dr.is_array_v(value) or len(perm) != value.Size:
        raise Exception("shuffle(): incompatible input!")

    result = type(value)()
    for i, j in enumerate(perm):
        result[i] = value[j]

    return result


def compress(mask):
    if not _dr.is_mask_v(mask) or not mask.IsDynamic:
        raise Exception("compress(): incompatible input!")
    return mask.compress_()


def count(a):
    if _var_is_drjit(a):
        if a.Type != VarType.Bool:
            raise Exception("count(): input array must be a mask!")
        return a.count_()
    elif isinstance(a, bool):
        return 1 if a else 0
    elif _dr.is_iterable_v(a):
        result = 0
        for index, value in enumerate(a):
            if index == 0:
                result = _dr.select(value, 0, 1)
            else:
                result = result + _dr.select(value, 1, 0)
        return result
    else:
        raise Exception("count(): input must be a boolean or an "
                        "iterable containing masks!")


def all(a):
    if _var_is_drjit(a):
        if a.Type != VarType.Bool:
            raise Exception("all(): input array must be a mask!")
        return a.all_()
    elif isinstance(a, bool):
        return a
    elif _dr.is_iterable_v(a):
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
    if _dr.is_jit_array_v(a) and a.Depth == 1:
        return value
    else:
        return _dr.all(a)


def any(a):
    if _var_is_drjit(a):
        if a.Type != VarType.Bool:
            raise Exception("any(): input array must be a mask!")
        return a.any_()
    elif isinstance(a, bool):
        return a
    elif _dr.is_iterable_v(a):
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
    if _dr.is_jit_array_v(a) and a.Depth == 1:
        return value
    else:
        return _dr.any(a)


def none(a):
    b = any(a)
    return not b if isinstance(b, bool) else ~b


def none_nested(a):
    b = any_nested(a)
    return not b if isinstance(b, bool) else ~b


def none_or(value, a):
    assert isinstance(value, bool)
    if _dr.is_jit_array_v(a) and a.Depth == 1:
        return value
    else:
        return _dr.none(a)


def hsum(a):
    if _var_is_drjit(a):
        return a.hsum_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _dr.is_iterable_v(a):
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
    return a.hsum_async_()


def hsum_nested(a):
    while True:
        b = hsum(a)
        if b is a:
            break
        a = b
    return a


def hmean(a):
    if hasattr(a, '__len__'):
        return _dr.hsum(a) / len(a)
    else:
        return a


def hmean_async(a):
    if hasattr(a, '__len__'):
        return _dr.hsum_async(a) / len(a)
    else:
        return a


def hmean_nested(a):
    while True:
        b = hmean(a)
        if b is a:
            break
        a = b
    return a


def hprod(a):
    if _var_is_drjit(a):
        return a.hprod_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _dr.is_iterable_v(a):
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
    return a.hprod_async_()


def hprod_nested(a):
    while True:
        b = hprod(a)
        if b is a:
            break
        a = b
    return a


def hmax(a):
    if _var_is_drjit(a):
        return a.hmax_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _dr.is_iterable_v(a):
        result = None
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = _dr.max(result, value)
        if result is None:
            raise Exception("hmax(): zero-sized array!")
        return result
    else:
        raise Exception("hmax(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def hmax_async(a):
    return a.hmax_async_()

def hmax_nested(a):
    while True:
        b = hmax(a)
        if b is a:
            break
        a = b
    return a


def hmin(a):
    if _var_is_drjit(a):
        return a.hmin_()
    elif isinstance(a, float) or isinstance(a, int):
        return a
    elif _dr.is_iterable_v(a):
        result = None
        for index, value in enumerate(a):
            if index == 0:
                result = value
            else:
                result = _dr.min(result, value)
        if result is None:
            raise Exception("hmin(): zero-sized array!")
        return result
    else:
        raise Exception("hmin(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def hmin_async(a):
    return a.hmin_async_()


def hmin_nested(a):
    while True:
        b = hmin(a)
        if b is a:
            break
        a = b
    return a


def dot(a, b):
    if _dr.is_matrix_v(a) or _dr.is_matrix_v(b):
        raise Exception("dot(): input shouldn't be a Matrix!"
                        "The @ operator should be used instead.")

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.dot_(b)


def dot_async(a, b):
    if _dr.is_matrix_v(a) or _dr.is_matrix_v(b):
        raise Exception("dot_async(): input shouldn't be a Matrix!"
                        "The @ operator should be used instead.")

    if type(a) is not type(b):
        a, b = _var_promote(a, b)

    elif hasattr(a, 'dot_async_'):
        return a.dot_async_(b)
    else:
        return type(a)(a.dot_(b))


def abs_dot(a, b):
    if _dr.is_matrix_v(a) or _dr.is_matrix_v(b):
        raise Exception("abs_dot(): input shouldn't be a Matrix!"
                        "The @ operator should be used instead.")

    return abs(dot(a, b))


def abs_dot_async(a, b):
    if _dr.is_matrix_v(a) or _dr.is_matrix_v(b):
        raise Exception("abs_dot_async(): input shouldn't be a Matrix!"
                        "The @ operator should be used instead.")

    return abs(dot_async(a, b))


def squared_norm(a):
    return dot(a, a)


def norm(a):
    return sqrt(dot(a, a))


def normalize(a):
    return a * rsqrt(squared_norm(a))


def conj(a):
    if _dr.is_complex_v(a):
        return type(a)(a.real, -a.imag)
    elif _dr.is_quaternion_v(a):
        return type(a)(-a.x, -a.y, -a.z, a.w)
    else:
        return a


def hypot(a, b):
    a, b = abs(a), abs(b)
    maxval = _dr.max(a, b)
    minval = _dr.min(a, b)
    ratio = minval / maxval
    inf = _dr.inf

    return _dr.select(
        (a < inf) & (b < inf) & (ratio < inf),
        maxval * _dr.sqrt(_dr.fmadd(ratio, ratio, 1)),
        a + b
    )


def tile(array, count: int):
    if not _dr.is_array_v(array) or not isinstance(count, int):
        raise("tile(): invalid input types!")
    elif not array.IsDynamic:
        raise("tile(): first input argument must be a dynamic Dr.Jit array!")

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
        index = _dr.arange(_dr.uint_array_t(t), size * count) % size
        return _dr.gather(t, array, index)


def repeat(array, count: int):
    if not _dr.is_array_v(array) or not isinstance(count, int):
        raise("tile(): invalid input types!")
    elif not array.IsDynamic:
        raise("tile(): first input argument must be a dynamic Dr.Jit array!")

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
        index = _dr.arange(_dr.uint_array_t(t), size * count) // count
        return _dr.gather(t, array, index)


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
        if not _dr.is_dynamic_array_v(v) or \
           _dr.array_depth_v(v) != 1 or type(v) is not t:
            raise Exception("meshgrid(): consistent 1D dynamic arrays expected!")

    size = _dr.hprod((len(v) for v in args))
    index = _dr.arange(_dr.uint32_array_t(t), size)

    result = []

    # This seems non-symmetric but is necessary to be consistent with NumPy
    if indexing == "xy":
        args = (args[1], args[0], *args[2:])

    for v in args:
        size //= len(v)
        index_v = index // size
        index = fnmadd(index_v, size, index)
        result.append(_dr.gather(t, v, index_v))

    if indexing == "xy":
        result[0], result[1] = result[1], result[0]

    return tuple(result)


def block_sum(value, block_size):
    if _dr.is_jit_array_v(value):
        return value.block_sum_(block_size)
    else:
        raise Exception("block_sum(): requires a JIT array!")


def binary_search(start, end, pred):
    assert isinstance(start, int) and isinstance(end, int)

    iterations = log2i(end - start) + 1 if start < end else 0

    for i in range(iterations):
        middle = (start + end) >> 1

        cond = pred(middle)
        start = _dr.select(cond, _dr.min(middle + 1, end), start)
        end = _dr.select(cond, end, middle)

    return start


# -------------------------------------------------------------------
#    Transformations, matrices, operations for 3D vector spaces
# -------------------------------------------------------------------

def cross(a, b):
    if _dr.array_size_v(a) != 3 or _dr.array_size_v(a) != 3:
        raise Exception("cross(): requires 3D input arrays!")

    ta, tb = type(a), type(b)

    return fmsub(ta(a.y, a.z, a.x), tb(b.z, b.x, b.y),
                 ta(a.z, a.x, a.y) * tb(b.y, b.z, b.x))


# -------------------------------------------------------------------
#                     Automatic differentiation
# -------------------------------------------------------------------


def detach(a, preserve_type=False):
    if _dr.is_diff_array_v(a):
        if preserve_type:
            return type(a)(a.detach_())
        else:
            return a.detach_()
    elif _dr.is_drjit_struct_v(a):
        result = type(a)()
        for k in type(a).DRJIT_STRUCT.keys():
            setattr(result, k, detach(getattr(a, k), preserve_type=preserve_type))
        return result
    else:
        return a


def grad(a):
    if _dr.is_diff_array_v(a):
        return a.grad_()
    elif _dr.is_drjit_struct_v(a):
        result = type(a)()
        for k in type(a).DRJIT_STRUCT.keys():
            setattr(result, k, grad(getattr(a, k)))
        return result
    elif isinstance(a, _Sequence):
        return type(a)([grad(v) for v in a])
    elif isinstance(a, _Mapping):
        return {k : grad(v) for k, v in a.items()}
    else:
        return _dr.zero(type(a))


def set_grad(a, value):
    if _dr.is_diff_array_v(a) and a.IsFloat:
        if _dr.is_diff_array_v(value):
            value = _dr.detach(value)

        t = _dr.detached_t(a)
        if type(value) is not t:
            value = t(value)

        a.set_grad_(value)
    elif isinstance(a, _Sequence):
        vs = isinstance(value, _Sequence)
        assert not vs or len(a) == len(value)
        for i in range(len(a)):
            set_grad(a[i], value[i] if vs else value)
    elif isinstance(a, _Mapping):
        vm = isinstance(value, _Mapping)
        assert not vm or a.keys() == value.keys()
        for k, v in a.items():
            set_grad(v, value[k] if vm else value)
    elif _dr.is_drjit_struct_v(a):
        ve = _dr.is_drjit_struct_v(value)
        assert not ve or type(value) is type(a)
        for k in type(a).DRJIT_STRUCT.keys():
            set_grad(getattr(a, k), getattr(value, k) if ve else value)


def accum_grad(a, value):
    if _dr.is_diff_array_v(a) and a.IsFloat:
        if _dr.is_diff_array_v(value):
            value = _dr.detach(value)

        t = _dr.detached_t(a)
        if type(value) is not t:
            value = t(value)

        a.accum_grad_(value)
    elif isinstance(a, _Sequence):
        vs = isinstance(value, _Sequence)
        assert not vs or len(a) == len(value)
        for i in range(len(a)):
            accum_grad(a[i], value[i] if vs else value)
    elif isinstance(a, _Mapping):
        vm = isinstance(value, _Mapping)
        assert not vm or a.keys() == value.keys()
        for k, v in a.items():
            accum_grad(v, value[k] if vm else value)
    elif _dr.is_drjit_struct_v(a):
        ve = _dr.is_drjit_struct_v(value)
        assert not ve or type(value) is type(a)
        for k in type(a).DRJIT_STRUCT.keys():
            accum_grad(getattr(a, k), getattr(value, k) if ve else value)


def grad_enabled(*args):
    result = False
    for a in args:
        if _dr.is_diff_array_v(a):
            result |= a.grad_enabled_()
        elif _dr.is_drjit_struct_v(a):
            for k in type(a).DRJIT_STRUCT.keys():
                result |= grad_enabled(getattr(a, k))
        elif isinstance(a, _Sequence):
            for v in a:
                result |= grad_enabled(v)
        elif isinstance(a, _Mapping):
            for k, v in a.items():
                result |= grad_enabled(v)
    return result


def set_grad_enabled(a, value):
    if _dr.is_diff_array_v(a) and a.IsFloat:
        a.set_grad_enabled_(value)
    elif _dr.is_drjit_struct_v(a):
        for k in type(a).DRJIT_STRUCT.keys():
            set_grad_enabled(getattr(a, k), value)
    elif isinstance(a, _Sequence):
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


def replace_grad(a, b):
    if type(a) is not type(b):
        a, b = _var_promote(a, b)

    ta, tb = type(a), type(b)

    if not (_dr.is_diff_array_v(ta) and ta.IsFloat and
            _dr.is_diff_array_v(tb) and tb.IsFloat):
        raise Exception("replace_grad(): unsupported input types!")

    la, lb = len(a), len(b)
    depth = a.Depth # matches b.Depth

    if la != lb:
        if la == 1 and depth == 1:
            a = a + zero(ta, lb)
        elif lb == 1 and depth == 1:
            b = b + zero(tb, la)
        else:
            raise Exception("replace_grad(): input arguments have "
                           "incompatible sizes (%i vs %i)!"
                           % (la, lb))

    if depth > 1:
        result = ta()
        if a.Size == Dynamic:
            result.init_(la)
        for i in range(la):
            result[i] = replace_grad(a[i], b[i])
        return result
    else:
        if _dr.is_tensor_v(a):
            return ta(replace_grad(a.array, b.array), a.shape)
        else:
            return ta.create_(b.index_ad(), a.detach_())


def enqueue(mode, *args):
    for a in args:
        if _dr.is_diff_array_v(a) and a.IsFloat:
            a.enqueue_(mode)
        elif isinstance(a, _Sequence):
            for v in a:
                enqueue(mode, v)
        elif isinstance(a, _Mapping):
            for k, v in a.items():
                enqueue(mode, v)
        elif _dr.is_drjit_struct_v(a):
            for k in type(a).DRJIT_STRUCT.keys():
                enqueue(mode, getattr(a, k))


def traverse(t, mode, flags=_dr.ADFlag.Default):
    assert isinstance(mode, _dr.ADMode)

    t = _dr.leaf_array_t(t)

    if not _dr.is_diff_array_v(t):
        raise Exception('traverse(): expected a differentiable array type!')

    t.traverse_(mode, flags)


def _check_grad_enabled(name, t, a):
    if _dr.is_diff_array_v(t) and t.IsFloat:
        if _dr.flag(_dr.JitFlag.VCallRecord) and not grad_enabled(a):
            raise Exception(
                f'{name}(): the argument does not depend on the input '
                'variable(s) being differentiated. Raising an exception '
                'since this is usually indicative of a bug (for example, '
                'you may have forgotten to call ek.enable_grad(..)). If '
                f'this is expected behavior, skip the call to {name}(..) '
                'if ek.grad_enabled(..) returns False.')
    else:
        raise Exception(f'{name}(): expected a differentiable array type!')


def forward_from(a, flags=_dr.ADFlag.Default):
    ta = type(a)
    _check_grad_enabled('forward_from', ta, a)
    set_grad(a, 1)
    enqueue(_dr.ADMode.Forward, a)
    traverse(ta, _dr.ADMode.Forward, flags)


def forward_to(*args, flags=_dr.ADFlag.Default):
    for a in args:
        if isinstance(a, (int, _dr.ADFlag)):
            raise Exception('forward_to(): AD flags should be passed via '
                            'the "flags=.." keyword argument')

    ta = _dr.leaf_array_t(args)
    _check_grad_enabled('forward_to', ta, args)
    enqueue(_dr.ADMode.Backward, *args)
    traverse(ta, _dr.ADMode.Forward, flags)

    return grad(args) if len(args) > 1 else grad(*args)


def forward(a, flags=_dr.ADFlag.Default):
    forward_from(a, flags)


def backward_from(a, flags=_dr.ADFlag.Default):
    ta = type(a)
    _check_grad_enabled('backward_from', ta, a)

    # Deduplicate components if 'a' is a vector
    if _dr.array_depth_v(a) > 1:
        a = a + ta(0)

    set_grad(a, 1)
    enqueue(_dr.ADMode.Backward, a)
    traverse(ta, _dr.ADMode.Backward, flags)


def backward_to(*args, flags=_dr.ADFlag.Default):
    for a in args:
        if isinstance(a, (int, _dr.ADFlag)):
            raise Exception('backward_to(): AD flags should be passed via '
                            'the "flags=.." keyword argument')

    ta = _dr.leaf_array_t(args)
    _check_grad_enabled('backward_to', ta, args)
    enqueue(_dr.ADMode.Forward, *args)
    traverse(ta, _dr.ADMode.Backward, flags)

    return grad(args) if len(args) > 1 else grad(*args)


def backward(a, flags=_dr.ADFlag.Default):
    backward_from(a, flags)


# -------------------------------------------------------------------
#                      Initialization operations
# -------------------------------------------------------------------


def zero(type_, shape=1):
    if not isinstance(type_, type):
        raise Exception('zero(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.zero_(shape)
    elif _dr.is_drjit_struct_v(type_):
        result = type_()
        for k, v in type_.DRJIT_STRUCT.items():
            setattr(result, k, zero(v, shape))
        if hasattr(type_, 'zero_'):
            result.zero_(shape)
        return result
    elif not type_ in (int, float, complex, bool):
        return None
    else:
        return type_(0)


def empty(type_, shape=1):
    if not isinstance(type_, type):
        raise Exception('empty(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.empty_(shape)
    elif _dr.is_drjit_struct_v(type_):
        result = type_()
        for k, v in type_.DRJIT_STRUCT.items():
            setattr(result, k, empty(v, shape))
        return result
    else:
        return type_(0)


def full(type_, value, shape=1):
    if not isinstance(type_, type):
        raise Exception('full(): Type expected as first argument')
    elif issubclass(type_, ArrayBase):
        return type_.full_(value, shape)
    else:
        return type_(value)


def opaque(type_, value, shape=1):
    if not isinstance(type_, type):
        raise Exception('opaque(): Type expected as first argument')
    if not _dr.is_jit_array_v(type_):
        return _dr.full(type_, value, shape)
    if _dr.is_static_array_v(type_):
        result = type_()
        for i in range(len(result)):
            result[i] = opaque(type_.Value, value, shape)
        return result
    if _dr.is_diff_array_v(type_):
        return _dr.opaque(_dr.detached_t(type_), value, shape)
    if _dr.is_jit_array_v(type_):
        if _dr.is_tensor_v(type_):
            return type_(_dr.opaque(type_.Array, value, _dr.hprod(shape)), shape)
        return type_.opaque_(value, shape)
    elif _dr.is_drjit_struct_v(type_):
        result = type_()
        for k, v in type_.DRJIT_STRUCT.items():
            setattr(result, k, opaque(v, getattr(value, k), shape))
        return result
    else:
        return type_(value)


def make_opaque(*args):
    for a in args:
        t = type(a)
        if issubclass(t, ArrayBase):
            if _dr.array_depth_v(t) > 1:
                for i in range(len(a)):
                    make_opaque(a.entry_ref_(i))
            elif _dr.is_diff_array_v(t):
                make_opaque(a.detach_ref_())
            elif _dr.is_tensor_v(t):
                make_opaque(a.array)
            elif _dr.is_jit_array_v(t):
                if not a.is_evaluated_():
                    a.assign(a.copy_())
                    a.data_()
        elif _dr.is_drjit_struct_v(t):
            for k in t.DRJIT_STRUCT.keys():
                make_opaque(getattr(a, k))
        elif issubclass(t, _Sequence):
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
    if _dr.is_special_v(type_):
        result = zero(type_, size)

        if type_.IsComplex or type_.IsQuaternion:
            result.real = identity(type_.Value, size)
        elif type_.IsMatrix:
            one = identity(type_.Value.Value, size)
            for i in range(type_.Size):
                result[i, i] = one
        return result
    elif _dr.is_array_v(type_):
        return full(type_, 1, size)
    else:
        return type_(1)

# -------------------------------------------------------------------
#                  Higher-level utility functions
# -------------------------------------------------------------------

def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    # Fast path for Dr.Jit arrays, avoid for special array types
    # due to their non-standard broadcasting behavior
    if _dr.is_array_v(a) or _dr.is_array_v(b):
        if _dr.is_diff_array_v(a):
            a = _dr.detach(a)
        if _dr.is_diff_array_v(b):
            b = _dr.detach(b)

        if _dr.is_array_v(a) and not _dr.is_floating_point_v(a):
            a, _ = _var_promote(a, 1.0)
        if _dr.is_array_v(b) and not _dr.is_floating_point_v(b):
            b, _ = _var_promote(b, 1.0)

        if type(a) is not type(b):
            a, b = _var_promote(a, b)

        diff = abs(a - b)
        shape = 1
        if _dr.is_tensor_v(diff):
            shape = diff.shape
        cond = diff <= abs(b) * rtol + _dr.full(type(diff), atol, shape)
        if _dr.is_floating_point_v(a):
            cond |= _dr.eq(a, b)  # plus/minus infinity
        if equal_nan:
            cond |= _dr.isnan(a) & _dr.isnan(b)
        return _dr.all_nested(cond)

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


def printf_async(fmt, *args, active=True):
    indices = []
    is_cuda, is_llvm = _dr.is_cuda_array_v(active), _dr.is_llvm_array_v(active)

    for a in args:
        cuda, llvm = _dr.is_cuda_array_v(a), _dr.is_llvm_array_v(a)
        if not (cuda or llvm) or _dr.array_depth_v(a) != 1 or _dr.is_mask_v(a):
            raise Exception("printf_async(): array argument of type '%s' not "
                            "supported (must be a depth-1 JIT (LLVM/CUDA) array, "
                            "and cannot be a mask)" % type(a).__name__)
        indices.append(a.index())
        is_cuda |= cuda
        is_llvm |= llvm

    if is_cuda == is_llvm:
        raise Exception("printf_async(): invalid input: must specify LLVM or CUDA arrays.")

    active = _dr.cuda.Bool(active) if is_cuda else _dr.llvm.Bool(active)
    _dr.detail.printf_async(is_cuda, active.index(), fmt, indices)


# -------------------------------------------------------------------
#               Context manager for setting JIT flags
# -------------------------------------------------------------------

class scoped_set_flag:
    def __init__(self, flag, value):
        self.flag = flag
        self.value = value

    def __enter__(self):
        self.backup = _dr.flag(self.flag)
        _dr.set_flag(self.flag, self.value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _dr.set_flag(self.flag, self.backup)

# -------------------------------------------------------------------
#                        Enabling/disabling AD
# -------------------------------------------------------------------

class _DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class _ADContextManager:
    def __init__(self, scope_type, array_type, array_indices):
        self.scope_type = scope_type
        self.array_type = array_type
        self.array_indices = array_indices

    def __enter__(self):
        if self.array_type is not None:
            self.array_type.scope_enter_(self.scope_type, self.array_indices)
        else:
            if hasattr(_dr, 'cuda'):
                _dr.cuda.ad.Float32.scope_enter_(self.scope_type, self.array_indices)
                _dr.cuda.ad.Float64.scope_enter_(self.scope_type, self.array_indices)
            if hasattr(_dr, 'llvm'):
                _dr.llvm.ad.Float32.scope_enter_(self.scope_type, self.array_indices)
                _dr.llvm.ad.Float64.scope_enter_(self.scope_type, self.array_indices)

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None

        if self.array_type is not None:
            self.array_type.scope_leave_(success)
        else:
            if hasattr(_dr, 'cuda'):
                _dr.cuda.ad.Float32.scope_leave_(success)
                _dr.cuda.ad.Float64.scope_leave_(success)
            if hasattr(_dr, 'llvm'):
                _dr.llvm.ad.Float32.scope_leave_(success)
                _dr.llvm.ad.Float64.scope_leave_(success)


def suspend_grad(*args, when=True):
    if not when:
        return _DummyContextManager()

    array_indices = []
    array_type = _dr.detail.diff_vars(args, array_indices, check_grad_enabled=False)
    if len(args) > 0 and len(array_indices) == 0:
        array_indices = [0]
    return _ADContextManager(_dr.detail.ADScope.Suspend, array_type, array_indices)


def resume_grad(*args, when=True):
    if not when:
        return _DummyContextManager()

    array_indices = []
    array_type = _dr.detail.diff_vars(args, array_indices, check_grad_enabled=False)
    if len(args) > 0 and len(array_indices) == 0:
        array_indices = [0]
    return _ADContextManager(_dr.detail.ADScope.Resume, array_type, array_indices)


def isolate_grad(when=True):
    if not when:
        return _DummyContextManager()
    return _ADContextManager(_dr.detail.ADScope.Isolate, None, [])


# -------------------------------------------------------------------
#             Automatic differentation of custom fuctions
# -------------------------------------------------------------------

class CustomOp:
    def __init__(self):
        self._implicit_in = []
        self._implicit_out = []

    def forward(self):
        raise Exception("CustomOp.forward(): not implemented")

    def backward(self):
        raise Exception("CustomOp.backward(): not implemented")

    def grad_out(self):
        return _dr.grad(self.output)

    def set_grad_out(self, value):
        _dr.accum_grad(self.output, value)

    def grad_in(self, name):
        if name not in self.inputs:
            raise Exception("CustomOp.grad_in(): Could not find "
                            "input argument named \"%s\"!" % name)
        return _dr.grad(self.inputs[name])

    def set_grad_in(self, name, value):
        if name not in self.inputs:
            raise Exception("CustomOp.set_grad_in(): Could not find "
                            "input argument named \"%s\"!" % name)
        _dr.accum_grad(self.inputs[name], value)

    def add_input(self, value):
        self._implicit_in.append(value)

    def add_output(self, value):
        self._implicit_out.append(value)

    def __del__(self):
        def ad_clear(o):
            if _dr.array_depth_v(o) > 1 \
               or isinstance(o, _Sequence):
                for v in o:
                    ad_clear(v)
            elif isinstance(o, _Mapping):
                for k, v in o.items():
                    ad_clear(v)
            elif _dr.is_diff_array_v(o):
                if _dr.is_tensor_v(o):
                    ad_clear(o.array)
                else:
                    o.set_index_ad_(0)
            elif _dr.is_drjit_struct_v(o):
                for k in type(o).DRJIT_STRUCT.keys():
                    ad_clear(getattr(o, k))
        ad_clear(getattr(self, 'output', None))

    def name(self):
        return "CustomOp[unnamed]"


def custom(cls, *args, **kwargs):
    # Clear primal values of a differentiable array
    def clear_primal(o, dec_ref):
        if _dr.array_depth_v(o) > 1 \
           or isinstance(o, _Sequence):
            return type(o)([clear_primal(v, dec_ref) for v in o])
        elif isinstance(o, _Mapping):
            return { k: clear_primal(v, dec_ref) for k, v in o.items() }
        elif _dr.is_diff_array_v(o) and _dr.is_floating_point_v(o):
            ot = type(o)

            if _dr.is_tensor_v(ot):
                value = ot.Array.create_(
                    o.array.index_ad(),
                    zero(_dr.detached_t(ot.Array), hprod(o.shape)))
                result = ot(value, o.shape)
            else:
                result = value = ot.create_(
                    o.index_ad(),
                    _dr.detached_t(ot)())
            if dec_ref:
                value.dec_ref_()
            return result
        elif _dr.is_drjit_struct_v(o):
            res = type(o)()
            for k in type(o).DRJIT_STRUCT.keys():
                setattr(res, k, clear_primal(getattr(o, k), dec_ref))
            return res
        else:
            return o

    inst = cls()

    # Convert args to kwargs
    kwargs.update(zip(inst.eval.__code__.co_varnames[1:], args))

    output = inst.eval(**{ k: _dr.detach(v) for k, v in kwargs.items() })
    if _dr.grad_enabled(output):
        raise Exception("drjit.custom(): the return value of CustomOp.eval() "
                        "should not be attached to the AD graph!")

    diff_vars_in = []
    _dr.detail.diff_vars(kwargs, diff_vars_in)
    _dr.detail.diff_vars(inst._implicit_in, diff_vars_in)

    if len(diff_vars_in) > 0:
        output = _dr.diff_array_t(output)
        Type = _dr.leaf_array_t(output)
        tmp_in, tmp_out = Type(), Type()
        _dr.enable_grad(tmp_in, tmp_out, output)

        inst.inputs = clear_primal(kwargs, dec_ref=False)
        inst.output = clear_primal(output, dec_ref=True)

        diff_vars_out = []
        _dr.detail.diff_vars(inst.output, diff_vars_out)
        _dr.detail.diff_vars(inst._implicit_out, diff_vars_out)

        if len(diff_vars_out) == 0:
            return output # Not relevant for AD after all..

        detail = _modules.get(Type.__module__ + ".detail")

        if len(diff_vars_in) > 1:
            _dr.set_label(tmp_in, inst.name() + "_in")
            for index in diff_vars_in:
                Type.add_edge_(index, tmp_in.index_ad())

        if len(diff_vars_out) > 1:
            _dr.set_label(tmp_out, inst.name() + "_out")
            for index in diff_vars_out:
                Type.add_edge_(tmp_out.index_ad(), index)

        Type.add_edge_(
            diff_vars_in[0]  if len(diff_vars_in)  == 1 else tmp_in.index_ad(),
            diff_vars_out[0] if len(diff_vars_out) == 1 else tmp_out.index_ad(),
            inst
        )

        inst._implicit_in = []
        inst._implicit_out = []

    return output


def get_cmake_dir():
    from os import path
    file_dir = path.abspath(path.dirname(__file__))
    cmake_path = path.join(file_dir, "share", "cmake", "drjit")
    if not path.exists(cmake_path):
        raise ImportError("Cannot find Dr.Jit CMake directory")
    return cmake_path
