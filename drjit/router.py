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
    '''
    Replace the scalar type associated to a Dr.Jit type, specified by a VarType
    '''
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


def shape(arg, /):
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

    s = []
    if not _shape_impl(arg, 0, s):
        return None
    else:
        return s


def width(arg, /):
    '''
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
    if _dr.is_array_v(arg):
        if _dr.is_tensor_v(arg):
            return width(arg.array)
        else:
            return shape(arg)[-1]
    elif _dr.is_struct_v(arg):
        result = 0
        for k in type(arg).DRJIT_STRUCT.keys():
            result = _builtins.max(result, width(getattr(arg, k)))
        return result
    else:
        return 1


def resize(arg, size):
    '''
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
    if _dr.depth_v(arg) > 1:
        for i in range(arg.Size):
            resize(arg[i], size)
    elif _dr.is_jit_v(arg):
        arg.resize_(size)
    elif _dr.is_struct_v(arg):
        for k in type(arg).DRJIT_STRUCT.keys():
            resize(getattr(arg, k), size)


def device(value=None):
    '''
    Return the CUDA device ID associated with the current thread.
    '''
    if value is None:
        return _dr.detail.device()
    elif _dr.depth_v(value) > 1:
        return device(value[0])
    elif _dr.is_diff_v(value):
        return device(_dr.detach(value, preserve_type=False))
    elif _dr.is_jit_v(value):
        return _dr.detail.device(value.index)
    else:
        return -1


# By default, don't print full contents of arrays with more than 20 entries
_print_threshold = 20


def _repr_impl(self, shape, buf, *idx):
    '''
    Implementation detail of op_repr()
    '''
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
    '''
    Return the maximum number of entries displayed when printing an array
    '''
    return _print_threshold


def set_print_threshold(size):
    '''
    Set the maximum number of entries displayed when printing an array
    '''
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
            raise IndexError("%s.__getitem__(): entry %i is out of bounds "
                             "(the array is of size %i)." % (type(self), index, size))

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
    if _dr.size_v(index) <= 1 and size == Dynamic:
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
        if _dr.is_struct_v(dtype):
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
        if _dr.is_struct_v(target_type):
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
            if _dr.size_v(target) != _dr.size_v(value):
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
        if _dr.is_struct_v(target_type):
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
            if _dr.size_v(target) != _dr.size_v(value):
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

    target = empty(target_type, prod(s))

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
    '''
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
    '''
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
    '''
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
    '''
    t = type(value)
    if _dr.depth_v(t) > 1 or issubclass(t, _Sequence):
        size = len(value)
        result = [None] * size
        for i in range(size):
            result[i] = _dr.slice(value[i], index)
        return result
    elif _dr.is_struct_v(value):
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
        raise TypeError("Use the floor division operator \"//\" for "
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
        raise TypeError("Use the true division operator \"/\" for "
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
    '''
    Return the element-wise comparison of the two arguments

    This function falls back to ``==`` when none of the arguments are Dr.Jit
    arrays.

    Args:
        a (object): Input array.

        b (object): Input array.

    Returns:
        object: Output array, element-wise comparison of ``a`` and ``b``
    '''
    if isinstance(a, ArrayBase) or isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.eq_(b)
    else:
        return a == b


def neq(a, b):
    '''
    Return the element-wise not-equal comparison of the two arguments

    This function falls back to ``!=`` when none of the arguments are Dr.Jit
    arrays.

    Args:
        a (object): Input array.

        b (object): Input array.

    Returns:
        object: Output array, element-wise comparison of ``a != b``
    '''
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


def abs(arg, /):
    '''
    abs(arg, /)
    Compute the absolute value of the provided input.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        float | int | drjit.ArrayBase: Absolute value of the input)
    '''
    if isinstance(arg, ArrayBase):
        return arg.abs_()
    else:
        return _builtins.abs(arg)


def floor(arg, /):
    '''
    floor(arg, /)
    Evaluate the floor, i.e. the largest integer <= arg.

    The function does not convert the type of the input array. A separate
    cast is necessary when integer output is desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Floor of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.floor_()
    else:
        return _math.floor(arg)


def ceil(arg, /):
    '''
    ceil(arg, /)
    Evaluate the ceiling, i.e. the smallest integer >= arg.

    The function does not convert the type of the input array. A separate
    cast is necessary when integer output is desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Ceiling of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.ceil_()
    else:
        return _math.ceil(arg)


def round(arg, /):
    '''
    round(arg, /)

    Rounds arg to the nearest integer using Banker's rounding for
    half-way values.

    This function is equivalent to ``std::rint`` in C++. It does not convert the
    type of the input array. A separate cast is necessary when integer output is
    desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Rounded result
    '''
    if isinstance(arg, ArrayBase):
        return arg.round_()
    else:
        return round(arg)


def trunc(arg, /):
    '''
    trunc(arg, /)
    Truncates arg to the nearest integer by towards zero.

    The function does not convert the type of the input array. A separate
    cast is necessary when integer output is desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Truncated result
    '''
    if isinstance(arg, ArrayBase):
        return arg.trunc_()
    else:
        return _math.trunc(arg)


def mulhi(a, b):
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.mulhi_(b)
    else:
        raise Exception("mulhi(): undefined for Python integers!")


def sqr(a):
    '''
    sqr(arg, /)
    Evaluate the square of the provided input.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square of the input
    '''
    return a * a


def sqrt(arg, /):
    '''
    sqrt(arg, /)
    Evaluate the square root of the provided input.

    Negative inputs produce a *NaN* output value.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.sqrt_()
    else:
        return _math.sqrt(arg)


def rcp(arg, /):
    '''
    rcp(arg, /)
    Evaluate the reciprocal (1 / arg) of the provided input.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU"). The result is slightly
    approximate in this case (refer to the documentation of the instruction
    `rcp.approx.ftz.f32` in the NVIDIA PTX manual for details).

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Reciprocal of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.rcp_()
    else:
        return 1.0 / arg


def rsqrt(arg, /):
    '''
    rsqrt(arg, /)
    Evaluate the reciprocal square root (1 / sqrt(arg)) of the provided input.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU"). The result is slightly
    approximate in this case (refer to the documentation of the instruction
    `rsqrt.approx.ftz.f32` in the NVIDIA PTX manual for details).

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Reciprocal square root of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.rsqrt_()
    else:
        return 1.0 / _math.sqrt(arg)


def maximum(a, b, /):
    '''
    maximum(arg0, arg1, /) -> float | int | drjit.ArrayBase
    Compute the element-wise maximum value of the provided inputs.

    This function returns a result of the type ``type(arg0 + arg1)`` (i.e.,
    according to the usual implicit type conversion rules).

    Args:
        arg0 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
        arg1 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Maximum of the input(s)
    '''
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.maximum_(b)
    else:
        return _builtins.max(a, b)


def minimum(a, b, /):
    '''
    minimum(arg0, arg1, /) -> float | int | drjit.ArrayBase
    Compute the element-wise minimum value of the provided inputs.

    This function returns a result of the type ``type(arg0 + arg1)`` (i.e.,
    according to the usual implicit type conversion rules).

    Args:
        arg0 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
        arg1 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Minimum of the input(s)
    '''
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.minimum_(b)
    else:
        return _builtins.min(a, b)


def fma(a, b, c, /):
    '''
    fma(arg0, arg1, arg2, /)
    Perform a *fused multiply-add* (FMA) operation.

    Given arguments ``arg0``, ``arg1``, and ``arg2``, this operation computes
    ``arg0`` * ``arg1`` + ``arg2`` using only one final rounding step. The
    operation is not only more accurate, but also more efficient, since FMA maps to
    a native machine instruction on platforms targeted by Dr.Jit.

    While FMA is traditionally a floating point operation, Dr.Jit also implements
    FMA for integer arrays and maps it onto dedicated instructions provided by the
    backend if possible (e.g. ``mad.lo.*`` for CUDA/PTX).

    Args:
        arg0 (float | drjit.ArrayBase): First multiplication operand
        arg1 (float | drjit.ArrayBase): Second multiplication operand
        arg2 (float | drjit.ArrayBase): Additive operand

    Returns:
        float | drjit.ArrayBase: Result of the FMA operation
    '''
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase) or \
       isinstance(c, ArrayBase):
        if type(a) is not type(b) or type(b) is not type(c):
            a, b, c = _var_promote(a, b, c)
        return a.fma_(b, c)
    else:
        return _dr.detail.fmadd_scalar(a, b, c)


def select(m, t, f, /):
    '''
    select(condition, x, y, /)
    Select elements from inputs based on a condition

    This function implements the component-wise operation

    .. math::

    \\mathrm{result}_i = \\begin{cases}
        x_i,\\quad&\\text{if condition}_i,\\\\
        y_i,\\quad&\\text{otherwise.}
    \\end{cases}

    Args:
        condition (bool | drjit.ArrayBase): A Python or Dr.Jit mask/boolean array
        x (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        y (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | int | drjit.ArrayBase: Component-wise result of the selection operation
    '''
    if isinstance(m, bool):
        return t if m else f
    type_t, type_f, type_m = type(t), type(f), type(m)

    if type_t is not type_f or type_m is not _dr.mask_t(t):
        if type_t is type_f and _dr.is_struct_v(type_t):
            result = type_t()
            for k in type_t.DRJIT_STRUCT.keys():
                setattr(result, k, _dr.select(m, getattr(t, k), getattr(f, k)))
            return result

        m, t, f = _var_promote_select(m, t, f)

    return type(t).select_(m, t, f)


def sign(arg, /):
    '''
    sign(arg, /)
    Return the element-wise sign of the provided array.

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | int | drjit.ArrayBase: Sign of the input array
    '''
    t = type(arg)
    return _dr.select(arg >= 0, t(1), t(-1))


def copysign(a, b, /):
    '''
    copysign(arg0, arg1, /)
    Copy the sign of ``arg1`` to ``arg0` element-wise.

    Args:
        arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to change the sign of
        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to copy the sign from

    Returns:
        float | int | drjit.ArrayBase: The values of ``arg0`` with the sign of ``arg1``
    '''
    a_a = _dr.abs(a)
    return _dr.select(b >= 0, a_a, -a_a)


def mulsign(a, b, /):
    '''
    mulsign(arg0, arg1, /)
    Multiply ``arg0`` by the sign of ``arg1` element-wise.

    This function is equivalent to

    .. code-block::

        a * dr.sign(b)

    Args:
        arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to multiply the sign of
        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit array to take the sign from

    Returns:
        float | int | drjit.ArrayBase: The values of ``arg0`` multiplied with the sign of ``arg1``
    '''
    return _dr.select(b >= 0, a, -a)


def isnan(arg, /):
    '''
    Performs an element-wise test for *NaN* (Not a Number) values

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test.
    '''
    if _dr.is_array_v(arg):
        return ~_dr.eq(arg, arg)
    else:
        return not (arg == arg)


def isinf(arg, /):
    '''
    Performs an element-wise test for positive or negative infinity

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test
    '''
    return _dr.eq(_dr.abs(arg), _dr.inf)


def isfinite(arg, /):
    '''
    Performs an element-wise test that checks whether values are finite and not
    equal to *NaN* (Not a Number)

    Args:
        arg (object): A Dr.Jit array or other kind of numeric sequence type.

    Returns:
        :py:func:`mask_t(arg) <mask_t>`: A mask value describing the result of the test
    '''
    return _dr.abs(arg) < _dr.inf


def lerp(a, b, t, /):
    '''
    lerp(a, b, t, /)
    Blends between the values ``a`` and ``b`` using the expression :math:`a \\cdot (1 - t) + b \\cdot t`

    Args:
        a (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        b (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        t (float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | int | drjit.ArrayBase: Blended value
    '''
    return _dr.fma(b, t, _dr.fma(-a, t, a))


def clip(value, min, max, /):
    '''
    Clip the provided input to the given interval.

    This function is equivalent to

    .. code-block::

        dr.maximum(dr.minimum(value, max), min)

    Dr.Jit also defines :py:func:`drjit.clamp` as an alias of this function.

    Args:
        value (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        min (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        max (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Clipped input
    '''
    return _dr.maximum(_dr.minimum(value, max), min)


def clamp(value, min, max, /):
    '''
    Clip the provided input to the given interval.

    This function is equivalent :py:func:`drjit.clamp`.

    Args:
        value (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        min (int | float | drjit.ArrayBase): A Python or Dr.Jit array
        max (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Clipped input
    '''
    return _dr.clip(value, min, max)


def arg(arg, /):
    '''
    Return the argument of a complex Dr.Jit array.

    When the provided array isn't an instance of :py:class:`drjit.Complex`, this
    function assumes that the input array represents the real part of a complex
    variable.

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Argument of the complex input array
    '''
    if _dr.is_complex_v(arg):
        return _dr.atan2(arg.imag, arg.real)
    else:
        return _dr.select(arg >= 0, 0, -_dr.pi)


def real(arg, /):
    '''
    Return the real part of a complex Dr.Jit array.

    When the provided array isn't an instance of :py:class:`drjit.Complex` or
    :py:class:`drjit.Quaternion`, this function returns the input unchanged.

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Real part of the input array
    '''
    if _dr.is_complex_v(arg):
        return arg[0]
    elif _dr.is_quaternion_v(arg):
        return arg[3]
    else:
        return arg


def imag(arg, /):
    '''
    Return the imaginary part of a complex Dr.Jit array.

    When the provided array isn't an instance of :py:class:`drjit.Complex` or
    :py:class:`drjit.Quaternion`, this function returns the input unchanged.

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Imaginary part of the input array
    '''
    if _dr.is_complex_v(arg):
        return arg[1]
    elif _dr.is_quaternion_v(arg):
        name = _dr.detail.array_name('Array', arg.Type, (3), arg.IsScalar)
        Array3f = getattr(_modules.get(arg.__module__), name)
        return Array3f(arg[0], arg[1], arg[2])
    else:
        return type(arg)(0)


def conj(arg, /):
    '''
    Return the complex conjugate of a provided Dr.Jit array.

    When the provided array isn't an instance of :py:class:`drjit.Complex` or
    :py:class:`drjit.Quaternion`, this function returns the input unchanged.

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        float | drjit.ArrayBase: Real part of the input array
    '''

    if _dr.is_complex_v(arg):
        return type(arg)(arg.real, -arg.imag)
    elif _dr.is_quaternion_v(arg):
        return type(arg)(-arg.x, -arg.y, -arg.z, arg.w)
    else:
        return arg


def tzcnt(arg, /):
    '''
    Return the number of trailing zero bits.

    This function assumes that ``arg`` is an integer array.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of trailing zero bits in the input array
    '''
    if isinstance(arg, ArrayBase):
        return arg.tzcnt_()
    else:
        # The following assumes that 'a' is a 32 bit integer
        assert arg >= 0 and arg <= 0xFFFFFFFF
        result = 32
        while arg & 0xFFFFFFFF:
            result -= 1
            arg <<= 1
        return result


def lzcnt(arg, /):
    '''
    Return the number of leading zero bits.

    This function assumes that ``arg`` is an integer array.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of leading zero bits in the input array
    '''
    if isinstance(arg, ArrayBase):
        return arg.lzcnt_()
    else:
        # The following assumes that 'a' is a 32 bit integer
        assert arg >= 0 and arg <= 0xFFFFFFFF
        result = 32
        while arg:
            result -= 1
            arg >>= 1
        return result


def popcnt(arg, /):
    '''
    Return the number of nonzero zero bits.

    This function assumes that ``arg`` is an integer array.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of nonzero zero bits in the input array
    '''
    if isinstance(arg, ArrayBase):
        return arg.popcnt_()
    else:
        result = 0
        while arg:
            result += arg & 1
            arg >>= 1
        return result


def log2i(arg, /):
    '''
    Return the floor of the base-two logarithm.

    This function assumes that ``arg`` is an integer array.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of leading zero bits in the input array
    '''
    if isinstance(arg, ArrayBase):
        return (arg.Type.Size * 8 - 1) - _dr.lzcnt(arg)
    else:
        return 31 - _dr.lzcnt(arg)


# -------------------------------------------------------------------
#   "Safe" functions that avoid domain errors due to rounding
# -------------------------------------------------------------------


def safe_sqrt(arg):
    '''
    Safely evaluate the square root of the provided input avoiding domain errors.

    Negative inputs produce a ``0.0`` output value.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input
    '''
    result = _dr.sqrt(_dr.maximum(arg, 0))
    if _dr.is_diff_v(arg) and _dr.grad_enabled(arg):
        alt = _dr.sqrt(_dr.maximum(arg, _dr.epsilon(arg)))
        result = _dr.replace_grad(result, alt)
    return result


def safe_asin(arg):
    '''
    Safe wrapper around :py:func:`drjit.asin` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arcsine approximation
    '''
    result = _dr.asin(_dr.clamp(arg, -1, 1))
    if _dr.is_diff_v(arg) and _dr.grad_enabled(arg):
        alt = _dr.asin(_dr.clamp(arg, -_dr.one_minus_epsilon(arg), _dr.one_minus_epsilon(arg)))
        result = _dr.replace_grad(result, alt)
    return result


def safe_acos(arg):
    '''
    Safe wrapper around :py:func:`drjit.acos` that avoids domain errors.

    Input values are clipped to the :math:`(-1, 1)` domain.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arccosine approximation
    '''
    result = _dr.acos(_dr.clamp(arg, -1, 1))
    if _dr.is_diff_v(arg) and _dr.grad_enabled(arg):
        alt = _dr.acos(_dr.clamp(arg, -_dr.one_minus_epsilon(arg), _dr.one_minus_epsilon(arg)))
        result = _dr.replace_grad(result, alt)
    return result


# -------------------------------------------------------------------
#       Vertical operations -- AD/JIT compilation-related
# -------------------------------------------------------------------


def label(arg):
    '''
    Returns the label of a given Dr.Jit array.

    Args:
        arg (object): a Dr.Jit array instance.

    Returns:
        str: the label of the given variable.
    '''
    if isinstance(arg, ArrayBase):
        return arg.label_()
    else:
        return None


def set_label(*args, **kwargs):
    '''
    Sets the label of a provided Dr.Jit array, either in the JIT or the AD system.

    When a :ref:`custom data structure <custom-struct>` is provided, the field names
    will be used as suffix for the variables labels.

    When a sequence or static array is provided, the item's indices will be appended
    to the label.

    When a mapping is provided, the item's key will be appended to the label.

    Args:
        *arg (tuple): a Dr.Jit array instance and its corresponding label ``str`` value.

        **kwarg (dict): A set of (keyword, object) pairs.
    '''
    n_args, n_kwargs = len(args), len(kwargs)
    if (n_kwargs and n_args) or (n_args and n_args != 2):
        raise TypeError('set_label(): incompatible function arguments')

    if n_args:
        a, label = args
        if _dr.is_jit_v(a) or _dr.is_diff_v(a):
            a.set_label_(label)
        elif isinstance(a, _Sequence):
            for i, v in enumerate(a):
                set_label(v, label + "_" + str(i))
        elif isinstance(a, _Mapping):
            for k, v in a.items():
                set_label(v, label + "_" + k)
        elif _dr.is_struct_v(a):
            for k in a.DRJIT_STRUCT.keys():
                set_label(getattr(a, k), label + "_" + k)
    elif n_kwargs:
        for k, v in kwargs.items():
            set_label(v, k)


def schedule(*args):
    '''
    Schedule the provided JIT variable(s) for later evaluation

    This function causes ``args`` to be evaluated by the next kernel launch. In
    other words, the effect of this operation is deferred: the next time that
    Dr.Jit's LLVM or CUDA backends compile and execute code, they will include the
    *trace* of the specified variables in the generated kernel and turn them into
    an explicit memory-based representation.

    Scheduling and evaluation of traced computation happens automatically, hence it
    is rare that a user would need to call this function explicitly. Explicit
    scheduling can improve performance in certain cases---for example, consider the
    following code:

    .. code-block::

        # Computation that produces Dr.Jit arrays
        a, b = ...

        # The following line launches a kernel that computes 'a'
        print(a)

        # The following line launches a kernel that computes 'b'
        print(b)

    If the traces of ``a`` and ``b`` overlap (perhaps they reference computation
    from an earlier step not shown here), then this is inefficient as these steps
    will be executed twice. It is preferable to launch bigger kernels that leverage
    common subexpressions, which is what :py:func:`drjit.schedule()` enables:

    .. code-block::

        a, b = ... # Computation that produces Dr.Jit arrays

        # Schedule both arrays for deferred evaluation, but don't evaluate yet
        dr.schedule(a, b)

        # The following line launches a kernel that computes both 'a' and 'b'
        print(a)

        # References the stored array, no kernel launch
        print(b)

    Note that :py:func:`drjit.eval()` would also have been a suitable alternative
    in the above example; the main difference to :py:func:`drjit.schedule()` is
    that it does the evaluation immediately without deferring the kernel launch.

    This function accepts a variable-length keyword argument and processes it
    as follows:

    - It recurses into sequences (``tuple``, ``list``, etc.)
    - It recurses into the values of mappings (``dict``, etc.)
    - It recurses into the fields of :ref:`custom data structures <custom-struct>`.

    During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
    arrays and incompatible types are ignored. Multiple variables can be
    equivalently scheduled with a single :py:func:`drjit.schedule()` call or a
    sequence of calls to :py:func:`drjit.schedule()`. Variables that are garbage
    collected between the original :py:func:`drjit.schedule()` call and the next
    kernel launch are ignored and will not be stored in memory.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances,
          :ref:`custom data structures <custom-struct>`, sequences, or mappings.
          The function will recursively traverse data structures to discover all
          Dr.Jit arrays.

    Returns:
        bool: ``True`` if a variable was scheduled, ``False`` if the operation did
        not do anything.
    '''
    result = False
    for a in args:
        t = type(a)
        if issubclass(t, ArrayBase):
            result |= a.schedule_()
        elif _dr.is_struct_v(t):
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
    '''
    Immediately evaluate the provided JIT variable(s)

    This function immediately invokes Dr.Jit's LLVM or CUDA backends to compile and
    then execute a kernel containing the *trace* of the specified variables,
    turning them into an explicit memory-based representation. The generated
    kernel(s) will also include previously scheduled computation. The function
    :py:func:`drjit.eval()` internally calls :py:func:`drjit.schedule()`---specifically,

    .. code-block::

        dr.eval(arg_1, arg_2, ...)

    is equivalent to

    .. code-block::

        dr.schedule(arg_1, arg_2, ...)
        dr.eval()

    Variable evaluation happens automatically as needed, hence it is rare that a
    user would need to call this function explicitly. Explicit evaluation can
    slightly improve performance in certain cases (the documentation of
    :py:func:`drjit.schedule()` shows an example of such a use case.)

    This function accepts a variable-length keyword argument and processes it
    as follows:

    - It recurses into sequences (``tuple``, ``list``, etc.)
    - It recurses into the values of mappings (``dict``, etc.)
    - It recurses into the fields of :ref:`custom data structures <custom-struct>`.

    During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
    arrays and incompatible types are ignored.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances,
          :ref:`custom data structures <custom-struct>`, sequences, or mappings.
          The function will recursively traverse data structures to discover all
          Dr.Jit arrays.

    Returns:
        bool: ``True`` if a variable was evaluated, ``False`` if the operation did
        not do anything.
    '''
    if schedule(*args) or len(args) == 0:
        _dr.detail.eval()


def graphviz(as_str=False):
    '''
    Assembles a graphviz diagram for the computational graph trace by the JIT.

    Args:
        as_str (bool): whether the function should return the graphviz object as
            a string representation or not.

    Returns:
        object: the graphviz obj (or its string representation).
    '''
    s = _dr.detail.graphviz()

    if as_str:
        return s

    try:
        from graphviz import Source
        return Source(s)
    except ImportError:
        raise Exception('The "graphviz" Python package not available! Install '
                        'via "python -m pip install graphviz". Alternatively, '
                        'you can call drjit.graphviz(as_str=True) to obtain '
                        'a string representation.')


def migrate(a, type_):
    if _dr.is_jit_v(a):
        return a.migrate_(type_)
    else:
        return a


# -------------------------------------------------------------------
#           Vertical operations -- transcendental functions
# -------------------------------------------------------------------


def sin(arg, /):
    '''
    sin(arg, /)
    Sine approximation based on the CEPHES library.

    The implementation of this function is designed to achieve low error on the domain
    :math:`|x| < 8192` and will not perform as well beyond this range. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Sine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.sin_()
    else:
        return _math.sin(arg)


def cos(arg, /):
    '''
    cos(arg, /)
    Cosine approximation based on the CEPHES library.

    The implementation of this function is designed to achieve low error on the
    domain :math:`|x| < 8192` and will not perform as well beyond this range. See
    the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Cosine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.cos_()
    else:
        return _math.cos(arg)


def sincos(arg, /):
    '''
    sincos(arg, /)
    Sine/cosine approximation based on the CEPHES library.

    The implementation of this function is designed to achieve low error on the
    domain :math:`|x| < 8192` and will not perform as well beyond this range. See
    the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using two operations involving the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Sine and cosine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.sincos_()
    else:
        return (_math.sin(arg), _math.cos(arg))


def tan(arg, /):
    '''
    tan(arg, /)
    Tangent approximation based on the CEPHES library.

    The implementation of this function is designed to achieve low error on the
    domain :math:`|x| < 8192` and will not perform as well beyond this range. See
    the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Tangent of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.tan_()
    else:
        return _math.tan(arg)


def csc(arg, /):
    '''
    csc(arg, /)
    Cosecant approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Cosecant of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.csc_()
    else:
        return 1 / _math.sin(arg)


def sec(arg):
    '''
    sec(arg, /)
    Secant approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Secant of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.sec_()
    else:
        return 1 / _math.cos(arg)


def cot(arg):
    '''
    cot(arg, /)
    Cotangent approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Cotangent of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.cot_()
    else:
        return 1.0 / _math.tan(arg)


def asin(arg, /):
    '''
    asin(arg, /)
    Arcsine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arcsine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.asin_()
    else:
        return _math.asin(arg)


def acos(arg, /):
    '''
    acos(arg, /)
    Arccosine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arccosine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.acos_()
    else:
        return _math.acos(arg)


def atan(arg, /):
    '''
    atan(arg, /)
    Arctangent approximation

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arctangent of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.atan_()
    else:
        return _math.atan(arg)


def atan2(a, b, /):
    '''
    atan2(y, x, /)
    Arctangent of two values

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        y (float | drjit.ArrayBase): A Python or Dr.Jit floating point type
        x (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arctangent of ``y``/``x``, using the argument signs to
        determine the quadrant of the return value
    '''
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b):
            a, b = _var_promote(a, b)
        return a.atan2_(b)
    else:
        return _math.atan2(a, b)


def exp(arg, /):
    '''
    exp(arg, /)
    Natural exponential approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Natural exponential of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.exp_()
    else:
        return _math.exp(arg)


def exp2(arg, /):
    '''
    exp2(arg, /)
    Base-2 exponential approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Base-2 exponential of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.exp2_()
    else:
        return _math.exp(arg * _math.log(2.0))


def log(arg, /):
    '''
    log(arg, /)
    Natural exponential approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Natural logarithm of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.log_()
    else:
        return _math.log(arg)


def log2(arg, /):
    '''
    log2(arg, /)
    Base-2 exponential approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function unit ("MUFU").

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Base-2 logarithm of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.log2_()
    else:
        return _math.log2(arg)


def power(a, b):
    '''
    Raise the first input value to the power given as second input value.

    This function handles both the case of integer and floating-point exponents.
    Moreover, when the exponent is an array, the function will calculate the
    element-wise powers of the input values.

    Args:
        x (int | float | drjit.ArrayBase): A Python or Dr.Jit array type as input value
        y (int | float | drjit.ArrayBase): A Python or Dr.Jit array type as exponent

    Returns:
        int | float | drjit.ArrayBase: input value raised to the power
    '''
    if _dr.is_tensor_v(a):
        return type(a)(a.array.power_(b if not _dr.is_tensor_v(b) else b.array), a.shape)
    if isinstance(a, ArrayBase) or \
       isinstance(b, ArrayBase):
        if type(a) is not type(b) and not \
           (isinstance(b, int) or isinstance(b, float)):
            a, b = _var_promote(a, b)
        return a.power_(b)
    else:
        return _math.pow(a, b)


def op_pow(a, b):
    return power(a, b)


def cbrt(arg, /):
    '''
    cbrt(arg, /)
    Evaluate the cube root of the provided input.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Cube root of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.cbrt_()
    else:
        return _math.pow(arg, 1.0 / 3.0)


def erf(arg, /):
    '''
    Evaluates the error function defined as

    .. math::

        \\mathrm{erf}(x)=\\frac{2}{\\sqrt{\\pi}}\\int_0^x e^{-t^2}\\,\\mathrm{d}t.

    Requires a real-valued input array ``x``.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Value of the error function at the input value
    '''
    if isinstance(arg, ArrayBase):
        return arg.erf_()
    else:
        return _math.erf(arg)


def erfinv(arg, /):
    '''
    Evaluates the inverse of the error function :py:func:`drjit.erf`.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Value of the inverse of the error function at the input value
    '''
    if isinstance(arg, ArrayBase):
        return arg.erfinv_()
    else:
        raise Exception("erfinv(): only implemented for drjit types!")


def lgamma(arg, /):
    '''
    Evaluates the natural logarithm of the Gamma function.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Value of the natural logarithm of the Gamma function at the input value
    '''
    if isinstance(arg, ArrayBase):
        return arg.lgamma_()
    else:
        return _math.lgamma(arg)


def tgamma(arg, /):
    '''
    Evaluates the Gamma function defined as

    .. math::

        \\Gamma(x)=\\int_0^\\infty t^{x-1} e^{-t}\\,\\mathrm{d}t.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Value of the Gamma function at the input value
    '''
    if isinstance(arg, ArrayBase):
        return arg.tgamma_()
    else:
        return _math.gamma(arg)


def sinh(arg, /):
    '''
    sinh(arg, /)
    Hyperbolic sine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic sine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.sinh_()
    else:
        return _math.sinh(arg)


def cosh(arg, /):
    '''
    cosh(arg, /)
    Hyperbolic cosine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic cosine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.cosh_()
    else:
        return _math.cosh(arg)


def sincosh(arg, /):
    '''
    sincosh(arg, /)
    Hyperbolic sine/cosine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Hyperbolic sine and cosine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.sincosh_()
    else:
        return (_math.sinh(arg), _math.cosh(arg))


def tanh(arg, /):
    '''
    tanh(arg, /)
    Hyperbolic tangent approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic tangent of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.tanh_()
    else:
        return _math.tanh(arg)


def asinh(arg, /):
    '''
    asinh(arg, /)
    Hyperbolic arcsine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic arcsine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.asinh_()
    else:
        return _math.asinh(arg)


def acosh(arg, /):
    '''
    acosh(arg, /)
    Hyperbolic arccosine approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic arccosine of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.acosh_()
    else:
        return _math.acosh(arg)


def atanh(arg, /):
    '''
    atanh(arg, /)
    Hyperbolic arctangent approximation based on the CEPHES library.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic arctangent of the input
    '''
    if isinstance(arg, ArrayBase):
        return arg.atanh_()
    else:
        return _math.atanh(arg)


def rad2deg(arg, /):
    '''
    rad2deg(arg, /) -> float | drjit.ArrayBase
    Convert angles from radians to degrees.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        The equivalent angle in degrees.
    '''
    return arg * (180.0 / _dr.pi)


def deg2rad(arg, /):
    '''
    deg2rad(arg, /) -> float | drjit.ArrayBase
    Convert angles from degrees to radians.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        The equivalent angle in radians.
    '''
    return arg * (_dr.pi / 180.0)


# -------------------------------------------------------------------
#                        Spherical harmonics
# -------------------------------------------------------------------


def sh_eval(arg, order, /):
    '''
    Evaluates the real spherical harmonics basis functions up to and including
    order ``order``.

    The directions provided to ``sh_eval`` must be normalized 3D vectors
    (i.e. using Cartesian instead of spherical coordinates).

    This function supports evaluation order up to 10 (e.g. ``order=9``).

    Args:
        arg (drjit.ArrayBase): A 3D Dr.Jit array type for the direction to be evaluated
        order (int): Order of the spherical harmonic evaluation

    Returns:
        list: List of spherical harmonics coefficients
    '''
    if not _dr.is_array_v(arg) or arg.Size != 3:
        raise Exception("sh_eval(): incompatible input!")

    return _dr.float32_array_t(arg).sh_eval_(arg, order)


# -------------------------------------------------------------------
#                       Horizontal operations
# -------------------------------------------------------------------


def shuffle(perm, value):
    '''
    Permute the entries of the provided Dr.Jit array for the indices given in ``perm``.

    The pseudocode for this operation is

    .. code-block:: python

        out = [value[p] for p in perm]

    Args:
        perm (drjit.ArrayBase): A Python list of integers
        value (drjit.ArrayBase): A Dr.Jit array type

    Returns:
        Shuffled input array
    '''
    if not _dr.is_array_v(value) or len(perm) != value.Size:
        raise Exception("shuffle(): incompatible input!")

    result = type(value)()
    for i, j in enumerate(perm):
        result[i] = value[j]

    return result


def compress(mask, /):
    '''
    compress(arg, /) -> int | drjit.ArrayBase
    Compress a mask into a array of nonzero indices.

    This function takes an boolean array as input and then returns the
    indices of nonzero entries.

    .. danger::
        This function internally performs a synchronization step.

    Args:
        arg (bool | drjit.ArrayBase): A Python or Dr.Jit boolean type

    Returns:
        Array of nonzero indices
    '''
    if not _dr.is_mask_v(mask) or not mask.IsDynamic:
        raise Exception("compress(): incompatible input!")
    return mask.compress_()


def count(arg, /):
    '''
    count(arg, /) -> int | drjit.ArrayBase
    Efficiently computes the number of entries whose boolean values
    are ``True``, i.e.

    .. code-block:: python

        (value[0] ? 1 : 0) + ... (value[Size - 1] ? 1 : 0)

    For 1D arrays, ``count()`` returns a result of type ``int``. For
    multidimensional arrays, the horizontal reduction is performed over the
    *outermost* dimension.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit boolean type

    Returns:
        Number of entries whose mask bits are turned on
    '''
    if _var_is_drjit(arg):
        if arg.Type != VarType.Bool:
            raise Exception("count(): input array must be a mask!")
        return arg.count_()
    elif isinstance(arg, bool):
        return 1 if arg else 0
    elif _dr.is_iterable_v(arg):
        result = 0
        for index, value in enumerate(arg):
            if index == 0:
                result = _dr.select(value, 0, 1)
            else:
                result = result + _dr.select(value, 1, 0)
        return result
    else:
        raise Exception("count(): input must be a boolean or an "
                        "iterable containing masks!")


def all(arg, /):
    '''
    all(arg, /) -> bool | drjit.ArrayBase
    Computes whether all input elements evaluate to ``True``.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Boolean array
    '''
    if _var_is_drjit(arg):
        if arg.Type != VarType.Bool:
            raise Exception("all(): input array must be a mask!")
        return arg.all_()
    elif isinstance(arg, bool):
        return arg
    elif _dr.is_iterable_v(arg):
        result = True
        for index, value in enumerate(arg):
            if index == 0:
                result = value
            else:
                result = result & value
        return result
    else:
        raise Exception("all(): input must be a boolean or an "
                        "iterable containing masks!")


def all_nested(arg, /):
    '''
    all_nested(arg, /) -> bool
    Iterates :py:func:`all` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Boolean
    '''
    while True:
        b = _dr.all(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta is bool or len(arg) == 1):
            break
        arg = b
    return arg


def any(arg, /):
    '''
    any(arg, /) -> bool | drjit.ArrayBase
    Computes whether any of the input elements evaluate to ``True``.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.


    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Boolean array
    '''
    if _var_is_drjit(arg):
        if arg.Type != VarType.Bool:
            raise Exception("any(): input array must be a mask!")
        return arg.any_()
    elif isinstance(arg, bool):
        return arg
    elif _dr.is_iterable_v(arg):
        result = False
        for index, value in enumerate(arg):
            if index == 0:
                result = value
            else:
                result = result | value
        return result
    else:
        raise Exception("any(): input must be a boolean or an "
                        "iterable containing masks!")


def any_nested(arg, /):
    '''
    any_nested(arg, /) -> bool
    Iterates :py:func:`any` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Boolean
    '''
    while True:
        b = _dr.any(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta is bool or len(arg) == 1):
            break
        arg = b
    return arg


def none(arg, /):
    '''
    none(arg, /) -> bool | drjit.ArrayBase
    Computes whether none of the input elements evaluate to ``True``.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Boolean array
    '''
    b = _dr.any(arg)
    return not b if isinstance(b, bool) else ~b


def none_nested(arg, /):
    '''
    none_nested(arg, /) -> bool
    Iterates :py:func:`none` until the type of the return value no longer
    changes. This can be used to reduce a nested mask array into a single
    value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Boolean
    '''
    b = _dr.any_nested(arg)
    return not b if isinstance(b, bool) else ~b


def sum(arg, /):
    '''
    sum(arg, /) -> float | int | drjit.ArrayBase
    Compute the sum of all array elements.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Sum of the input
    '''
    if _var_is_drjit(arg):
        return arg.sum_()
    elif isinstance(arg, float) or isinstance(arg, int):
        return arg
    elif _dr.is_iterable_v(arg):
        result = 0
        for index, value in enumerate(arg):
            if index == 0:
                result = value
            else:
                result = result + value
        return result
    else:
        raise Exception("sum(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def sum_nested(arg, /):
    '''
    sum_nested(arg, /) -> float | int
    Iterates :py:func:`sum` until the return value is reduced to a single value.

    This function recursively calls :py:func:`drjit.sum` on all elements of
    the input array in order to reduce the returned value to a single entry.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Sum of the input
    '''
    while True:
        b = _dr.sum(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta in [int, float] or len(arg) == 1):
            break
        arg = b
    return arg


def mean(arg, /):
    '''
    mean(arg, /) -> float | drjit.ArrayBase
    Compute the mean of all array elements.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Mean of the input
    '''
    if hasattr(arg, '__len__'):
        v = _dr.sum(arg)
        return _dr.float_array_t(v)(v) / len(arg)
    else:
        return arg


def mean_nested(arg, /):
    '''
    mean_nested(arg, /) -> float | int
    Iterates :py:func:`mean` until the return value is reduced to a single value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Mean of the input
    '''
    while True:
        b = _dr.mean(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta is float or len(arg) == 1):
            break
        arg = b
    return arg


def prod(arg, /):
    '''
    prod(arg, /) -> float | int | drjit.ArrayBase
    Compute the product of all array elements.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Product of the input
    '''
    if _var_is_drjit(arg):
        return arg.prod_()
    elif isinstance(arg, float) or isinstance(arg, int):
        return arg
    elif _dr.is_iterable_v(arg):
        result = 1
        for index, value in enumerate(arg):
            if index == 0:
                result = value
            else:
                result = result * value
        return result
    else:
        raise Exception("prod(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def prod_nested(arg, /):
    '''
    prod_nested(arg, /) -> float | int
    Iterates :py:func:`prod` until the return value is reduced to a single value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Product of the input
    '''
    while True:
        b = _dr.prod(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta in [int, float] or len(arg) == 1):
            break
        arg = b
    return arg


def max(arg, /):
    '''
    max(arg, /) -> float | int | drjit.ArrayBase
    Compute the maximum value in the provided input.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Maximum of the input
    '''
    if _var_is_drjit(arg):
        return arg.max_()
    elif isinstance(arg, float) or isinstance(arg, int):
        return arg
    elif _dr.is_iterable_v(arg):
        result = None
        for index, value in enumerate(arg):
            if index == 0:
                result = value
            else:
                result = _dr.maximum(result, value)
        if result is None:
            raise Exception("max(): zero-sized array!")
        return result
    else:
        raise Exception("max(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def max_nested(arg, /):
    '''
    max_nested(arg, /) -> float | int
    Iterates :py:func:`max` until the return value is reduced to a single value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Maximum scalar value of the input
    '''
    while True:
        b = _dr.max(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta in [int, float] or len(arg) == 1):
            break
        arg = b
    return arg


def min(arg, /):
    '''
    min(arg, /) -> float | int | drjit.ArrayBase
    Compute the minimum value in the provided input.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Minimum of the input
    '''
    if _var_is_drjit(arg):
        return arg.min_()
    elif isinstance(arg, float) or isinstance(arg, int):
        return arg
    elif _dr.is_iterable_v(arg):
        result = None
        for index, value in enumerate(arg):
            if index == 0:
                result = value
            else:
                result = _dr.minimum(result, value)
        if result is None:
            raise Exception("min(): zero-sized array!")
        return result
    else:
        raise Exception("min(): input must be a boolean or an iterable "
                        "containing arithmetic types!")


def min_nested(arg, /):
    '''
    min_nested(arg, /) -> float | int
    Iterates :py:func:`min` until the return value is reduced to a single value.

    Args:
        arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Minimum scalar value of the input
    '''
    while True:
        b = _dr.min(arg)
        ta, tb = type(arg), type(b)
        if (tb is ta) and (ta in [int, float] or len(arg) == 1):
            break
        arg = b
    return arg


def dot(a, b, /):
    '''
    dot(arg0, arg1, /) -> float | int | drjit.ArrayBase
    Computes the dot product of two arrays.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg0 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

        arg1 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Dot product of inputs
    '''
    if _dr.is_matrix_v(a) or _dr.is_matrix_v(b):
        raise Exception("dot(): input shouldn't be a Matrix!"
                        "The @ operator should be used instead.")

    if type(a) is not type(b):
        a, b = _var_promote(a, b)
    return a.dot_(b)


def abs_dot(a, b):
    '''
    abs_dot(arg0, arg1, /) -> float | int | drjit.ArrayBase
    Computes the absolute value of dot product of two arrays.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg0 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

        arg1 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Absolute value of the dot product of inputs
    '''
    if _dr.is_matrix_v(a) or _dr.is_matrix_v(b):
        raise Exception("abs_dot(): input shouldn't be a Matrix!"
                        "The @ operator should be used instead.")

    return _dr.abs(_dr.dot(a, b))


def squared_norm(arg, /):
    '''
    squared_norm(arg, /) -> float | int | drjit.ArrayBase
    Computes the squared norm of an array.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Squared norm of the input
    '''
    return _dr.dot(arg, arg)


def norm(arg, /):
    '''
    norm(arg, /) -> float | int | drjit.ArrayBase
    Computes the norm of an array.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Norm of the input
    '''
    return _dr.sqrt(_dr.dot(arg, arg))


def normalize(arg, /):
    '''
    normalize(arg, /) -> drjit.ArrayBase
    Normalizes the provided array.

    When the argument is a dynamic array, function performs a horizontal reduction.
    Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
    for details.

    Args:
        arg (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Normalized input array
    '''
    return arg * _dr.rsqrt(_dr.squared_norm(arg))


def hypot(a, b):
    '''
    Computes :math:`\\sqrt{x^2+y^2}` while avoiding overflow and underflow.

    Args:
        arg (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        Hypotenuse value
    '''
    a, b = _dr.abs(a), _dr.abs(b)
    maxval = _dr.maximum(a, b)
    minval = _dr.minimum(a, b)
    ratio = minval / maxval
    inf = _dr.inf

    return _dr.select(
        (a < inf) & (b < inf) & (ratio < inf),
        maxval * _dr.sqrt(_dr.fma(ratio, ratio, 1)),
        a + b
    )


def block_sum(value, size):
    '''
    Sum over elements within blocks

    This function adds all elements of contiguous blocks of size ``size``
    in the input array ``value`` and writes them to the returned array.
    For example, ``a, b, c, d, e, f`` turns into ``a+b, c+d, e+f`` when
    ``size == 2``. The length of the input array must be a multiple of ``size``.

    Args:
        arg (drjit.ArrayBase): A Python or Dr.Jit arithmetic type
        size (int): size of the block

    Returns:
        Sum over elements within blocks
    '''
    if _dr.is_jit_v(value):
        return value.block_sum_(size)
    else:
        raise Exception("block_sum(): requires a JIT array!")


def binary_search(start, end, pred):
    '''
    Perform binary search over a range given a predicate ``pred``, which
    monotonically decreases over this range (i.e. max one ``True`` -> ``False``
    transition).

    Given a (scalar) ``start`` and ``end`` index of a range, this function
    evaluates a predicate ``floor(log2(end-start) + 1)`` times with index
    values on the interval [start, end] (inclusive) to find the first index
    that no longer satisfies it. Note that the template parameter ``Index`` is
    automatically inferred from the supplied predicate. Specifically, the
    predicate takes an index array as input argument. When ``pred`` is ``False``
    for all entries, the function returns ``start``, and when it is ``True`` for
    all cases, it returns ``end``.

    The following code example shows a typical use case: ``data`` contains a
    sorted list of floating point numbers, and the goal is to map floating
    point entries of ``x`` to the first index ``j`` such that ``data[j] >= threshold``
    (and all of this of course in parallel for each vector element).

    .. code-block::

        dtype = dr.llvm.Float
        data = dtype(...)
        threshold = dtype(...)

        index = dr.binary_search(
            0, len(data) - 1,
            lambda index: dr.gather(dtype, data, index) < threshold
        )

    Args:
        start (int): Starting index for the search range
        end (int): Ending index for the search range
        pred (function): The predicate function to be evaluated

    Returns:
        Index array resulting from the binary search
    '''
    assert isinstance(start, int) and isinstance(end, int)

    iterations = _dr.log2i(end - start) + 1 if start < end else 0

    for i in range(iterations):
        middle = (start + end) >> 1

        cond = pred(middle)
        start = _dr.select(cond, _dr.minimum(middle + 1, end), start)
        end = _dr.select(cond, end, middle)

    return start


# -------------------------------------------------------------------
#    Transformations, matrices, operations for 3D vector spaces
# -------------------------------------------------------------------

def cross(a, b, /):
    '''
    Returns the cross-product of the two input 3D arrays

    Args:
        arg0 (list | drjit.ArrayBase): A Python or Dr.Jit 3D type
        arg1 (list | drjit.ArrayBase): A Python or Dr.Jit 3D type

    Returns:
        Cross-product of the two input 3D arrays
    '''
    if _dr.size_v(a) != 3 or _dr.size_v(a) != 3:
        raise Exception("cross(): requires 3D input arrays!")

    ta, tb = type(a), type(b)

    return _dr.fma(ta(a.y, a.z, a.x), tb(b.z, b.x, b.y),
                   -ta(a.z, a.x, a.y) * tb(b.y, b.z, b.x))


# -------------------------------------------------------------------
#                     Automatic differentiation
# -------------------------------------------------------------------


def detach(arg, preserve_type=True):
    '''
    Transforms the input variable into its non-differentiable version (*detaches* it
    from the AD computational graph).

    This function is able to traverse data-structures such a sequences, mappings or
    :ref:`custom data structure <custom-struct>` and applies the transformation to the
    underlying variables.

    When the input variable isn't a Dr.Jit differentiable array, it is returned as it is.

    While the type of the returned array is preserved by default, it is possible to
    set the ``preserve_type`` argument to false to force the returned type to be
    non-differentiable.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequence, or mapping.

        preserve_type (bool): Defines whether the returned variable should preserve
          the type of the input variable.
    Returns:
        object: The detached variable.
    '''
    if _dr.is_diff_v(arg):
        if preserve_type:
            return type(arg)(arg.detach_())
        else:
            return arg.detach_()
    elif _dr.is_struct_v(arg):
        result = type(arg)()
        if not preserve_type:
            raise TypeError("detach(): preserve_type=True is required when "
                            "detaching custom data structures!")
        for k in type(arg).DRJIT_STRUCT.keys():
            setattr(result, k, detach(getattr(arg, k),
                preserve_type=preserve_type))
        return result
    else:
        return arg


def grad(arg, preserve_type=True):
    '''
    Return the gradient value associated to a given variable.

    When the variable doesn't have gradient tracking enabled, this function returns ``0``.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequences, or mapping.

        preserve_type (bool): Defines whether the returned variable should preserve
          the type of the input variable.

    Returns:
        object: the gradient value associated to the input variable.
    '''
    if _dr.is_diff_v(arg):
        if _dr.is_integral_v(arg):
            grads = _dr.zeros(_dr.detached_t(type(arg)))
        else:
            grads = arg.grad_()

        if preserve_type:
            return type(arg)(grads)
        else:
            return grads
    elif _dr.is_struct_v(arg):
        result = type(arg)()
        if not preserve_type:
            raise TypeError("grad(): preserve_type=True is required when "
                            "getting the gradient of a custom data structure!")
        for k in type(arg).DRJIT_STRUCT.keys():
            setattr(result, k, grad(getattr(arg, k), preserve_type))
        return result
    elif isinstance(arg, _Sequence):
        return type(arg)([grad(v, preserve_type) for v in arg])
    elif isinstance(arg, _Mapping):
        return {k : grad(v, preserve_type) for k, v in arg.items()}
    else:
        return _dr.zeros(type(arg))


def set_grad(dst, src):
    '''
    Set the gradient value to the provided variable.

    Broadcasting is applied to the gradient value if necessary and possible to match
    the type of the input variable.

    Args:
        dst (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequences, or mapping.

        src (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequences, or mapping.
    '''
    if _dr.is_diff_v(dst) and dst.IsFloat:
        if _dr.is_diff_v(src):
            src = _dr.detach(src, preserve_type=False)

        t = _dr.detached_t(dst)
        if type(src) is not t:
            src = t(src)

        dst.set_grad_(src)
    elif isinstance(dst, _Sequence):
        vs = isinstance(src, _Sequence)
        if vs and len(dst) != len(src):
            raise RuntimeError("set_grad(): argument sizes are not matching "
                            "({len(dst)}, {len(src)})")
        for i in range(len(dst)):
            set_grad(dst[i], src[i] if vs else src)
    elif isinstance(dst, _Mapping):
        vm = isinstance(src, _Mapping)
        assert not vm or dst.keys() == src.keys()
        for k, v in dst.items():
            set_grad(v, src[k] if vm else src)
    elif _dr.is_struct_v(dst):
        ve = _dr.is_struct_v(src)
        assert not ve or type(src) is type(dst)
        for k in type(dst).DRJIT_STRUCT.keys():
            set_grad(getattr(dst, k), getattr(src, k) if ve else src)


def accum_grad(dst, src):
    '''
    Accumulate into the gradient of a variable.

    Broadcasting is applied to the gradient value if necessary and possible to match
    the type of the input variable.

    Args:
        dst (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequences, or mapping.

        src (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequences, or mapping.
    '''
    if _dr.is_diff_v(dst) and dst.IsFloat:
        if _dr.is_diff_v(src):
            src = _dr.detach(src, preserve_type=False)

        t = _dr.detached_t(dst)
        if type(src) is not t:
            src = t(src)

        dst.accum_grad_(src)
    elif isinstance(dst, _Sequence):
        vs = isinstance(src, _Sequence)
        if vs and len(dst) != len(src):
            raise RuntimeError("accum_grad(): argument sizes are not matching "
                            "({len(dst)}, {len(src)})")
        for i in range(len(dst)):
            accum_grad(dst[i], src[i] if vs else src)
    elif isinstance(dst, _Mapping):
        vm = isinstance(src, _Mapping)
        assert not vm or dst.keys() == src.keys()
        for k, v in dst.items():
            accum_grad(v, src[k] if vm else src)
    elif _dr.is_struct_v(dst):
        ve = _dr.is_struct_v(src)
        assert not ve or type(src) is type(dst)
        for k in type(dst).DRJIT_STRUCT.keys():
            accum_grad(getattr(dst, k), getattr(src, k) if ve else src)


def grad_enabled(*args):
    '''
    Return whether gradient tracking is enabled on any of the given variables.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances,
          :ref:`custom data structures <custom-struct>`, sequences, or mappings.
          The function will recursively traverse data structures to discover all
          Dr.Jit arrays.

    Returns:
        bool: ``True`` if any variable has gradient tracking enabled, ``False`` otherwise.
    '''
    result = False
    for a in args:
        if _dr.is_diff_v(a):
            result |= a.grad_enabled_()
        elif _dr.is_struct_v(a):
            for k in type(a).DRJIT_STRUCT.keys():
                result |= grad_enabled(getattr(a, k))
        elif isinstance(a, _Sequence):
            for v in a:
                result |= grad_enabled(v)
        elif isinstance(a, _Mapping):
            for k, v in a.items():
                result |= grad_enabled(v)
    return result


def set_grad_enabled(arg, value):
    '''
    Enable or disable gradient tracking on the provided variables.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor,
          :ref:`custom data structure <custom-struct>`, sequence, or mapping.

        value (bool): Defines whether gradient tracking should be enabled or
          disabled.
    '''
    if _dr.is_diff_v(arg) and arg.IsFloat:
        arg.set_grad_enabled_(value)
    elif _dr.is_struct_v(arg):
        for k in type(arg).DRJIT_STRUCT.keys():
            set_grad_enabled(getattr(arg, k), value)
    elif isinstance(arg, _Sequence):
        for v in arg:
            set_grad_enabled(v, value)
    elif isinstance(arg, _Mapping):
        for k, v in arg.items():
            set_grad_enabled(v, value)


def enable_grad(*args):
    '''
    Enable gradient tracking for the provided variables.

    This function accepts a variable-length list of arguments and processes it
    as follows:

    - It recurses into sequences (``tuple``, ``list``, etc.)
    - It recurses into the values of mappings (``dict``, etc.)
    - It recurses into the fields of :ref:`custom data structures <custom-struct>`.

    During recursion, the function enables gradient tracking for all Dr.Jit arrays.
    For every other types, this function won't do anything.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances,
          :ref:`custom data structures <custom-struct>`, sequences, or mappings.
    '''
    for arg in args:
        set_grad_enabled(arg, True)


def disable_grad(*args):
    '''
    Disable gradient tracking for the provided variables.

    This function accepts a variable-length list of arguments and processes it
    as follows:

    - It recurses into sequences (``tuple``, ``list``, etc.)
    - It recurses into the values of mappings (``dict``, etc.)
    - It recurses into the fields of :ref:`custom data structures <custom-struct>`.

    During recursion, the function disables gradient tracking for all Dr.Jit arrays.
    For every other types, this function won't do anything.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances,
          :ref:`custom data structures <custom-struct>`, sequences, or mappings.
    '''
    for arg in args:
        set_grad_enabled(arg, False)


def replace_grad(dst, src):
    '''
    Replace the gradient value of ``dst`` with the one of ``src``.

    Broadcasting is applied to ``dst`` if necessary to match the type of ``src``.

    Args:
        dst (object): An arbitrary Dr.Jit array, tensor, or scalar builtin instance.

        src (object): An differentiable Dr.Jit array or tensor.

    Returns:
        object: the variable with the replaced gradients.
    '''
    tdst, tsrc = type(dst), type(src)

    if not (_dr.is_diff_v(tsrc) and tsrc.IsFloat):
        raise TypeError("replace_grad(): unsupported input types!")

    if type(dst) is not type(src):
        dst, src = _var_promote(dst, src)
    tdst, tsrc = type(dst), type(src)

    if not (_dr.is_diff_v(tdst) and tdst.IsFloat and
            _dr.is_diff_v(tsrc) and tsrc.IsFloat):
        raise Exception("replace_grad(): unsupported input types!")

    ldst, lsrc = len(dst), len(src)
    depth = dst.Depth # matches b.Depth

    if ldst != lsrc:
        if ldst == 1 and depth == 1:
            dst = dst + _dr.zeros(tdst, lsrc)
        elif lsrc == 1 and depth == 1:
            src = src + _dr.zeros(tsrc, ldst)
        else:
            raise Exception("replace_grad(): input arguments have "
                            "incompatible sizes (%i vs %i)!"
                            % (ldst, lsrc))

    if depth > 1:
        result = tdst()
        if dst.Size == Dynamic:
            result.init_(ldst)
        for i in range(ldst):
            result[i] = replace_grad(dst[i], src[i])
        return result
    else:
        if _dr.is_tensor_v(dst):
            return tdst(replace_grad(dst.array, src.array), dst.shape)
        else:
            return tdst.create_(src.index_ad, dst.detach_())


def enqueue(mode, *args):
    '''
    Enqueues variable for the subsequent AD traversal.

    In Dr.Jit, the process of automatic differentiation is split into two parts:

    1. Discover and enqueue the variables to be considered as inputs during the
       subsequent AD traversal.
    2. Traverse the AD graph starting from the enqueued variables to propagate the
       gradients towards the output variables (e.g. leaf in the AD graph).


    This function handles the first part can operate in different modes depending on
    the specified ``mode``:

    - ``ADMode.Forward``: the provided ``value`` will be considered as input during
      the subsequent AD traversal.

    - ``ADMode.Backward``: a traversal of the AD graph starting from the provided
      ``value`` will take place to find all potential source of gradients and
      enqueue them.

    For example, a typical chain of operations to forward propagate the gradients
    from ``a`` to ``b`` would look as follow:

    .. code-block::

        a = dr.llvm.ad.Float(1.0)
        dr.enable_grad(a)
        b = f(a) # some computation involving `a`
        dr.set_gradient(a, 1.0)
        dr.enqueue(dr.ADMode.Forward, a)
        dr.traverse(dr.llvm.ad.Float, dr.ADMode.Forward)
        grad = dr.grad(b)

    It could be the case that ``f(a)`` involves other differentiable variables that
    already contain some gradients. In this situation we can use ``ADMode.Backward``
    to discover and enqueue them before the traversal.

    .. code-block::

        a = dr.llvm.ad.Float(1.0)
        dr.enable_grad(a)
        b = f(a, ...) # some computation involving `a` and some hidden variables
        dr.set_gradient(a, 1.0)
        dr.enqueue(dr.ADMode.Backward, b)
        dr.traverse(dr.llvm.ad.Float, dr.ADMode.Forward)
        grad = dr.grad(b)

    Dr.Jit also provides a higher level API that encapsulate this logic in a few
    different functions:

    - :py:func:`drjit.forward_from`, :py:func:`drjit.forward`, :py:func:`drjit.forward_to`
    - :py:func:`drjit.backward_from`, :py:func:`drjit.backward`, :py:func:`drjit.backward_to`

    Args:
        mode (ADMode): defines the enqueuing mode (backward or forward)

        *args (tuple): A variable-length list of Dr.Jit array instances, tensors,
          :ref:`custom data structures <custom-struct>`, sequences, or mappings.
    '''
    for a in args:
        if _dr.is_diff_v(a) and a.IsFloat:
            a.enqueue_(mode)
        elif isinstance(a, _Sequence):
            for v in a:
                enqueue(mode, v)
        elif isinstance(a, _Mapping):
            for k, v in a.items():
                enqueue(mode, v)
        elif _dr.is_struct_v(a):
            for k in type(a).DRJIT_STRUCT.keys():
                enqueue(mode, getattr(a, k))


def traverse(dtype, mode, flags=_dr.ADFlag.Default):
    '''
    Propagate derivatives through the enqueued set of edges in the AD computational
    graph in the direction specified by ``mode``.

    By default, Dr.Jit's AD system destructs the enqueued input graph during AD
    traversal. This frees up resources, which is useful when working with large
    wavefronts or very complex computation graphs. However, this also prevents
    repeated propagation of gradients through a shared subgraph that is being
    differentiated multiple times.

    To support more fine-grained use cases that require this, the following flags
    can be used to control what should and should not be destructed:

    - ``ADFlag.ClearNone``: clear nothing
    - ``ADFlag.ClearEdges``: delete all traversed edges from the computation graph
    - ``ADFlag.ClearInput``: clear the gradients of processed input vertices (in-degree == 0)
    - ``ADFlag.ClearInterior``: clear the gradients of processed interior vertices (out-degree != 0)
    - ``ADFlag.ClearVertices``: clear gradients of processed vertices only, but leave edges intact
    - ``ADFlag.Default``: clear everything (default behaviour)

    Args:
        dtype (type): defines the Dr.JIT array type used to build the AD graph

        mode (ADMode): defines the mode traversal (backward or forward)

        flags (ADFlag | int): flags to control what should and should not be
          destructed during forward/backward mode traversal.
    '''
    assert isinstance(mode, _dr.ADMode)

    dtype = _dr.leaf_array_t(dtype)

    if not _dr.is_diff_v(dtype):
        raise Exception('traverse(): expected a differentiable array type!')

    dtype.traverse_(mode, flags)


def _check_grad_enabled(name, t, a):
    if _dr.is_diff_v(t) and t.IsFloat:
        if _dr.flag(_dr.JitFlag.VCallRecord) and not grad_enabled(a):
            raise TypeError(
                f'{name}(): the argument does not depend on the input '
                'variable(s) being differentiated. Raising an exception '
                'since this is usually indicative of a bug (for example, '
                'you may have forgotten to call dr.enable_grad(..)). If '
                f'this is expected behavior, skip the call to {name}(..) '
                'if ek.grad_enabled(..) returns False.')
    else:
        raise TypeError(f'{name}(): expected a differentiable array type!')


def forward_from(arg, flags=_dr.ADFlag.Default):
    '''
    Forward propagates gradients from a provided Dr.Jit differentiable array.

    This function will first see the gradient value of the provided variable to ``1.0``
    before executing the AD graph traversal.

    An exception will be raised when the provided array doesn't have gradient tracking
    enabled or if it isn't an instance of a Dr.Jit differentiable array type.

    Args:
        arg (object): A Dr.Jit differentiable array instance.

        flags (ADFlag | int): flags to control what should and should not be
          destructed during the traversal. The default value is ``ADFlag.Default``.
    '''
    ta = type(arg)
    _check_grad_enabled('forward_from', ta, arg)
    set_grad(arg, 1)
    enqueue(_dr.ADMode.Forward, arg)
    traverse(ta, _dr.ADMode.Forward, flags)


def forward_to(*args, flags=_dr.ADFlag.Default):
    '''
    Forward propagates gradients to a set of provided Dr.Jit differentiable arrays.

    Internally, the AD computational graph will be first traversed backward to find
    all potential source of gradient for the provided array. Then only the forward
    gradient propagation traversal takes place.

    The ``flags`` argument should be provided as a keyword argument for this function.

    An exception will be raised when the provided array doesn't have gradient tracking
    enabled or if it isn't an instance of a Dr.Jit differentiable array type.

    Args:
        *args (tuple): A variable-length list of Dr.Jit differentiable array, tensor,
            :ref:`custom data structure <custom-struct>`, sequences, or mapping.

        flags (ADFlag | int): flags to control what should and should not be
          destructed during the traversal. The default value is ``ADFlag.Default``.

    Returns:
        object: the gradient value associated to the output variables.
    '''
    for a in args:
        if isinstance(a, (int, _dr.ADFlag)):
            raise TypeError('forward_to(): AD flags should be passed via '
                            'the "flags=.." keyword argument')

    ta = _dr.leaf_array_t(args)
    _check_grad_enabled('forward_to', ta, args)
    enqueue(_dr.ADMode.Backward, *args)
    traverse(ta, _dr.ADMode.Forward, flags)

    return grad(args) if len(args) > 1 else grad(*args)


def forward(arg, flags=_dr.ADFlag.Default):
    '''
    Forward propagates gradients from a provided Dr.Jit differentiable array.

    This function will first see the gradient value of the provided variable to ``1.0``
    before executing the AD graph traversal.

    An exception will be raised when the provided array doesn't have gradient tracking
    enabled or if it isn't an instance of a Dr.Jit differentiable array type.

    This function is an alias of :py:func:`drjit.forward_from`.

    Args:
        arg (object): A Dr.Jit differentiable array instance.

        flags (ADFlag | int): flags to control what should and should not be
          destructed during the traversal. The default value is ``ADFlag.Default``.
    '''
    forward_from(arg, flags)


def backward_from(arg, flags=_dr.ADFlag.Default):
    '''
    Backward propagates gradients from a provided Dr.Jit differentiable array.

    An exception will be raised when the provided array doesn't have gradient tracking
    enabled or if it isn't an instance of a Dr.Jit differentiable array type.

    Args:
        arg (object): A Dr.Jit differentiable array instance.

        flags (ADFlag | int): flags to control what should and should not be
          destructed during the traversal. The default value is ``ADFlag.Default``.
    '''
    ta = type(arg)
    _check_grad_enabled('backward_from', ta, arg)

    # Deduplicate components if 'a' is a vector
    if _dr.depth_v(arg) > 1:
        arg = arg + ta(0)

    set_grad(arg, 1)
    enqueue(_dr.ADMode.Backward, arg)
    traverse(ta, _dr.ADMode.Backward, flags)


def backward_to(*args, flags=_dr.ADFlag.Default):
    '''
    Backward propagate gradients to a set of provided Dr.Jit differentiable arrays.

    Internally, the AD computational graph will be first traversed *forward* to find
    all potential source of gradient for the provided array. Then only the backward
    gradient propagation traversal takes place.

    The ``flags`` argument should be provided as a keyword argument for this function.

    An exception will be raised when the provided array doesn't have gradient tracking
    enabled or if it isn't an instance of a Dr.Jit differentiable array type.

    Args:
        *args (tuple): A variable-length list of Dr.Jit differentiable array, tensor,
          :ref:`custom data structure <custom-struct>`, sequences, or mapping.

        flags (ADFlag | int): flags to control what should and should not be
          destructed during the traversal. The default value is ``ADFlag.Default``.

    Returns:
        object: the gradient value associated to the output variables.
    '''
    for a in args:
        if isinstance(a, (int, _dr.ADFlag)):
            raise Exception('backward_to(): AD flags should be passed via '
                            'the "flags=.." keyword argument')

    ta = _dr.leaf_array_t(args)
    _check_grad_enabled('backward_to', ta, args)
    enqueue(_dr.ADMode.Forward, *args)
    traverse(ta, _dr.ADMode.Backward, flags)

    return grad(args) if len(args) > 1 else grad(*args)


def backward(arg, flags=_dr.ADFlag.Default):
    '''
    Backward propagate gradients from a provided Dr.Jit differentiable array.

    An exception will be raised when the provided array doesn't have gradient tracking
    enabled or if it isn't an instance of a Dr.Jit differentiable array type.

    This function is an alias of :py:func:`drjit.backward_from`.

    Args:
        arg (object): A Dr.Jit differentiable array instance.

        flags (ADFlag | int): flags to control what should and should not be
          destructed during the traversal. The default value is ``ADFlag.Default``.
    '''
    backward_from(arg, flags)


# -------------------------------------------------------------------
#                      Initialization operations
# -------------------------------------------------------------------


def zeros(dtype, shape=1):
    '''
    Return a zero-initialized instance of the desired type and shape

    This function can create zero-initialized instances of various types. In
    particular, ``dtype`` can be:

    - A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
      specifies a sequence, it must be compatible with static dimensions of the
      ``dtype``. For example, ``dr.zeros(dr.cuda.Array2f, shape=(3, 100))`` fails,
      since the leading dimension is incompatible with
      :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
      the size of the last (dynamic) dimension, if available.

    - A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
      specifies a sequence (list/tuple/..), it determines the tensor rank and
      shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
      the specified size.

    - A :ref:`custom data structure <custom-struct>`. In this case,
      :py:func:`drjit.zeros()` will invoke itself recursively to zero-initialize
      each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
      parameter is ignored in this case.

    Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
    initialized to ``False`` as opposed to zero.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`custom data structure <custom-struct>`.

        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A zero-initialized instance of type ``dtype``.
    '''
    if not isinstance(dtype, type):
        raise Exception('zeros(): Type expected as first argument')
    elif issubclass(dtype, ArrayBase):
        return dtype.zero_(shape)
    elif _dr.is_struct_v(dtype):
        result = dtype()
        for k, v in dtype.DRJIT_STRUCT.items():
            setattr(result, k, zeros(v, shape))
        if hasattr(dtype, 'zero_'):
            result.zero_(shape)
        return result
    elif not dtype in (int, float, complex, bool):
        return None
    else:
        return dtype(0)


def empty(dtype, shape=1):
    '''
    Return an uninitialized Dr.Jit array of the desired type and shape.

    This function can create uninitialized buffers of various types. It is
    essentially a wrapper around CPU/GPU variants of ``malloc()`` and produces
    arrays filled with uninitialized/undefined data. It should only be used in
    combination with a subsequent call to an operation like
    :py:func:`drjit.scatter()` that overwrites the array contents with valid data.

    The ``dtype`` parameter can be used to request:

    - A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
      specifies a sequence, it must be compatible with static dimensions of the
      ``dtype``. For example, ``dr.empty(dr.cuda.Array2f, shape=(3, 100))`` fails,
      since the leading dimension is incompatible with
      :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
      the size of the last (dynamic) dimension, if available.

    - A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
      specifies a sequence (list/tuple/..), it determines the tensor rank and
      shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
      the specified size.

    - A :ref:`custom data structure <custom-struct>`. In this case,
      :py:func:`drjit.empty()` will invoke itself recursively to allocate memory
      for each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
      parameter is ignored in this case, and the function returns a
      zero-initialized result (there is little point in instantiating uninitialized
      versions of scalar Python types).

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`custom data structure <custom-struct>`.

        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: An instance of type ``dtype`` with arbitrary/undefined contents.
    '''
    if not isinstance(dtype, type):
        raise Exception('empty(): Type expected as first argument')
    elif issubclass(dtype, ArrayBase):
        return dtype.empty_(shape)
    elif _dr.is_struct_v(dtype):
        result = dtype()
        for k, v in dtype.DRJIT_STRUCT.items():
            setattr(result, k, empty(v, shape))
        return result
    else:
        return dtype(0)


def full(dtype, value, shape=1):
    '''
    Return a constant-valued instance of the desired type and shape

    This function can create constant-valued instances of various types. In
    particular, ``dtype`` can be:

    - A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
    specifies a sequence, it must be compatible with static dimensions of the
    ``dtype``. For example, ``dr.full(dr.cuda.Array2f, value=1.0, shape=(3,
    100))`` fails, since the leading dimension is incompatible with
    :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
    the size of the last (dynamic) dimension, if available.

    - A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
    specifies a sequence (list/tuple/..), it determines the tensor rank and
    shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
    the specified size.

    - A :ref:`custom data structure <custom-struct>`. In this case,
    :py:func:`drjit.full()` will invoke itself recursively to initialize
    each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
    parameter is ignored in this case.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`custom data structure <custom-struct>`.

        value (object): An instance of the underlying scalar type
          (``float``/``int``/``bool``, etc.) that will be used to initialize the
          array contents.

        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A instance of type ``dtype`` filled with ``value``
    '''
    if not isinstance(dtype, type):
        raise Exception('full(): Type expected as first argument')
    elif issubclass(dtype, ArrayBase):
        return dtype.full_(value, shape)
    else:
        return dtype(value)


def ones(dtype, shape=1):
    '''
    Return a constant-valued instance of the desired type and shape filled with ones.

    This function can create constant-valued instances of various types. In
    particular, ``dtype`` can be:

    - A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
    specifies a sequence, it must be compatible with static dimensions of the
    ``dtype``. For example, ``dr.ones(dr.cuda.Array2f, shape=(3, 100))`` fails,
    since the leading dimension is incompatible with
     :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
    the size of the last (dynamic) dimension, if available.

    - A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
    specifies a sequence (list/tuple/..), it determines the tensor rank and
    shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
    the specified size.

    - A :ref:`custom data structure <custom-struct>`. In this case,
    :py:func:`drjit.ones()` will invoke itself recursively to initialize
    each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
    parameter is ignored in this case.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`custom data structure <custom-struct>`.

        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A instance of type ``dtype`` filled with ones
    '''
    if not isinstance(dtype, type):
        raise Exception('ones(): Type expected as first argument')
    elif issubclass(dtype, ArrayBase):
        return dtype.full_(1.0, shape)
    else:
        return dtype(1.0)


def linspace(dtype, start, stop, num=1, endpoint=True):
    '''
    This function generates an evenly spaced floating point sequence of size
    ``num`` covering the interval [``start``, ``stop``].

    Args:
        dtype (type): Desired Dr.Jit array type. The ``dtype`` must refer to a
          dynamically sized 1D Dr.Jit floating point array, such as
          :py:class:`drjit.scalar.ArrayXf` or :py:class:`drjit.cuda.Float`.

        start (float): Start of the interval.

        stop (float): End of the interval.

        num (int): Number of samples to generate.

        endpoint (bool): Should the interval endpoint be included? The default is `True`.

    Returns:
        object: The computed sequence of type ``dtype``.
    '''
    if not isinstance(dtype, type):
        raise Exception('linspace(): Type expected as first argument')
    elif issubclass(dtype, ArrayBase):
        return dtype.linspace_(start, stop, num, endpoint)
    else:
        return dtype(start)


def arange(dtype, start=None, stop=None, step=1):
    '''
    This function generates an integer sequence on the interval [``start``,
    ``stop``) with step size ``step``, where ``start`` = 0 and ``step`` = 1 if not
    specified.

    Args:
        dtype (type): Desired Dr.Jit array type. The ``dtype`` must refer to a
          dynamically sized 1D Dr.Jit array such as :py:class:`drjit.scalar.ArrayXu`
          or :py:class:`drjit.cuda.Float`.

        start (int): Start of the interval. The default value is `0`.

        stop/size (int): End of the interval (not included). The name of this
          parameter differs between the two provided overloads.

        step (int): Spacing between values. The default value is `1`.

    Returns:
        object: The computed sequence of type ``dtype``.
    '''
    if start is None:
        start = 0
        stop = 1
    elif stop is None:
        stop = start
        start = 0

    if not isinstance(dtype, type):
        raise Exception('arange(): Type expected as first argument')
    elif issubclass(dtype, ArrayBase):
        return dtype.arange_(start, stop, step)
    else:
        return dtype(start)


def identity(dtype, size=1):
    '''
    Return the identity array of the desired type and size

    This function can create identity instances of various types. In
    particular, ``dtype`` can be:

    - A Dr.Jit matrix type (like :py:class:`drjit.cuda.Matrix4f`).

    - A Dr.Jit complex type (like :py:class:`drjit.cuda.Quaternion4f`).

    - Any other Dr.Jit array type. In this case this function is equivalent to ``full(dtype, 1, size)``

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``size``
      parameter is ignored in this case.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`custom data structure <custom-struct>`.

        value (object): An instance of the underlying scalar type
          (``float``/``int``/``bool``, etc.) that will be used to initialize the
          array contents.

        size (int): Size of the desired array | matrix

    Returns:
        object: The identity array of type ``dtype`` of size ``size``
    '''
    if _dr.is_special_v(dtype):
        result = _dr.zeros(dtype, size)

        if dtype.IsComplex or dtype.IsQuaternion:
            result.real = identity(dtype.Value, size)
        elif dtype.IsMatrix:
            one = identity(dtype.Value.Value, size)
            for i in range(dtype.Size):
                result[i, i] = one
        return result
    elif _dr.is_array_v(dtype):
        return full(dtype, 1, size)
    else:
        return dtype(1)


def tile(arg, count: int):
    '''
    This function constructs an Dr.Jit array by repeating ``arg`` ``count`` times.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit type
        count (int): Number of repetitions

    Returns:
        object: The tiled output array.
    '''
    if not _dr.is_array_v(arg) or not isinstance(count, int):
        raise("tile(): invalid input types!")
    elif not arg.IsDynamic:
        raise("tile(): first input argument must be a dynamic Dr.Jit array!")

    size = len(arg)
    t = type(arg)

    if arg.Depth > 1:
        result = t()

        if arg.Size == Dynamic:
            result.init_(size)

        for i in range(size):
            result[i] = _dr.tile(arg[i], count)

        return result
    else:
        index = _dr.arange(_dr.uint_array_t(t), size * count) % size
        return _dr.gather(t, arg, index)


def repeat(array, count: int):
    '''
    This function constructs an Dr.Jit array by repeating the elements of ``arg``
    ``count`` times.

    Args:
        arg (drjit.ArrayBase): A Dr.Jit type
        count (int): Number of repetitions for the elements

    Returns:
        object: Output array where the elements where repeated.
    '''
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
            result[i] = _dr.repeat(array[i], count)

        return result
    else:
        index = _dr.arange(_dr.uint_array_t(t), size * count) // count
        return _dr.gather(t, array, index)


def meshgrid(*args, indexing='xy'):
    '''
    Creates a grid coordinates based on the coordinates contained in the
    provided one-dimensional arrays.

    The indexing keyword argument allows this function to support both matrix
    and Cartesian indexing conventions. If given the string 'ij', it will return
    a grid coordinates with matrix indexing. If given 'xy', it will return a
    grid coordinates with Cartesian indexing.

    .. code-block::

        import drjit as dr

        x, y = dr.meshgrid(
            dr.arange(dr.llvm.UInt, 4),
            dr.arange(dr.llvm.UInt, 4)
        )

        # x = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        # y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    Args:
        args (drjit.ArrayBase): Dr.Jit one-dimensional coordinate arrays

        indexing (str): Specifies the indexing conventions

    Returns:
        tuple: Grid coordinates
    '''
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
           _dr.depth_v(v) != 1 or type(v) is not t:
            raise Exception("meshgrid(): consistent 1D dynamic arrays expected!")

    size = _dr.prod((len(v) for v in args))
    index = _dr.arange(_dr.uint32_array_t(t), size)

    result = []

    # This seems non-symmetric but is necessary to be consistent with NumPy
    if indexing == "xy":
        args = (args[1], args[0], *args[2:])

    for v in args:
        size //= len(v)
        index_v = index // size
        index = _dr.fma(-index_v, size, index)
        result.append(_dr.gather(t, v, index_v))

    if indexing == "xy":
        result[0], result[1] = result[1], result[0]

    return tuple(result)


def opaque(type_, value, shape=1):
    if not isinstance(type_, type):
        raise Exception('opaque(): Type expected as first argument')
    if not _dr.is_jit_v(type_):
        return _dr.full(type_, value, shape)
    if _dr.is_static_array_v(type_):
        result = type_()
        for i in range(len(result)):
            result[i] = opaque(type_.Value, value, shape)
        return result
    if _dr.is_diff_v(type_):
        return _dr.opaque(_dr.detached_t(type_), value, shape)
    if _dr.is_jit_v(type_):
        if _dr.is_tensor_v(type_):
            return type_(_dr.opaque(type_.Array, value, _dr.prod(shape)), shape)
        return type_.opaque_(value, shape)
    elif _dr.is_struct_v(type_):
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
            if _dr.depth_v(t) > 1:
                for i in range(len(a)):
                    make_opaque(a.entry_ref_(i))
            elif _dr.is_diff_v(t):
                make_opaque(a.detach_ref_())
            elif _dr.is_tensor_v(t):
                make_opaque(a.array)
            elif _dr.is_jit_v(t):
                if not a.is_evaluated_():
                    a.assign(a.copy_())
                    a.data_()
        elif _dr.is_struct_v(t):
            for k in t.DRJIT_STRUCT.keys():
                make_opaque(getattr(a, k))
        elif issubclass(t, _Sequence):
            for v in a:
                make_opaque(v)
        elif issubclass(t, _Mapping):
            for k, v in a.items():
                make_opaque(v)


# -------------------------------------------------------------------
#                  Higher-level utility functions
# -------------------------------------------------------------------

def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    '''
    Returns ``True`` if two arrays are element-wise equal within a given error
    tolerance.

    The function considers both absolute and relative error thresholds. Specifically
    **a** and **b** are considered equal if all elements satisfy

    .. math::
        |a - b| \le |b| \cdot \mathrm{rtol} + \mathrm{atol}.

    Args:
        a (object): A Dr.Jit array or other kind of numeric sequence type.

        b (object): A Dr.Jit array or other kind of numeric sequence type.

        rtol (float): A relative error threshold. The default is :math:`10^{-5}`.

        atol (float): An absolute error threshold. The default is :math:`10^{-8}`.

        equal_nan (bool): If **a** and **b** *both* contain a *NaN* (Not a Number) entry,
                          should they be considered equal? The default is ``False``.

    Returns:
        bool: The result of the comparison.
    '''
    if _dr.is_array_v(a) or _dr.is_array_v(b):
        # No derivative tracking in the following
        a, b = _dr.detach(a), _dr.detach(b)

        if _dr.is_array_v(a) and not _dr.is_float_v(a):
            a, _ = _var_promote(a, 1.0)
        if _dr.is_array_v(b) and not _dr.is_float_v(b):
            b, _ = _var_promote(b, 1.0)

        if type(a) is not type(b):
            a, b = _var_promote(a, b)

        diff = abs(a - b)
        shape = 1
        if _dr.is_tensor_v(diff):
            shape = diff.shape
        cond = diff <= abs(b) * rtol + _dr.full(type(diff), atol, shape)
        if _dr.is_float_v(a):
            cond |= _dr.eq(a, b)  # plus/minus infinity

        if equal_nan:
            cond |= _dr.isnan(a) & _dr.isnan(b)

        return _dr.all_nested(cond)

    def safe_len(x):
        try:
            return len(x)
        except TypeError:
            return 0

    def safe_getitem(x, len_x, i):
        if len_x == 0:
            return x
        elif len_x == 1:
            return x[0]
        else:
            return x[i]

    len_a, len_b = safe_len(a), safe_len(b)
    len_ab = _builtins.max(len_a, len_b)

    if len_a != len_ab and len_a > 1 or \
       len_b != len_ab and len_b > 1:
        raise RuntimeError('drjit.allclose(): incompatible sizes '
                           '(%i and %i)!' % (len_a, len_b))
    elif len_ab == 0:
        if equal_nan and _math.isnan(a) and _math.isnan(b):
            return True
        return _dr.abs(a - b) <= _dr.abs(b) * rtol + atol
    else:
        for i in range(len_ab):
            ia = safe_getitem(a, len_a, i)
            ib = safe_getitem(b, len_b, i)
            if not allclose(ia, ib, rtol, atol, equal_nan):
                return False
        return True


def printf_async(fmt, *args, active=True):
    '''
    Print the specified variable contents from the kernel asynchronously.

    This function inserts a print statement directly into the kernel being
    generated. Note that this may produce a very large volume of output,
    and a nonzero ``active`` parameter can be supplied to suppress it based
    on condition.

    Args:
        fmt (str): The string to be printed. It might contain *format specifiers*
          (e.g. subsequences beginning with %)

        *args (tuple): Additional array arguments to be formatted and inserted
          in the printed string replacing their respective specifiers.

        active (bool | drjit.ArrayBase): Mask array to suppress printing specific
          elements in the supplied additional arrays.
    '''
    indices = []
    is_cuda, is_llvm = _dr.is_cuda_v(active), _dr.is_llvm_v(active)

    for a in args:
        cuda, llvm = _dr.is_cuda_v(a), _dr.is_llvm_v(a)
        if not (cuda or llvm) or _dr.depth_v(a) != 1 or _dr.is_mask_v(a):
            raise Exception("printf_async(): array argument of type '%s' not "
                            "supported (must be a depth-1 JIT (LLVM/CUDA) array, "
                            "and cannot be a mask)" % type(a).__name__)
        indices.append(a.index)
        is_cuda |= cuda
        is_llvm |= llvm

    if is_cuda == is_llvm:
        raise Exception("printf_async(): invalid input: must specify LLVM or CUDA arrays.")

    active = _dr.cuda.Bool(active) if is_cuda else _dr.llvm.Bool(active)
    _dr.detail.printf_async(is_cuda, active.index, fmt, indices)


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
    '''
    suspend_grad(*args, when = True)
    Context manager for temporally suspending derivative tracking.

    Dr.Jit's AD layer keeps track of a set of variables for which derivative
    tracking is currently enabled. Using this context manager is it possible to
    define a scope in which variables will be subtracted from that set, thereby
    controlling what derivative terms shouldn't be generated in that scope.

    The variables to be subtracted from the current set of enabled variables can be
    provided as function arguments. If none are provided, the scope defined by this
    context manager will temporally disable all derivative tracking.

    .. code-block::

        a = dr.llvm.ad.Float(1.0)
        b = dr.llvm.ad.Float(2.0)
        dr.enable_grad(a, b)

        with suspend_grad(): # suspend all derivative tracking
            c = a + b

        assert not dr.grad_enabled(c)

        with suspend_grad(a): # only suspend derivative tracking on `a`
            d = 2.0 * a
            e = 4.0 * b

        assert not dr.grad_enabled(d)
        assert dr.grad_enabled(e)

    In a scope where derivative tracking is completely suspended, the AD layer will
    ignore any attempt to enable gradient tracking on a variable:

    .. code-block::

        a = dr.llvm.ad.Float(1.0)

        with suspend_grad():
            dr.enable_grad(a) # <-- ignored
            assert not dr.grad_enabled(a)

        assert not dr.grad_enabled(a)

    The optional ``when`` boolean keyword argument can be defined to specifed a
    condition determining whether to suspend the tracking of derivatives or not.

    .. code-block::

        a = dr.llvm.ad.Float(1.0)
        dr.enable_grad(a)

        cond = condition()

        with suspend_grad(when=cond):
            b = 4.0 * a

        assert dr.grad_enabled(b) == not cond

    Args:
        *args (tuple): A variable-length list of differentiable Dr.Jit array
          instances, :ref:`custom data structures <custom-struct>`, sequences, or
          mappings. The function will recursively traverse data structures to
          discover all Dr.Jit arrays.

        when (bool): An optional Python boolean determining whether to suspend
          derivative tracking.
    '''
    if not when:
        return _DummyContextManager()

    array_indices = []
    array_type = _dr.detail.diff_vars(args, array_indices, check_grad_enabled=False)
    if len(args) > 0 and len(array_indices) == 0:
        array_indices = [0]
    return _ADContextManager(_dr.detail.ADScope.Suspend, array_type, array_indices)


def resume_grad(*args, when=True):
    '''
    resume_grad(*args, when = True)
    Context manager for temporally resume derivative tracking.

    Dr.Jit's AD layer keeps track of a set of variables for which derivative
    tracking is currently enabled. Using this context manager is it possible to
    define a scope in which variables will be added to that set, thereby controlling
    what derivative terms should be generated in that scope.

    The variables to be added to the current set of enabled variables can be
    provided as function arguments. If none are provided, the scope defined by this
    context manager will temporally resume derivative tracking for all variables.

    .. code-block::

        a = dr.llvm.ad.Float(1.0)
        b = dr.llvm.ad.Float(2.0)
        dr.enable_grad(a, b)

        with suspend_grad():
            c = a + b

            with resume_grad():
                d = a + b

            with resume_grad(a):
                e = 2.0 * a
                f = 4.0 * b

        assert not dr.grad_enabled(c)
        assert dr.grad_enabled(d)
        assert dr.grad_enabled(e)
        assert not dr.grad_enabled(f)

    The optional ``when`` boolean keyword argument can be defined to specifed a
    condition determining whether to resume the tracking of derivatives or not.

    .. code-block::

        a = dr.llvm.ad.Float(1.0)
        dr.enable_grad(a)

        cond = condition()

        with suspend_grad():
            with resume_grad(when=cond):
                b = 4.0 * a

        assert dr.grad_enabled(b) == cond

    Args:
        *args (tuple): A variable-length list of differentiable Dr.Jit array
          instances, :ref:`custom data structures <custom-struct>`, sequences, or
          mappings. The function will recursively traverse data structures to
          discover all Dr.Jit arrays.

        when (bool): An optional Python boolean determining whether to resume
          derivative tracking.
    '''
    if not when:
        return _DummyContextManager()

    array_indices = []
    array_type = _dr.detail.diff_vars(args, array_indices, check_grad_enabled=False)
    if len(args) > 0 and len(array_indices) == 0:
        array_indices = [0]
    return _ADContextManager(_dr.detail.ADScope.Resume, array_type, array_indices)


def isolate_grad(when=True):
    '''
    Context manager to temporarily isolate outside world from AD traversals.

    Dr.Jit provides isolation boundaries to postpone AD traversals steps leaving a
    specific scope. For instance this function is used internally to implement
    differentiable loops and polymorphic calls.
    '''
    if not when:
        return _DummyContextManager()
    return _ADContextManager(_dr.detail.ADScope.Isolate, None, [])


# -------------------------------------------------------------------
#             Automatic differentation of custom functions
# -------------------------------------------------------------------

class CustomOp:
    '''
    Base class to implement custom differentiable operations.

    Dr.Jit can compute derivatives of builtin operations in both forward and reverse
    mode. In some cases, it may be useful or even necessary to tell Dr.Jit how a
    particular operation should be differentiated.

    This can be achieved by extending this class, overwriting callback functions
    that will later be invoked when the AD backend traverses the associated node in
    the computation graph. This class also provides a convenient way of stashing
    temporary results during the original function evaluation that can be accessed
    later on as part of forward or reverse-mode differentiation.

    Look at the section on :ref:`AD custom operations <custom-op>` for more detailed
    information.

    A class that inherits from this class should override a few methods as done in
    the code snippet below. :py:func:`dr.custom` can then be used to evaluate the
    custom operation and properly attach it to the AD graph.

    .. code-block::

        class MyCustomOp(dr.CustomOp):
            def eval(self, *args):
                # .. evaluate operation ..

            def forward(self):
                # .. compute forward-mode derivatives ..

            def backward(self):
                # .. compute backward-mode derivatives ..

            def name(self):
                return "MyCustomOp[]"

        dr.custom(MyCustomOp, *args)
    '''
    def __init__(self):
        self._implicit_in = []
        self._implicit_out = []

    def eval(self, *args):
        '''
        eval(self, *args) -> object
        Evaluate the custom function in primal mode.

        The inputs will be detached from the AD graph, and the output *must* also be
        detached.

        .. danger::

            This method must be overriden, no default implementation provided.
        '''
        raise Exception("CustomOp.eval(): not implemented")

    def forward(self):
        '''
        Evaluated forward-mode derivatives.

        .. danger::

            This method must be overriden, no default implementation provided.
        '''
        raise Exception("CustomOp.forward(): not implemented")

    def backward(self):
        '''
        Evaluated backward-mode derivatives.

        .. danger::

            This method must be overriden, no default implementation provided.
        '''
        raise Exception("CustomOp.backward(): not implemented")

    def grad_out(self):
        '''
        Access the gradient associated with the output argument (backward mode AD).

        Returns:
            object: the gradient value associated with the output argument.
        '''
        return _dr.grad(self.output)

    def set_grad_out(self, value):
        '''
        Accumulate a gradient value into the output argument (forward mode AD).

        Args:
            value (object): gradient value to accumulate.
        '''
        _dr.accum_grad(self.output, value)

    def grad_in(self, name):
        '''
        Access the gradient associated with the input argument ``name`` (fwd. mode AD).

        Args:
            name (str): name associated to an input variable (e.g. keyword argument).

        Returns:
            object: the gradient value associated with the input argument.
        '''
        if name not in self.inputs:
            raise Exception("CustomOp.grad_in(): Could not find "
                            "input argument named \"%s\"!" % name)
        return _dr.grad(self.inputs[name])

    def set_grad_in(self, name, value):
        '''
        Accumulate a gradient value into an input argument (backward mode AD).

        Args:
            name (str): name associated to the input variable (e.g. keyword argument).
            value (object): gradient value to accumulate.
        '''
        if name not in self.inputs:
            raise Exception("CustomOp.set_grad_in(): Could not find "
                            "input argument named \"%s\"!" % name)
        _dr.accum_grad(self.inputs[name], value)

    def add_input(self, value):
        '''
        Register an implicit input dependency of the operation on an AD variable.

        This function should be called by the ``eval()`` implementation when an
        operation has a differentiable dependence on an input that is not an
        input argument (e.g. a private instance variable).

        Args:
            value (object): variable this operation depends on implicitly.
        '''
        self._implicit_in.append(value)

    def add_output(self, value):
        '''
        Register an implicit output dependency of the operation on an AD variable.

        This function should be called by the \ref eval() implementation when an
        operation has a differentiable dependence on an output that is not an
        return value of the operation (e.g. a private instance variable).

        Args:
            value (object): variable this operation depends on implicitly.
        '''
        self._implicit_out.append(value)

    def __del__(self):
        def ad_clear(o):
            if _dr.depth_v(o) > 1 \
               or isinstance(o, _Sequence):
                for i in range(len(o)):
                    ad_clear(o[i])
            elif isinstance(o, _Mapping):
                for k, v in o.items():
                    ad_clear(v)
            elif _dr.is_diff_v(o):
                if _dr.is_tensor_v(o):
                    ad_clear(o.array)
                else:
                    o.set_index_ad_(0)
            elif _dr.is_struct_v(o):
                for k in type(o).DRJIT_STRUCT.keys():
                    ad_clear(getattr(o, k))
        ad_clear(getattr(self, 'output', None))

    def name(self):
        '''
        Return a descriptive name of the ``CustomOp`` instance.

        The name returned by this method is used in the GraphViz output.

        If not overriden, this method returns ``"CustomOp[unnamed]"``.
        '''
        return "CustomOp[unnamed]"


def custom(cls, *args, **kwargs):
    '''
    Evaluate a custom differentiable operation (see :py:class:`CustomOp`).

    Look at the section on :ref:`AD custom operations <custom-op>` for more detailed
    information.
    '''
    # Clear primal values of a differentiable array
    def clear_primal(o, dec_ref):
        if _dr.depth_v(o) > 1 \
           or isinstance(o, _Sequence):
            return type(o)([clear_primal(o[i], dec_ref) for i in range(len(o))])
        elif isinstance(o, _Mapping):
            return { k: clear_primal(v, dec_ref) for k, v in o.items() }
        elif _dr.is_diff_v(o) and _dr.is_float_v(o):
            ot = type(o)

            if _dr.is_tensor_v(ot):
                value = ot.Array.create_(
                    o.array.index_ad,
                    _dr.zeros(_dr.detached_t(ot.Array), prod(o.shape)))
                result = ot(value, o.shape)
            else:
                result = value = ot.create_(
                    o.index_ad,
                    _dr.detached_t(ot)())
            if dec_ref:
                value.dec_ref_()
            return result
        elif _dr.is_struct_v(o):
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
        raise RuntimeError("drjit.custom(): the return value of CustomOp.eval() "
                           "should not be attached to the AD graph!")

    diff_vars_in = []
    _dr.detail.diff_vars(kwargs, diff_vars_in)
    _dr.detail.diff_vars(inst._implicit_in, diff_vars_in)

    if len(diff_vars_in) > 0:
        output = _dr.diff_array_t(output, allow_non_array=True)
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
                Type.add_edge_(index, tmp_in.index_ad)

        if len(diff_vars_out) > 1:
            _dr.set_label(tmp_out, inst.name() + "_out")
            for index in diff_vars_out:
                Type.add_edge_(tmp_out.index_ad, index)

        Type.add_edge_(
            diff_vars_in[0]  if len(diff_vars_in)  == 1 else tmp_in.index_ad,
            diff_vars_out[0] if len(diff_vars_out) == 1 else tmp_out.index_ad,
            inst
        )

        inst._implicit_in = []
        inst._implicit_out = []

    return output


def graphviz_ad(as_str=False):
    '''
    Assembles a graphviz diagram for the computational graph trace by the AD system.

    Args:
        as_str (bool): whether the function should return the graphviz object as
            a string representation or not.

    Returns:
        object: the graphviz obj (or its string representation).
    '''
    s = _dr.detail.graphviz_ad()

    if as_str:
        return s

    try:
        from graphviz import Source
        return Source(s)
    except ImportError:
        raise Exception('The "graphviz" Python package not available! Install '
                        'via "python -m pip install graphviz". Alternatively, '
                        'you can call drjit.graphviz_ad(as_str=True) to obtain '
                        'a string representation.')

def get_cmake_dir():
    from os import path
    file_dir = path.abspath(path.dirname(__file__))
    cmake_path = path.join(file_dir, "share", "cmake", "drjit")
    if not path.exists(cmake_path):
        raise ImportError("Cannot find Dr.Jit CMake directory")
    return cmake_path
