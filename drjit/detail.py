import drjit as _dr
import sys
import inspect
from collections.abc import Mapping, Sequence

VAR_TYPE_NAME = [
    "Void",    "Bool",  "Int8",  "UInt8", "Int16",
    "UInt16",  "Int",   "UInt",  "Int64", "UInt64", "Pointer",
    "Float16", "Float", "Float64"
]

VAR_TYPE_SUFFIX = [
    "???", "b", "i8",  "u8",  "i16", "u16", "i", "u",
    "i64", "u64", "p", "f16", "f", "f64"
]


def array_name(prefix, vt, shape, scalar):
    """
    Determines the name of an array (e.g. Float32, ArrayXf32, etc.). This
    function is used when arrays are created during initialization of the Dr.Jit
    extension module, and during implicit type promotion where array types are
    determined dynamically.

    Parameter ``prefix`` (``str``):
        Array flavor prefix (Array/Matrix/Complex/Quaternion)

    Parameter ``vt`` (``drjit.VarType``):
        Underlying scalar type (e.g. ``VarType.Int32``) of the desired array

    Parameter ``shape`` (``Tuple[int]``):
        Size per dimension

    Parameter ``scalar`` (``bool``):
        Arrays in the ``drjit.scalar.*`` module use a different depth
        convention, which is indicated via this parameter.
    """

    if not scalar and not prefix == 'Tensor':
        shape = shape[:-1]
    if prefix == "Matrix":
        if vt != _dr.VarType.Bool:
            shape = shape[1:]
        else:
            prefix = "Array"

    if len(shape) == 0 and prefix != "Tensor":
        return VAR_TYPE_NAME[int(vt)]

    return "%s%s%s" % (
        prefix,
        "".join(repr(s) if s != _dr.Dynamic else "X" for s in shape),
        VAR_TYPE_SUFFIX[int(vt)]
    )


def array_from_dlpack(t, capsule):
    descr = _dr.detail.from_dlpack(capsule)

    device_type = descr["device_type"]
    data = descr["data"]
    dtype = descr["dtype"]
    shape = descr["shape"]
    ndim = len(shape)
    strides = descr["strides"]

    if strides is None:
        tmp = 1
        strides = [0] * ndim
        for i in reversed(range(ndim)):
            strides[i] = tmp
            tmp *= shape[i]

    if t.IsCUDA and device_type != 2:
        raise _dr.Exception("Cannot create an Dr.Jit GPU array from a "
                              "DLPack CPU tensor!")
    elif not t.IsCUDA and device_type != 1:
        raise _dr.Exception("Cannot create an Dr.Jit CPU array from a "
                              "DLPack GPU tensor!")

    if dtype != t.Type:
        raise _dr.Exception("Incompatible type!")

    shape_target = list(reversed(_dr.shape(t())))
    if len(shape_target) != ndim:
        raise _dr.Exception("Incompatible dimension!")
    for i in range(ndim):
        if shape_target[i] != shape[i] and shape_target[i] != 0:
            raise _dr.Exception("Incompatible shape!")

    value = t
    while issubclass(value.Value, _dr.ArrayBase):
        value = value.Value

    descr["consume"](capsule)
    data = value.map_(data, _dr.prod(shape), descr["release"])

    def load(t, i, offset):
        size = shape[-1 - i]
        stride = strides[-1 - i]

        if i == ndim - 1:
            if type(offset) is int and stride == 1:
                return data
            else:
                i = _dr.arange(_dr.int32_array_t(t), size)
                return t.gather_(data, offset + stride * i, True, False)
        else:
            result = t()
            for j in range(size):
                result[j] = load(t.Value, i + 1, offset + stride * j)
            return result

    return load(t, 0, 0)


def sub_len(o):
    ol = len(o[0])
    try:
        for i in range(1, len(o)):
            if len(o[i]) != ol:
                return -1
    except Exception:
        return -1
    return ol


def array_init(self, args):
    """
    This generic initialization routine initializes an arbitrary Dr.Jit array
    from a variable-length argument list (which could be a scalar broadcast, a
    component list, or a NumPy/PyTorch/Tensorflow array..)
    """
    n = len(args)
    if n == 0:
        return

    size = self.Size
    value_type = self.Value
    dynamic = size == _dr.Dynamic
    err = None

    try:
        if n == 1:
            o = args[0]
            t = type(o)
            mod = t.__module__
            name = t.__name__
            is_array = issubclass(t, _dr.ArrayBase)
            is_static_array = is_array and not o.Size == _dr.Dynamic
            is_sequence = issubclass(t, Sequence)

            # Matrix initialization from nested list
            if is_sequence and self.IsMatrix and \
                len(o) == size and sub_len(o) == size:
                for x in range(size):
                    for y in range(size):
                        self[x, y] = value_type.Value(o[x][y])
            elif is_array or is_sequence:
                os = len(o)
                if dynamic:
                    size = os
                    self.init_(size)

                if size == 0:
                    pass
                elif size != os or value_type is t:
                    # Size mismatch!
                    if self.IsMatrix and getattr(t, "IsMatrix", False):
                        # If both are matrices, copy the top-left block
                        for x in range(size):
                            for y in range(size):
                                if x < o.Size and y < o.Size:
                                    self[x, y] = value_type.Value(o[x, y])
                                else:
                                    self[x, y] = value_type.Value(1 if x == y else 0)
                    elif self.IsMatrix and value_type is t:
                        for x in range(size):
                            self[x] = o
                    else:
                        # Otherwise, try to broadcast to all entries
                        self.broadcast_(value_type(o)
                                        if not issubclass(t, value_type)
                                        and not self.IsMatrix else o)
                else:
                    # Size matches, copy element by element
                    if self.IsJIT and getattr(t, "IsJIT", False) and \
                       self.Depth == 1 and t.Depth == 1:
                        raise _dr.Exception(
                            "Refusing to do an extremely inefficient "
                            "element-by-element array conversion from type %s "
                            "to %s. Did you forget a cast or detach operation?"
                            % (str(type(o)), str(type(self))))

                    if isinstance(o[0], value_type) or self.IsMatrix:
                        for i in range(size):
                            self.set_entry_(i, o[i])
                    else:
                        for i in range(size):
                            self.set_entry_(i, value_type(o[i]))
            elif issubclass(t, (int, float)) and (not self.IsJIT or self.Depth > 1):
                if dynamic:
                    size = 1
                    self.init_(size)
                self.broadcast_(o)
            elif issubclass(t, complex) and self.IsComplex:
                self.set_entry_(0, o.real)
                self.set_entry_(1, o.imag)
            elif mod == "numpy":
                import numpy as np

                s1 = tuple(reversed(_dr.shape(self)))
                s2 = o.shape

                # Remove unnecessary outer dimension is possible
                if s2[0] == 1 and len(s2) > 1:
                    o = o[0, ...]
                    s2 = o.shape

                if o.dtype == np.complex64:
                    s2 = (*s2, 2)
                    o = o.view(np.float32).reshape(s2)
                elif o.dtype == np.complex128:
                    s2 = (*s2, 2)
                    o = o.view(np.float64).reshape(s2)

                if o.dtype != self.Type.NumPy:
                    o = o.astype(self.Type.NumPy)

                dim1 = len(s1)
                dim2 = len(s2)

                # Numpy array might have one dimension less when initializing dynamic arrays
                if not dim1 == dim2 and not (dim1 == dim2 + 1 and self.IsDynamic):
                    raise _dr.Exception("Incompatible dimension!")
                for i in reversed(range(dim2)):
                    if s1[i] != s2[i] and s1[i] != 0:
                        raise _dr.Exception("Incompatible shape!")

                if dim1 == 0:
                    pass
                elif dim1 == 1 and self.IsDynamic:
                    o = np.ascontiguousarray(o)
                    holder = (o, o.__array_interface__["data"][0])
                    self.assign(self.load_(holder[1], s2[0]))
                else:
                    for i in range(s1[-1]):
                        if dim2 == 1 and self.IsDynamic:
                            self.set_entry_(i, value_type.Value(o[i]))
                        else:
                            self.set_entry_(i, value_type(o[..., i]))

            elif mod == "builtins" and name == "PyCapsule":
                self.assign(array_from_dlpack(type(self), o))
            elif mod == "torch":
                from torch.utils.dlpack import to_dlpack
                self.assign(array_from_dlpack(type(self), to_dlpack(o)))
            elif mod.startswith("tensorflow."):
                from tensorflow.experimental.dlpack import to_dlpack
                self.assign(array_from_dlpack(type(self), to_dlpack(o)))
            elif mod.startswith("jax.") or mod.startswith("jaxlib."):
                from jax.dlpack import to_dlpack
                self.assign(array_from_dlpack(type(self), to_dlpack(o)))
            else:
                raise _dr.Exception("Don\"t know how to create an Dr.Jit array "
                                      "from type \"%s.%s\"!" % (mod, name))
        elif n == size or dynamic:
            if dynamic:
                size = n
                self.init_(size)
            for i in range(size):
                self.set_entry_(i, value_type(args[i]))
        elif self.IsMatrix and n == self.Size * self.Size:
            tbl = [[args[i*self.Size + j] for i in range(self.Size)]
                   for j in range(self.Size)]
            array_init(self, tbl)
        else:
            raise _dr.Exception("Invalid size!")
    except Exception as e:
        err = e

    if err is not None:
        if dynamic:
            raise TypeError("%s constructor expects: arbitrarily many values "
                            "of type \"%s\", a matching list/tuple, or a NumPy/"
                            "PyTorch/TF/Jax array." % (type(self).__name__,
                                                       value_type.__name__)) from err
        else:
            raise TypeError("%s constructor expects: %s%i values "
                            "of type \"%s\", a matching list/tuple, or a NumPy/"
                            "PyTorch/TF/Jax array." % (type(self).__name__, "" if
                                                       size == 1 else "1 or ", size,
                                                       value_type.__name__)) from err


def tensor_init(tensor_type, obj):
    mod = type(obj).__module__
    if 'tensorflow' in mod:
        import tensorflow as tf
        return tensor_type(tensor_type.Array(tf.reshape(obj, [-1])), obj.shape)
    elif mod.startswith('torch'):
        return tensor_type(tensor_type.Array(obj.flatten()), obj.shape)
    elif mod.startswith(('numpy', 'jax')):
        return tensor_type(tensor_type.Array(obj.ravel()), obj.shape)
    else:
        info = getattr(obj, '__array_interface__', None)
        if info is not None:
            shape = info['shape']
            typestr = str(info['typestr'])[3:-1]
            cls = tensor_type.Array
            if typestr != tensor_type.Array.Type.NumPy:
                name = None
                for v in VAR_TYPE_NAME:
                    t = getattr(_dr.VarType, v, None)
                    if t and t.NumPy == typestr:
                        name = array_name('Array', t, [_dr.Dynamic], False)
                        break
                if name and hasattr(mod, name):
                    cls = getattr(mod, name)
                else:
                    import numpy as np
                    np_data = np.array(obj).astype(tensor_type.Array.Type.NumPy)
                    return tensor_init(tensor_type, np_data)

            data = cls.load_(info['data'][0], _dr.prod(shape))
            return tensor_type(tensor_type.Array(data), shape)
        else:
            raise TypeError("TensorXf: expect an array that implements the "
                            "array interface protocol!")

@property
def prop_x(self):
    return self[0]


@prop_x.setter
def prop_x(self, value):
    self[0] = value


@property
def prop_y(self):
    return self[1]


@prop_y.setter
def prop_y(self, value):
    self[1] = value


@property
def prop_z(self):
    return self[2]


@prop_z.setter
def prop_z(self, value):
    self[2] = value


@property
def prop_w(self):
    return self[3]


@prop_w.setter
def prop_w(self, value):
    self[3] = value


@property
def prop_xyz(self):
    return self.Imag(self[0], self[1], self[2])


@prop_xyz.setter
def prop_xyz(self, value):
    if not isinstance(value, self.Imag):
        value = self.Imag(value)
    self.x = value.x
    self.y = value.y
    self.z = value.z


def array_configure(cls, shape, type_, value):
    """Populates an Dr.Jit array class with extra type trait fields"""
    depth = 1

    cls.Value = value
    cls.Type = type_
    cls.Shape = shape
    cls.Size = shape[0]
    cls.IsDynamic = cls.Size == _dr.Dynamic or \
        getattr(value, "IsDynamic", False)

    while issubclass(value, _dr.ArrayBase):
        value = value.Value
        depth += 1

    cls.Depth = depth
    cls.Scalar = value
    cls.IsDrJit = True
    cls.IsMask = issubclass(value, bool)
    cls.IsIntegral = issubclass(value, int) and not cls.IsMask
    cls.IsFloat = issubclass(value, float)
    cls.IsArithmetic = cls.IsIntegral or cls.IsFloat
    cls.IsSigned = cls.IsFloat or "i" in VAR_TYPE_SUFFIX[int(type_)]

    mod = cls.__module__
    cls.IsScalar = mod.startswith("drjit.scalar")
    cls.IsPacket = mod.startswith("drjit.packet")
    cls.IsDiff = mod.endswith(".ad")
    cls.IsLLVM = mod.startswith("drjit.llvm")
    cls.IsCUDA = mod.startswith("drjit.cuda")
    cls.IsJIT = cls.IsLLVM or cls.IsCUDA

    name = cls.__name__
    cls.IsMatrix = "Matrix" in name
    cls.IsComplex = "Complex" in name
    cls.IsQuaternion = "Quaternion" in name
    cls.IsTensor = "Tensor" in name
    cls.IsSpecial = cls.IsMatrix or cls.IsComplex or cls.IsQuaternion
    cls.IsVector = cls.Size != _dr.Dynamic and not \
        (cls.IsPacket and cls.Depth == 1) and not cls.IsSpecial

    prefix = name
    for i, c in enumerate(name):
        if c.isdigit() or c == 'X':
            prefix = name[:i]
            break
    mask_name = prefix

    cls.Prefix = prefix

    if cls.IsSpecial:
        mask_name = 'Array'

        if cls.IsComplex:
            cls.real = prop_x
            cls.imag = prop_y
        elif cls.IsQuaternion:
            cls.real = prop_w
            cls.imag = prop_xyz
            cls.Imag = getattr(sys.modules.get(mod),
                               name.replace("Quaternion4", "Array3"))
            cls.Complex = getattr(sys.modules.get(mod),
                                  name.replace("Quaternion4", "Complex2"))

    if cls.IsTensor:
        cls.__getitem__ = _dr.detail.tensor_getitem
        cls.__setitem__ = _dr.detail.tensor_setitem

    elif (not cls.IsSpecial or cls.IsQuaternion) \
            and not cls.Size == _dr.Dynamic:
        if cls.Size > 0:
            cls.x = prop_x
        if cls.Size > 1:
            cls.y = prop_y
        if cls.Size > 2:
            cls.z = prop_z
        if cls.Size > 3:
            cls.w = prop_w

    cls.MaskType = getattr(
        sys.modules.get(mod),
        array_name(mask_name, _dr.VarType.Bool, cls.Shape, cls.IsScalar))


def _loop_process_state(value: type, in_state: list, out_state: list,
                        write: bool, in_struct: bool = False):
    '''
    This helper function is used by ``drjit.*.Loop`` to collect the set of loop
    state variables and ensure that their types stay consistent over time. It
    traverses a python object tree in ``value`` and writes state variable
    indices to ``out_state``. If provided, it performs a consistency check
    against the output of a prior call provided via ``in_state``. If ``write``
    is set to ``True``, it mutates the input value based on the information in
    ``in_state``.
    '''
    t = type(value)

    out_state.append(t)
    if in_state:
        assert len(in_state) > 0
        t_old = in_state.pop()
        if t is not t_old:
            raise _dr.Exception(
                "loop_process_state(): the type of loop state variables must "
                "remain the same throughout the loop. However, one of the "
                "supplied variables changed from type %s to %s!"
                % (t_old.__name__, t.__name__))

    if issubclass(t, tuple) or issubclass(t, list):
        for entry in value:
            _loop_process_state(entry, in_state, out_state, write, in_struct)
        return

    if _dr.is_tensor_v(t):
        _loop_process_state(value.array, in_state, out_state, in_struct)
    elif _dr.is_jit_v(t):
        if t.Depth > 1:
            for i in range(len(value)):
                _loop_process_state(value.entry_ref_(i), in_state,
                                    out_state, write, in_struct)
        else:
            index = value.index
            index_ad = value.index_ad if t.IsDiff else 0

            if index_ad != 0 and _dr.flag(_dr.JitFlag.LoopRecord):
                raise _dr.Exception(
                    "loop_process_state(): one of the supplied loop state variables "
                    "of type %s is attached to the AD graph (i.e., grad_enabled(..) "
                    "is true). However, propagating derivatives through multiple "
                    "iterations of a recorded loop is not supported (and never "
                    "will be). Please see the documentation on differentiating loops "
                    "for details and suggested alternatives." % t.__name__)

            if index == 0:
                raise _dr.Exception(
                    "loop_process_state(): one of the supplied loop state "
                    "variables of type %s is uninitialized!" % t.__name__)

            ad_float_precision = value.Type.Size * 8 if (t.IsDiff and t.IsFloat) else 0
            out_state.append((index, index_ad, ad_float_precision))

            if in_state:
                assert len(in_state) > 0
                index, index_ad, _ = in_state.pop()

                if write:
                    value.set_index_(index)
                    if t.IsDiff:
                        value.set_index_ad_(index_ad)
    elif _dr.is_struct_v(t):
        for k, v in t.DRJIT_STRUCT.items():
            _loop_process_state(getattr(value, k), in_state, out_state, True)
    elif hasattr(value, 'loop_put') or value is None:
        pass
    elif not in_struct:
        raise _dr.Exception(
            "loop_process_state(): one of the provided loop state variables "
            "was of type '%s', which is not allowed (you must use Dr.Jit "
            "arrays/structs that are managed by the JIT compiler)"
            % t.__name__)


def loop_process_state(loop, funcs, state, write):
    if len(state) == 0:
        old_state = None

        for func in funcs:
            values = func()

            if isinstance(values, Sequence):
                for value in values:
                    if hasattr(value, 'loop_put'):
                        value.loop_put(loop)

            # Automatically label loop variables
            cv = inspect.getclosurevars(func)
            _dr.set_label(**cv.globals)
            _dr.set_label(**cv.nonlocals)

            del values

        assert old_state is None or len(old_state) == 0
    else:
        old_state = list(state)
        old_state.reverse()

    del loop # Keep 'loop' out of the garbage collector's reach

    state.clear()

    for func in funcs:
        _loop_process_state(func(), old_state, state, write)

    assert old_state is None or len(old_state) == 0


def slice_tensor(shape, indices, uint32):
    """
    This function takes an array shape (integer tuple) and a tuple containing
    slice indices. It returns the resulting array shape and a flattened 32-bit
    unsigned integer array containing element indices.
    """
    components = []
    ellipsis = False
    none_count = 0
    shape_offset = 0

    for v in indices:
        if v is None:
            none_count += 1

    for v in indices:
        if v is None:
            components.append(None)
            continue

        if shape_offset >= len(shape):
            raise IndexError("slice_tensor(): too many indices specified!")

        size = shape[shape_offset]

        if isinstance(v, int):
            # Simple integer index, handle wrap-around
            if v < 0:
                v += size
            if v >= size:
                raise IndexError("slice_tensor(): index %i for dimension %i is "
                                 "out of range (size = %i)!" %
                                 (v, len(components), size))
            components.append((v, v+1, 1))
        elif isinstance(v, slice):
            # Rely on Python's slice.indices() function to determine everything
            components.append(v.indices(size))
        elif _dr.is_dynamic_array_v(v) and _dr.is_integral_v(v):
            if _dr.is_signed_v(v):
                v = uint32(_dr.select(v >= 0, v, v + size))
            components.append(v)
        elif isinstance(v, Sequence):
            components.append(uint32([v2 if v2 >= 0 else v2 + size for v2 in v]))
        elif v is Ellipsis:
            if ellipsis:
                raise IndexError("slice_tensor(): multiple ellipses (...) are not allowed!")
            ellipsis = True

            for j in range(len(shape) - len(indices) + none_count + 1):
                components.append((0, shape[shape_offset], 1))
                shape_offset += 1
            continue
        else:
            raise TypeError("slice_tensor(): type '%s' cannot be used to index into a tensor!",
                            type(v).__name__)
        shape_offset += 1

    # Implicit ellipsis
    for j in range(len(shape) - shape_offset):
        components.append((0, shape[shape_offset], 1))
        shape_offset += 1

    # Compute total index size
    size_out = 1
    shape_out = []
    for comp in components:
        if comp is None:
            shape_out.append(1)
        else:
            size = len(comp if isinstance(comp, uint32) else range(*comp))
            if size != 1:
                shape_out.append(size)
                size_out *= shape_out[-1]
    shape_out = tuple(shape_out)

    index_tmp = _dr.arange(uint32, size_out)
    index_out = uint32()

    if size_out > 0:
        size_out = 1
        index_out = uint32(0)
        shape_offset = len(shape)-1

        for i in reversed(range(len(components))):
            comp = components[i]
            if comp is None:
                continue
            size = len(comp if isinstance(comp, uint32) else range(*comp))
            index_next = index_tmp // size
            index_rem = index_tmp - index_next * size

            if isinstance(comp, uint32):
                index_val = _dr.gather(uint32, comp, index_rem)
            else:
                if comp[0] >= 0 and comp[2] >= 0:
                    index_val = comp[0] + comp[2] * index_rem
                else:
                    index_val = uint32(comp[0] + comp[2] * _dr.int32_array_t(index_rem)(index_rem))

            index_out += index_val * size_out
            index_tmp = index_next
            size_out *= shape[shape_offset]
            shape_offset -= 1

    return shape_out, index_out


def tensor_getitem(tensor, slice_arg):
    if not isinstance(slice_arg, tuple):
        slice_arg = (slice_arg,)
    tensor_t = type(tensor)
    shape, index = slice_tensor(tensor.shape, slice_arg, tensor_t.Index)
    return tensor_t(_dr.gather(tensor_t.Array, tensor.array, index), shape)


def tensor_setitem(tensor, slice_arg, value):
    if not isinstance(slice_arg, tuple):
        slice_arg = (slice_arg,)
    tensor_t = type(tensor)
    shape, index = slice_tensor(tensor.shape, slice_arg, tensor_t.Index)
    _dr.scatter(target=tensor.array, value=value, index=index)


def diff_vars(o, indices, check_grad_enabled=True):
    """
    Extract indices of differentiable variables, returns
    the type of the underlying differentiable array
    """

    result = None
    if _dr.depth_v(o) > 1 or isinstance(o, Sequence):
        for i in range(len(o)):
            t = diff_vars(o[i], indices, check_grad_enabled)
            if t is not None:
                result = t
    elif isinstance(o, Mapping):
        for k, v in o.items():
            t = diff_vars(v, indices, check_grad_enabled)
            if t is not None:
                result = t
    elif _dr.is_diff_v(o) and o.IsFloat:
        if _dr.is_tensor_v(o):
            result = diff_vars(o.array, indices, check_grad_enabled)
        elif o.index_ad != 0 and (not check_grad_enabled or o.grad_enabled_()):
            indices.append(o.index_ad)
            result = type(o)
    elif _dr.is_struct_v(o):
        for k in type(o).DRJIT_STRUCT.keys():
            t = diff_vars(getattr(o, k), indices, check_grad_enabled)
            if t is not None:
                result = t
    return result
