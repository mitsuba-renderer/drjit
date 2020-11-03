import enoki
import sys

VAR_TYPE_NAME = [
    'Global', 'Invalid', 'Bool',  'Int8',   'UInt8',   'Int16',
    'UInt16', 'Int',     'UInt',  'Int64',  'UInt64',  'Float16',
    'Float',  'Float64', 'Pointer'
]

VAR_TYPE_SUFFIX = [
    '???', '???', 'b', 'i8',  'u8',  'i16', 'u16', 'i', 'u',
    'i64', 'u64', 'f16', 'f', 'f64',  'p'
]


def array_name(prefix, vt, shape, scalar):
    """
    Determines the name of an array (e.g. Float32, ArrayXf32, etc.). This
    function is used when arrays are created during initialization of the Enoki
    extension module, and during implicit type promotion where array types are
    determined dynamically.

    Parameter ``prefix`` (``str``):
        Array flavor prefix (Array/Matrix/Complex/Quaternion)

    Parameter ``vt`` (``enoki.VarType``):
        Underlying scalar type (e.g. ``VarType.Int32``) of the desired array

    Parameter ``size`` (``Tuple[int]``):
        Number of components

    Parameter ``scalar`` (``bool``):
        Arrays in the ``enoki.scalar.*`` module use a different depth
        convention, which is indicated via this parameter.
    """

    if not scalar:
        shape = shape[:-1]
    if prefix == 'Matrix':
        if vt != enoki.VarType.Bool:
            shape = shape[1:]
        else:
            prefix = 'Array'

    if len(shape) == 0:
        return VAR_TYPE_NAME[int(vt)]

    return "%s%s%s" % (
        prefix,
        ''.join(repr(s) if s != enoki.Dynamic else 'X' for s in shape),
        VAR_TYPE_SUFFIX[int(vt)]
    )


def array_from_dlpack(t, capsule):
    descr = enoki.detail.from_dlpack(capsule)

    device_type = descr['device_type']
    data = descr['data']
    dtype = descr['dtype']
    shape = descr['shape']
    ndim = len(shape)
    strides = descr['strides']

    if strides is None:
        tmp = 1
        strides = [0] * ndim
        for i in reversed(range(ndim)):
            strides[i] = tmp
            tmp *= shape[i]

    if t.IsCUDA and device_type != 2:
        raise Exception("Cannot create an Enoki GPU array from a "
                        "DLPack CPU tensor!")
    elif not t.IsCUDA and device_type != 1:
        raise Exception("Cannot create an Enoki CPU array from a "
                        "DLPack GPU tensor!")

    if dtype != t.Type:
        raise Exception("Incompatible type!")

    shape_target = list(reversed(enoki.shape(t())))
    if len(shape_target) != ndim:
        raise Exception("Incompatible dimension!")
    for i in range(ndim):
        if shape_target[i] != shape[i] and shape_target[i] != 0:
            raise Exception("Incompatible shape!")

    value = t
    while issubclass(value.Value, enoki.ArrayBase):
        value = value.Value

    descr['consume'](capsule)
    data = value.map_(data, enoki.hprod(shape), descr['release'])

    def load(t, i, offset):
        size = shape[-1 - i]
        stride = strides[-1 - i]

        if i == ndim - 1:
            if type(offset) is int and stride == 1:
                return data
            else:
                i = enoki.arange(enoki.int32_array_t(t), size)
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
    This generic initialization routine initializes an arbitrary Enoki array
    from a variable-length argument list (which could be a scalar broadcast, a
    component list, or a NumPy/PyTorch/Tensorflow array..)
    """
    n = len(args)
    if n == 0:
        return

    size = self.Size
    value_type = self.Value
    dynamic = size == enoki.Dynamic
    err = None

    try:
        if n == 1:
            o = args[0]
            t = type(o)
            mod = t.__module__
            name = t.__name__
            is_array = issubclass(t, enoki.ArrayBase)
            is_static_array = is_array and not o.Size == enoki.Dynamic
            is_sequence = issubclass(t, list) or issubclass(t, tuple)

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
                elif size != os or (is_static_array and size != o.Size):
                    if self.IsMatrix and o.IsMatrix:
                        for x in range(size):
                            for y in range(size):
                                if x < o.Size and y < o.Size:
                                    self[x, y] = value_type.Value(o[x, y])
                                else:
                                    self[x, y] = value_type.Value(1 if x == y else 0)
                    else:
                        self.broadcast_(value_type(o)
                                        if not isinstance(o, value_type)
                                        and not self.IsMatrix else o)
                else:
                    if self.IsJIT and getattr(t, 'IsJIT', 0) and \
                       self.Depth == 1 and t.Depth == 1:
                        raise Exception(
                            'Refusing to do an extremely inefficient '
                            'element-by-element array conversion from type %s '
                            'to %s. Did you forget a cast or detach operation?'
                            % (str(type(o)), str(type(self))))

                    if isinstance(o[0], value_type) or self.IsMatrix:
                        for i in range(size):
                            self.set_entry_(i, o[i])
                    else:
                        for i in range(size):
                            self.set_entry_(i, value_type(o[i]))
            elif issubclass(t, (int, float)):
                if dynamic:
                    size = 1
                    self.init_(size)
                self.broadcast_(o)
            elif issubclass(t, complex) and self.IsComplex:
                self.set_entry_(0, o.real)
                self.set_entry_(1, o.imag)
            elif mod == 'numpy':
                import numpy as np
                s1 = tuple(reversed(enoki.shape(self)))
                s2 = o.shape

                # Remove unnecessary outer dimension is possible
                if s2[0] == 1:
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
                    raise Exception("Incompatible dimension!")
                for i in reversed(range(dim2)):
                    if s1[i] != s2[i] and s1[i] != 0:
                        raise Exception("Incompatible shape!")

                if dim1 == 0:
                    pass
                elif dim1 == 1 and self.IsDynamic:
                    o = np.ascontiguousarray(o)
                    holder = (o, o.__array_interface__['data'][0])
                    self.assign(self.load_(holder[1], s2[0]))
                else:
                    for i in range(s1[-1]):
                        if dim2 == 1 and self.IsDynamic:
                            self.set_entry_(i, value_type.Value(o[i]))
                        else:
                            self.set_entry_(i, value_type(o[..., i]))

            elif mod == 'builtins' and name == 'PyCapsule':
                self.assign(array_from_dlpack(type(self), o))
            elif mod == 'torch':
                from torch.utils.dlpack import to_dlpack
                self.assign(array_from_dlpack(type(self), to_dlpack(o)))
            elif mod.startswith('tensorflow.'):
                from tensorflow.experimental.dlpack import to_dlpack
                self.assign(array_from_dlpack(type(self), to_dlpack(o)))
            elif mod.startswith('jax.'):
                from jax.dlpack import to_dlpack
                self.assign(array_from_dlpack(type(self), to_dlpack(o)))
            else:
                raise Exception('Don\'t know how to create an Enoki array '
                                'from type \"%s.%s\"!' % (mod, name))
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
            raise Exception('Invalid size!')
    except Exception as e:
        err = e

    if err is not None:
        if dynamic:
            raise TypeError("%s constructor expects: arbitrarily many values "
                            "of type '%s', a matching list/tuple, or a NumPy/"
                            "PyTorch array." % (type(self).__name__,
                                                value_type.__name__)) from err
        else:
            raise TypeError("%s constructor expects: %s%i values "
                            "of type '%s', a matching list/tuple, or a NumPy/"
                            "PyTorch array." % (type(self).__name__, "" if
                                                size == 1 else "1 or ", size,
                                                value_type.__name__)) from err


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
    """Populates an Enoki array class with extra type trait fields"""
    depth = 1

    cls.Value = value
    cls.Type = type_
    cls.Shape = shape
    cls.Size = shape[0]
    cls.IsDynamic = cls.Size == enoki.Dynamic or \
        getattr(value, 'IsDynamic', False)

    while issubclass(value, enoki.ArrayBase):
        value = value.Value
        depth += 1

    cls.Depth = depth
    cls.Scalar = value
    cls.IsEnoki = True
    cls.IsMask = issubclass(value, bool)
    cls.IsIntegral = issubclass(value, int) and not cls.IsMask
    cls.IsFloat = issubclass(value, float)
    cls.IsArithmetic = cls.IsIntegral or cls.IsFloat
    cls.IsSigned = cls.IsFloat or 'i' in VAR_TYPE_SUFFIX[int(type_)]

    mod = cls.__module__
    cls.IsScalar = mod.startswith('enoki.scalar')
    cls.IsPacket = mod.startswith('enoki.packet')
    cls.IsDiff = mod.endswith('.ad')
    cls.IsLLVM = mod.startswith('enoki.llvm')
    cls.IsCUDA = mod.startswith('enoki.cuda')
    cls.IsJIT = cls.IsLLVM or cls.IsCUDA

    name = cls.__name__
    cls.IsMatrix = 'Matrix' in name
    cls.IsComplex = 'Complex' in name
    cls.IsQuaternion = 'Quaternion' in name
    cls.IsSpecial = cls.IsMatrix or cls.IsComplex or cls.IsQuaternion
    cls.IsVector = cls.Size != enoki.Dynamic and not \
        (cls.IsPacket and cls.Depth == 1) and not cls.IsSpecial

    if cls.IsSpecial:
        for i, c in enumerate(name):
            if c.isdigit():
                cls.Prefix = name[:i]
                break

        if cls.IsComplex:
            cls.real = prop_x
            cls.imag = prop_y
        elif cls.IsQuaternion:
            cls.real = prop_w
            cls.imag = prop_xyz
            cls.Imag = getattr(sys.modules.get(mod),
                               name.replace('Quaternion4', 'Array3'))
            cls.Complex = getattr(sys.modules.get(mod),
                                  name.replace('Quaternion4', 'Complex2'))
    else:
        cls.Prefix = 'Array'

    if not cls.IsSpecial or cls.IsQuaternion:
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
        array_name("Array", enoki.VarType.Bool,
                   cls.Shape, cls.IsScalar))


# Read JIT variable IDs, used by ek.cuda/llvm.loop
def read_indices(*args):
    result = []
    for a in args:
        if enoki.is_array_v(a):
            if a.Depth > 1:
                for i in range(len(a)):
                    result.extend(read_indices(a.entry_ref_(i)))
            elif a.IsDiff:
                if enoki.grad_enabled(a):
                    raise enoki.Exception(
                        'Symbolic loop encountered a differentiable array '
                        'with enabled gradients! This is not supported.')
                result.extend(read_indices(a.detach_()))
            elif a.IsJIT:
                result.append(a.index())
        elif isinstance(a, tuple) or isinstance(a, list):
            for b in a:
                result.extend(read_indices(b))
        elif enoki.is_enoki_struct_v(a):
            for k, v in type(a).ENOKI_STRUCT.items():
                result.extend(read_indices(getattr(a, k)))
        else:
            print(" do not know what to do with %s\n" % str(a))
    return result


# Write JIT variable IDs, used by ek.cuda/llvm.loop
def write_indices(indices, *args):
    for a in args:
        if enoki.is_array_v(a):
            if a.Depth > 1:
                for i in range(len(a)):
                    write_indices(indices, a.entry_ref_(i))
            elif a.IsDiff:
                if enoki.grad_enabled(a):
                    raise enoki.Exception(
                        'Symbolic loop encountered a differentiable array '
                        'with enabled gradients! This is not supported.')
                write_indices(indices, a.detach_())
            elif a.IsJIT:
                a.set_index_(indices.pop(0))
        elif isinstance(a, tuple) or isinstance(a, list):
            for b in a:
                write_indices(indices, b)
        elif enoki.is_enoki_struct_v(a):
            for k, v in type(a).ENOKI_STRUCT.items():
                write_indices(indices, getattr(a, k))
        else:
            print(" do not know what to do with %s\n" % str(a))
