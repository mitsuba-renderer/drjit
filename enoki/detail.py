import enoki
import sys

VAR_TYPE_NAME = [
    'Invalid', 'Bool',  'Int8',   'UInt8',   'Int16',   'UInt16',  'Int',
    'UInt',    'Int64', 'UInt64', 'Float16', 'Float',   'Float64', 'Pointer'
]

VAR_TYPE_SUFFIX = [
    '???', 'b', 'i8',  'u8',  'i16', 'u16', 'i', 'u', 'i64', 'u64', 'f16',
    'f', 'f64',  'p'
]


def array_name(vt, depth, size, scalar):
    """
    Determines the name of an array (e.g. Float32 ArrayXf32, etc.). This
    function is used when arrays are created during initialization of the Enoki
    extension module, and during implicit type promotion where array types are
    determined dynamically.

    Parameter ``vt`` (``enoki.VarType``):
        Underlying scalar type (e.g. ``VarType.Int32``) of the desired array

    Parameter ``depth`` (``int``):
        Depth of the desired array type (0-2D supported)

    Parameter ``size`` (``int``):
        Number of components

    Parameter ``scalar`` (``bool``):
        Arrays in the ``enoki.scalar.*`` module use a different depth
        convention, which is indicated via this parameter.
    """

    if depth == 0 or (not scalar and depth == 1):
        return VAR_TYPE_NAME[int(vt)]

    return "Array%s%s" % (
        'X' if size == enoki.Dynamic else str(size),
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
                return t.gather_(data, offset + stride * i, True)
        else:
            result = t()
            for j in range(size):
                result[j] = load(t.Value, i + 1, offset + stride * j)
            return result

    return load(t, 0, 0)


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

            if issubclass(t, enoki.ArrayBase) or \
               issubclass(t, list) or \
               issubclass(t, tuple):
                os = len(o)
                if dynamic:
                    size = os
                    self.init_(size)
                if size == 0:
                    pass
                elif size != os:
                    if not isinstance(o, value_type):
                        o = value_type(o)
                    for i in range(size):
                        self.set_entry_(i, o)
                else:
                    if isinstance(o[0], value_type):
                        for i in range(size):
                            self.set_entry_(i, o[i])
                    else:
                        for i in range(size):
                            self.set_entry_(i, value_type(o[i]))
            elif issubclass(t, (int, float)):
                if dynamic:
                    size = 1
                    self.init_(size)
                if not isinstance(o, value_type):
                    o = value_type(o)
                for i in range(size):
                    self.set_entry_(i, o)
            elif mod == 'numpy':
                if o.dtype != self.Type.NumPy:
                    raise Exception("Incompatible dtype!")
                s1 = tuple(reversed(enoki.shape(self)))
                s2 = o.shape
                dim = len(s1)
                if dim != len(s2):
                    raise Exception("Incompatible dimension!")
                for i in range(dim):
                    if s1[i] != s2[i] and s1[i] != 0:
                        raise Exception("Incompatible shape!")
                if dim == 0:
                    pass
                elif dim == 1:
                    import numpy
                    o = numpy.ascontiguousarray(o)
                    d = o.__array_interface__['data'][0]
                    self.assign_(self.load_(d, s2[0]))
                else:
                    for i in range(s1[-1]):
                        self.set_entry_(i, value_type(o[..., i]))
            elif mod == 'builtins' and name == 'PyCapsule':
                self.assign_(array_from_dlpack(type(self), o))
            elif mod == 'torch':
                from torch.utils.dlpack import to_dlpack
                self.assign_(array_from_dlpack(type(self), to_dlpack(o)))
            elif mod.startswith('tensorflow.'):
                from tensorflow.experimental.dlpack import to_dlpack
                self.assign_(array_from_dlpack(type(self), to_dlpack(o)))
            elif mod.startswith('jax.'):
                from jax.dlpack import to_dlpack
                self.assign_(array_from_dlpack(type(self), to_dlpack(o)))
            else:
                raise Exception('Don\'t know how to create an Enoki array from type \"%s.%s\"!' % (mod, name))
        elif n == size or dynamic:
            if dynamic:
                size = n
                self.init_(size)
            for i in range(size):
                self.set_entry_(i, value_type(args[i]))
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


def array_configure(cls):
    """Populates an Enoki array class with extra type trait fields"""
    depth = 1

    value = cls.Value
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
    cls.IsScalar = cls.__module__.startswith('enoki.scalar')
    cls.IsPacket = cls.__module__.startswith('enoki.packet')
    cls.IsLLVM = cls.__module__.startswith('enoki.llvm')
    cls.IsCUDA = cls.__module__.startswith('enoki.cuda')
    cls.IsDiff = '.ad' in cls.__module__
    cls.IsJIT = cls.IsLLVM or cls.IsCUDA
    cls.MaskType = getattr(
        sys.modules.get(cls.__module__),
        array_name(enoki.VarType.Bool, cls.Depth, cls.Size, cls.IsScalar))
