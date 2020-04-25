import enoki
import sys

VAR_TYPE_NAME = [
    'Invalid', 'Int8',   'UInt8',   'Int16',   'UInt16',  'Int32', 'UInt32',
    'Int64',   'UInt64', 'Float16', 'Float32', 'Float64', 'Bool',  'Pointer'
]

VAR_TYPE_SUFFIX = [
    '???', 'i8',  'u8',  'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'f16',
    'f32', 'f64', 'b', 'p'
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
    success = False
    err = None

    try:
        if n == 1:
            o = args[0]

            if isinstance(o, enoki.ArrayBase) or \
               isinstance(o, list) or \
               isinstance(o, tuple):
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
                        self.set_coeff(i, o)
                else:
                    if isinstance(o[0], value_type):
                        for i in range(size):
                            self.set_coeff(i, o[i])
                    else:
                        for i in range(size):
                            self.set_coeff(i, value_type(o[i]))
                success = True
            else:
                if dynamic:
                    size = 1
                    self.init_(size)
                if not isinstance(o, value_type):
                    o = value_type(o)
                for i in range(size):
                    self.set_coeff(i, o)
                success = True
        elif n == size or dynamic:
            if dynamic:
                size = n
                self.init_(size)
            for i in range(size):
                self.set_coeff(i, value_type(args[i]))
            success = True
    except Exception as e:
        err = e

    if not success:
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
    cls.IsScalar = cls.__module__ == 'enoki.scalar'
    cls.IsPacket = cls.__module__ == 'enoki.packet'
    cls.IsLLVM = cls.__module__ == 'enoki.llvm'
    cls.IsCUDA = cls.__module__ == 'enoki.cuda'
    cls.IsJIT = cls.IsLLVM or cls.IsCUDA
    cls.MaskType = getattr(
        sys.modules.get(cls.__module__),
        array_name(enoki.VarType.Bool, cls.Depth, cls.Size, cls.IsScalar))
