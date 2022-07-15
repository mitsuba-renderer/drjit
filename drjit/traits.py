from drjit import VarType, Exception, Dynamic
import sys as _sys
from collections.abc import Mapping as _Mapping, \
                            Sequence as _Sequence


def is_array_v(arg, /):
    '''
    is_array_v(arg, /)
    Check if the input is a Dr.Jit array instance or type

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` or type(``arg``) is a Dr.Jit array type, and ``False`` otherwise
    '''
    return getattr(arg, 'IsDrJit', False)


def size_v(arg, /):
    '''
    size_v(arg, /)
    Return the (static) size of the outermost dimension of the provided Dr.Jit
    array instance or type

    Note that this function mainly exists to query type-level information. Use the
    Python ``len()`` function to query the size in a way that does not distinguish
    between static and dynamic arrays.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        int: Returns either the static size or :py:data:`drjit.Dynamic` when
        ``arg`` is a dynamic Dr.Jit array. Returns ``1`` for all other types.
    '''
    return getattr(arg, 'Size', 1)


def depth_v(arg, /):
    '''
    depth_v(arg, /)
    Return the depth of the provided Dr.Jit array instance or type

    For example, an array consisting of floating point values (for example,
    :py:class:`drjit.scalar.Array3f`) has depth ``1``, while an array consisting of
    sub-arrays (e.g., :py:class:`drjit.cuda.Array3f`) has depth ``2``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        int: Returns the depth of the input, if it is a Dr.Jit array instance or
        type. Returns ``0`` for all other inputs.
    '''
    return getattr(arg, 'Depth', 0)


def scalar_t(arg, /):
    '''
    scalar_t(arg, /)
    Return the *scalar type* associated with the provided Dr.Jit array or type (i.e., the
    representation of elements at the lowest level)

    When the input is not a Dr.Jit array or type, the function returns the input
    unchanged. The following assertions illustrate the behavior of
    :py:func:`scalar_t`.


    .. code-block::

        assert dr.scalar_t(dr.scalar.Array3f) is bool
        assert dr.scalar_t(dr.cuda.Array3f) is float
        assert dr.scalar_t(dr.cuda.Matrix4f) is float
        assert dr.scalar_t("test") is str

    Args:
        arg (object): An arbitrary Python object

    Returns:
        int: Returns the scalar type of the provided Dr.Jit array, or the type of
        the input.
    '''
    if not isinstance(arg, type):
        arg = type(arg)
    return getattr(arg, 'Scalar', arg)


def value_t(arg, /):
    '''
    value_t(arg, /)
    Return the *value type* underlying the provided Dr.Jit array or type (i.e., the
    type of values obtained by accessing the contents using a 1D index).

    When the input is not a Dr.Jit array or type, the function returns the input
    unchanged. The following code fragment shows several example uses of
    :py:func:`value_t`.

    .. code-block::

        assert dr.value_t(dr.scalar.Array3f) is float
        assert dr.value_t(dr.cuda.Array3f) is dr.cuda.Float
        assert dr.value_t(dr.cuda.Matrix4f) is dr.cuda.Array4f
        assert dr.value_t(dr.cuda.TensorXf) is float
        assert dr.value_t("test") is str

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Returns the value type of the provided Dr.Jit array, or the type of
        the input.
    '''
    if not isinstance(arg, type):
        arg = type(arg)
    return getattr(arg, 'Value', arg)


def mask_t(arg, /):
    '''
    mask_t(arg, /)
    Return the *mask type* associated with the provided Dr.Jit array or type (i.e., the
    type produced by comparisons involving the argument).

    When the input is not a Dr.Jit array or type, the function returns the scalar
    Python ``bool`` type. The following assertions illustrate the behavior of
    :py:func:`mask_t`.


    .. code-block::

        assert dr.mask_t(dr.scalar.Array3f) is dr.scalar.Array3b
        assert dr.mask_t(dr.cuda.Array3f) is dr.cuda.Array3b
        assert dr.mask_t(dr.cuda.Matrix4f) is dr.cuda.Array44b
        assert dr.mask_t("test") is bool

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Returns the mask type associated with the input or ``bool`` when the
        input is not a Dr.Jit array.
    '''
    if not isinstance(arg, type):
        arg = type(arg)
    return getattr(arg, 'MaskType', bool)


def is_mask_v(arg, /):
    '''
    is_mask_v(arg, /)
    Check whether the input array instance or type is a Dr.Jit mask array or a
    Python ``bool`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents a Dr.Jit mask array or Python ``bool``
        instance or type.
    '''
    return scalar_t(arg) is bool


def is_float_v(arg, /):
    '''
    is_float_v(arg, /)
    Check whether the input array instance or type is a Dr.Jit floating point array
    or a Python ``float`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents a Dr.Jit floating point array or
        Python ``float`` instance or type.
    '''
    return scalar_t(arg) is float


def is_integral_v(arg, /):
    '''
    is_integral_v(arg, /)
    Check whether the input array instance or type is an integral Dr.Jit array
    or a Python ``int`` value/type.

    Note that a mask array is not considered to be integral.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an integral Dr.Jit array or
        Python ``int`` instance or type.
    '''
    return scalar_t(arg) is int


def is_arithmetic_v(arg, /):
    '''
    is_arithmetic_v(arg, /)
    Check whether the input array instance or type is an arithmetic Dr.Jit array
    or a Python ``int`` or ``float`` value/type.

    Note that a mask type (e.g. ``bool``, :py:class:`drjit.scalar.Array2b`, etc.)
    is *not* considered to be arithmetic.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an arithmetic Dr.Jit array or
        Python ``int`` or ``float`` instance or type.
    '''
    return scalar_t(arg) in [int, float]


def is_cuda_v(arg, /):
    '''
    is_cuda_v(arg, /)
    Check whether the input is a Dr.Jit CUDA array instance or type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an array type from the
        ``drjit.cuda.*`` namespace, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsCUDA', False)


def is_llvm_v(arg, /):
    '''
    is_llvm_v(arg, /)
    Check whether the input is a Dr.Jit LLVM array instance or type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an array type from the
        ``drjit.llvm.*`` namespace, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsLLVM', False)


def is_jit_v(arg, /):
    '''
    is_jit_v(arg, /)
    Check whether the input array instance or type represents a type that
    undergoes just-in-time compilation.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an array type from the
        ``drjit.cuda.*`` or ``drjit.llvm.*`` namespaces, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsJIT', False)


def is_diff_v(arg, /):
    '''
    is_diff_v(arg, /)
    Check whether the input is a differentiable Dr.Jit array instance or type.

    Note that this is a type-based statement that is unrelated to mathematical
    differentiability. For example, the integral type :py:class:`drjit.cuda.ad.Int`
    from the CUDA AD namespace satisfies ``is_diff_v(..) = 1``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an array type from the
        ``drjit.[cuda/llvm].ad.*`` namespace, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsDiff', False)


def is_complex_v(arg, /):
    '''
    is_complex_v(arg, /)
    Check whether the input is a Dr.Jit array instance or type representing a complex number.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsComplex', False) or isinstance(arg, complex)


def is_matrix_v(arg, /):
    '''
    is_matrix_v(arg, /)
    Check whether the input is a Dr.Jit array instance or type representing a matrix.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsMatrix', False)


def is_quaternion_v(arg, /):
    '''
    is_quaternion_v(arg, /)
    Check whether the input is a Dr.Jit array instance or type representing a quaternion.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsQuaternion', False)


def is_tensor_v(arg, /):
    '''
    is_tensor_v(arg, /)
    Check whether the input is a Dr.Jit array instance or type representing a tensor.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsTensor', False)


def is_texture_v(a):
    return getattr(a, 'IsTexture', False)


def is_vector_v(a):
    return getattr(a, 'IsVector', False)


def is_special_v(arg, /):
    '''
    is_special_v(arg, /)
    Check whether the input is a *special* Dr.Jit array instance or type.

    A *special* array type requires precautions when performing arithmetic
    operations like multiplications (complex numbers, quaternions, matrices).

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.
    '''
    return getattr(arg, 'IsSpecial', False)


def is_static_array_v(a):
    return getattr(a, 'Size', Dynamic) != Dynamic


def is_dynamic_array_v(a):
    return getattr(a, 'Size', Dynamic) == Dynamic


def is_dynamic_v(a):
    return getattr(a, 'IsDynamic', False)


def is_unsigned_v(arg, /):
    '''
    is_unsigned_v(arg, /)
    Check whether the input array instance or type is an unsigned integer Dr.Jit
    array or a Python ``bool`` value/type (masks and boolean values are also
    considered to be unsigned).

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an unsigned Dr.Jit array or
        Python ``bool`` instance or type.
    '''
    if not is_array_v(arg):
        return False

    vt = arg.Type

    return vt == VarType.UInt8 or \
        vt == VarType.UInt16 or \
        vt == VarType.UInt32 or \
        vt == VarType.UInt64


def is_signed_v(arg, /):
    '''
    is_signed_v(arg, /)
    Check whether the input array instance or type is an signed Dr.Jit array
    or a Python ``int`` or ``float`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an signed Dr.Jit array or
        Python ``int`` or ``float`` instance or type.
    '''
    return not is_unsigned_v(arg)


def is_iterable_v(a):
    if isinstance(a, str):
        return False
    try:
        iter(a)
        return True
    except TypeError:
        return False


def bool_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into a *boolean* version with
    the same element size.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3i64`), it
       returns an *boolean* version (e.g. :py:class:`drjit.cuda.Array3b64`).

    2. When the input isn't a type, it returns ``bool_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``bool``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return mask_t(arg)

def int_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into a *signed integer*
    version with the same element size.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
       returns an *signed integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

    2. When the input isn't a type, it returns ``int_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``int``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    if not is_array_v(arg):
        return int

    size = arg.Type.Size
    if size == 1:
        vt = VarType.Int8
    elif size == 2:
        vt = VarType.Int16
    elif size == 4:
        vt = VarType.Int32
    elif size == 8:
        vt = VarType.Int64
    else:
        raise Exception("Unsupported variable size!")

    t = arg.ReplaceScalar(vt)
    return t


def uint_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into a *unsigned integer*
    version with the same element size.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
       returns an *unsigned integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

    2. When the input isn't a type, it returns ``uint_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``int``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    if not is_array_v(arg):
        return int

    size = arg.Type.Size
    if size == 1:
        vt = VarType.UInt8
    elif size == 2:
        vt = VarType.UInt16
    elif size == 4:
        vt = VarType.UInt32
    elif size == 8:
        vt = VarType.UInt64
    else:
        raise Exception("Unsupported variable size!")

    t = arg.ReplaceScalar(vt)
    return t


def float_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into a *floating point*
    version with the same element size.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u64`), it
       returns an *floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

    2. When the input isn't a type, it returns ``float_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``float``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    if not is_array_v(arg):
        return int

    size = arg.Type.Size
    if size == 2:
        vt = VarType.Float16
    elif size == 4:
        vt = VarType.Float32
    elif size == 8:
        vt = VarType.Float64
    else:
        raise Exception("Unsupported variable size!")

    return arg.ReplaceScalar(vt)


def uint32_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into a *unsigned 32 bit*
    version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
       returns an *unsigned 32 bit* version (e.g. :py:class:`drjit.cuda.Array3u`).

    2. When the input isn't a type, it returns ``uint32_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``int``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return arg.ReplaceScalar(VarType.UInt32) if is_array_v(arg) else int


def int32_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into a *signed 32 bit*
    version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
       returns an *signed 32 bit* version (e.g. :py:class:`drjit.cuda.Array3i`).

    2. When the input isn't a type, it returns ``int32_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``int``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return arg.ReplaceScalar(VarType.Int32) if is_array_v(arg) else int


def uint64_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into an *unsigned 64 bit*
    version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
       returns an *unsigned 64 bit* version (e.g. :py:class:`drjit.cuda.Array3u64`).

    2. When the input isn't a type, it returns ``uint64_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``int``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return arg.ReplaceScalar(VarType.UInt64) if is_array_v(arg) else int


def int64_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into an *signed 64 bit* version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
       returns an *signed 64 bit* version (e.g. :py:class:`drjit.cuda.Array3i64`).

    2. When the input isn't a type, it returns ``int64_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``int``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return arg.ReplaceScalar(VarType.Int64) if is_array_v(arg) else int


def float32_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into an 32 bit floating point version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
       returns a *32 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f`).

    2. When the input isn't a type, it returns ``float32_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``float``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return arg.ReplaceScalar(VarType.Float32) if is_array_v(arg) else float


def float64_array_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into an 64 bit floating point version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
       returns a *64 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

    2. When the input isn't a type, it returns ``float64_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``float``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    return arg.ReplaceScalar(VarType.Float64) if is_array_v(arg) else float


def diff_array_t(a, allow_non_array=False):
    '''
    Converts the provided Dr.Jit array/tensor type into a differentiable version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
       returns a *differentiable* version (e.g. :py:class:`drjit.cuda.ad.Array3f`).

    2. When the input isn't a type, it returns ``diff_array_t(type(arg))(arg)``.

    3. When the input is is a list or a tuple, it recursively call ``diff_array_t`` over all elements.

    4. When the input is not a Dr.Jit array or type, the function throws an exception.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    if isinstance(a, tuple):
        return tuple(diff_array_t(v, allow_non_array=allow_non_array)
                     for v in a)
    elif isinstance(a, list):
        return [diff_array_t(v) for v in a]
    elif not is_array_v(a):
        if allow_non_array:
            return a
        raise Exception("diff_array_t(): requires an Dr.Jit input array!")
    elif not isinstance(a, type):
        return diff_array_t(type(a), allow_non_array=allow_non_array)(a)
    elif a.IsDiff:
        return a
    else:
        return a.ReplaceScalar(a.Type, diff=True)


def detached_t(arg):
    '''
    Converts the provided Dr.Jit array/tensor type into an non-differentiable version.

    This function implements the following set of behaviors:

    1. When invoked with a differentiable Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.ad.Array3f`), it
       returns a non-differentiable version (e.g. :py:class:`drjit.cuda.Array3f`).

    2. When the input isn't a type, it returns ``detached_t(type(arg))``.

    3. When the input type is non-differentiable or not a Dr.Jit array type, the function returns it unchanged.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.
    '''
    if not is_array_v(arg):
        raise Exception("detached_t(): requires an Dr.Jit input array!")
    elif not isinstance(arg, type):
        return detached_t(type(arg))
    elif not arg.IsDiff:
        return arg
    else:
        return arg.ReplaceScalar(arg.Type, diff=False)


def is_struct_v(arg, /):
    '''
    is_struct_v(arg, /)
    Check if the input is a Dr.Jit-compatible data structure

    Custom data structures can be made compatible with various Dr.Jit operations by
    specifying a ``DRJIT_STRUCT`` member. See the section on :ref:`custom data
    structure <custom-struct>` for details. This type trait can be used to check
    for the existence of such a field.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` has a ``DRJIT_STRUCT`` member
    '''
    return hasattr(arg, 'DRJIT_STRUCT')


def leaf_array_t(arg):
    '''
    Extracts a leaf array type underlying a Python object tree, with a preference
    for differentiable arrays.

    This function implements the following set of behaviors:

    1. When the input isn't a type, it returns ``leaf_array_t(type(arg))``.

    2. When invoked with a Dr.Jit array type, returns the lowest-level array type
       underlying a potentially nested array.

    3. When invoked with a sequence, mapping or custom data structure made of Dr.Jit arrays,
       examines underlying Dr.Jit array types and returns the lowest-level array type with
       a preference for differentiable arrays and floating points arrays.
       E.g. when passing a list containing arrays of type :py:class:`drjit.cuda.ad.Float` and :py:class:`drjit.cuda.UInt`,
       the function will return :py:class:`drjit.cuda.ad.Float`.

    4. Otherwise returns ``None``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the extraction as described above.
    '''
    t = None

    if isinstance(arg, _Sequence):
        for e in arg:
            t = leaf_array_t(e)
            if is_diff_v(t) and is_float_v(t):
                break
    elif isinstance(arg, _Mapping):
        for k, v in arg:
            t = leaf_array_t(v)
            if is_diff_v(t) and is_float_v(t):
                break
    elif is_struct_v(arg):
        for k in type(arg).DRJIT_STRUCT.keys():
            t = leaf_array_t(getattr(arg, k))
            if is_diff_v(t) and is_float_v(t):
                break
    elif is_tensor_v(arg):
        t = leaf_array_t(arg.Array)
    elif is_array_v(arg):
        t = arg
        if not isinstance(t, type):
            t = type(t)
        while is_array_v(value_t(t)):
            t = t.Value

    return t
