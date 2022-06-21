import drjit as _dr
from collections.abc import Sequence as _Sequence

def upsample(t, shape):
    r'''
    upsample(source, shape)
    Up-sample the input tensor or texture according to the provided shape.

    This behavior of this function depends on the type of ``source``:

    1. When ``source`` is a Dr.Jit tensor, ``source`` is assumed to have a shape
    that follows the following patterns: ``(D_0, ..., D_N)`` or ``(D_0, ..., D_N, C)``
    with ``D_0, ..., D_N`` the resolution along each dimensions and ``C`` the
    channel count. This function assumes that the last dimension of `source`
    represents different channels when its resolution is not a power of two. The
    resolution of all other dimensions (``D_0, ..., D_N``) must be powers of two.

    The target ``shape`` defines the target resolution along each dimensions
    ``T_0, ..., T_N`` which have to be powers of two. Optionally, ``shape`` can also
    specify the channel number which need to match the one of ``source``.

    2. When ``source`` is a Dr.Jit texture type, this function supports source
    and target shapes that are not powers of two. In this case, ``shape`` should
    define the target resolution along each dimensions ``T_0, ..., T_N``. The
    up-sampling will be performed using the interpolation scheme set on the
    texture itself, otherwise nearest neighbor will be used instead.

    Args:
        source (object): A Dr.Jit tensor or texture type.

        shape (list): The target shape

    Returns:
        object: the up-sampled tensor or texture object
    '''

    if not isinstance(shape, _Sequence):
        raise TypeError("upsample(): unsupported shape input type, expected a list!")

    if  not getattr(t, 'IsTexture', False) and not _dr.is_tensor_v(t):
        raise TypeError("upsample(): unsupported input type, expected Dr.Jit "
                        "tensor or texture type!")

    if len(shape) != len(t.shape) and len(shape) != len(t.shape) - 1:
        raise TypeError("upsample(): invalid number of dimensions in target shape!"
                        "Should be equal to the number of dimensions of the input"
                        "or one less (assuming the last dimension represents "
                        "different channels)")

    if getattr(t, 'IsTexture', False):
        value_type = type(t.value())
        dim = len(t.shape) - 1

        if dim != len(shape):
            raise TypeError("upsample(): invalid number of dimensions in target shape!")

        for i in range(dim):
            if shape[i] < t.shape[i]:
                raise TypeError("upsample(): target resolution must be larger or "
                                "equal to texture resolution!")

         # Create the query coordinates
        coords = list(_dr.meshgrid(*[
                _dr.linspace(value_type, 0.0, 1.0, shape[i], endpoint=False)
                for i in range(dim)
            ],
            indexing='ij'
        ))

        # Offset coordinates by half a voxel to hit the center of the new voxels
        for i in range(dim):
            coords[i] += 0.5 / shape[i]

        # Reverse coordinates order according to dr.Texture convention
        coords.reverse()

        # Evaluate the texture at all voxel coordinates with interpolation
        values = t.eval(coords)

        # Concatenate output values to a flatten buffer
        channels = len(values)
        width = _dr.width(values[0]) * channels
        index = _dr.arange(_dr.uint32_array_t(value_type), width) // channels
        data = _dr.zeros(value_type, width)
        for c in range(channels):
            _dr.scatter(data, values[c], index + c)

        # Create the up-sampled texture
        tex = type(t)(shape, channels,
                      use_accel=t.use_accel(),
                      migrate=t.migrate(),
                      filter_mode=t.filter_mode(),
                      wrap_mode=t.wrap_mode())
        tex.set_value(data)

        return tex

    def is_power_of_2(x):
        return ((x != 0) and not (x & (x - 1)))

    dim = len(shape)
    channels = 1
    if dim == len(t.shape) - 1:
        channels = t.shape[dim]
    elif not is_power_of_2(t.shape[dim-1]) or t.shape[dim-1] == 1:
        channels = t.shape[dim-1]
        dim -= 1

        if shape[dim] != t.shape[dim]:
            raise TypeError("upsample(): target channel count must match the one"
                            "of the input tensor!")

    for i in range(dim):
        if not is_power_of_2(t.shape[i]):
            raise TypeError("upsample(): tensor resolution must be a power of two!")

        if not is_power_of_2(shape[i]):
            raise TypeError("upsample(): target resolution must be a power of two!")

        if shape[i] < t.shape[i]:
            raise TypeError("upsample(): target resolution must be larger or "
                            "equal to tensor resolution!")

    size = _dr.prod(shape[:dim]) * channels
    base = _dr.arange(_dr.uint32_array_t(type(t.array)), size)

    index = base % channels
    base //= channels

    stride = 1
    for i in range(dim):
        ratio = shape[i] // t.shape[i]
        index += (base // ratio % t.shape[i]) * stride * channels
        base //= shape[i]
        stride *= t.shape[i]

    shape = [shape[i] for i in range(dim)]
    shape.append(channels)
    return type(t)(_dr.gather(type(t.array), t.array, index), shape)
