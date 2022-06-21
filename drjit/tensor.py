import drjit as _dr

def upsample(v, shape):
    r'''
    upsample(source, shape)
    Up-sample the input tensor or texture according to the provided shape.

    ``source`` is assumed to have as shape ``(D_0, ..., D_N)`` or
    ``(D_0, ..., D_N, C)`` with ``D_0, ..., D_N`` the resolution along each
    dimensions and ``C`` the channel count. This function assumes the last
    dimension of `source` to represent the different channels when its
    resolution is not a power of two. The resolution of all other dimensions
    (``D_0, ..., D_N``) must be powers of two.

    The target ``shape`` defines the target resolution along each dimensions
    ``T_0, ..., T_N`` which have to be powers of two. Optionally, ``shape`` can also
    specify the channel number which need to match the one of ``source``.

    This function also supports Dr.Jit texture, in which case the underlying
    tensor will be up-sampled and returned in a new texture object.

    Args:
        source (object): A Dr.Jit tensor or texture type.

        shape (list): The target shape

    Returns:
        object: the up-sampled tensor or texture object
    '''
    if hasattr(v, 'IsTexture') and v.IsTexture:
        return type(v)(
            _dr.upsample(v.tensor(), shape),
            use_accel = v.use_accel(),
            migrate = v.migrate(),
            filter_mode = v.filter_mode(),
            wrap_mode = v.wrap_mode(),
        )

    if not _dr.is_tensor_v(v):
        raise TypeError("upsample(): unsupported input type, expected Dr.Jit "
                        "tensor or texture type!")

    dim = len(shape)
    if dim != len(v.shape) and dim != len(v.shape) - 1:
        raise TypeError("upsample(): invalid number of dimensions in target shape."
                        "Should be equal to the number of dimensions of the tensor"
                        "or one less (assuming the last tensor dimension "
                        "represents a number of channels) ")

    def is_power_of_2(x):
        return ((x != 0) and not (x & (x - 1)))

    channels = 1
    if dim == len(v.shape) - 1:
        channels = v.shape[dim]
    elif not is_power_of_2(v.shape[dim-1]) or v.shape[dim-1] == 1:
        channels = v.shape[dim-1]
        dim -= 1

        if shape[dim] != v.shape[dim]:
            raise TypeError("upsample(): target channel count must match the one"
                            "of the input tensor!")

    for i in range(dim):
        if not is_power_of_2(v.shape[i]):
            raise TypeError("upsample(): tensor resolution must be a power of two!")

        if not is_power_of_2(shape[i]):
            raise TypeError("upsample(): target resolution must be a power of two!")

        if shape[i] < v.shape[i]:
            raise TypeError("upsample(): target resolution must be larger or "
                            "equal to tensor resolution!")

    size = _dr.prod(shape[:dim]) * channels
    base = _dr.arange(_dr.uint32_array_t(type(v.array)), size)

    index = base % channels
    base //= channels

    stride = 1
    for i in range(dim):
        ratio = shape[i] // v.shape[i]
        index += (base // ratio % v.shape[i]) * stride * channels
        base //= shape[i]
        stride *= v.shape[i]

    shape = [shape[i] for i in range(dim)]
    shape.append(channels)
    return type(v)(_dr.gather(type(v.array), v.array, index), shape)
