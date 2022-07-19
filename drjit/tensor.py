from typing import Type
import drjit as _dr
from collections.abc import Sequence as _Sequence

def upsample(t, shape=None, scale_factor=None):
    '''
    upsample(source, shape=None, scale_factor=None)
    Up-sample the input tensor or texture according to the provided shape.

    Alternatively to specifying the target shape, a scale factor can be provided.

    The behavior of this function depends on the type of ``source``:

    1. When ``source`` is a Dr.Jit tensor, nearest neighbor up-sampling will use
    hence the target ``shape`` values must be multiples of the source shape
    values. When `scale_factor` is used, its values must be integers.

    2. When ``source`` is a Dr.Jit texture type, the up-sampling will be
    performed according to the filter mode set on the input texture. Target
    ``shape`` values are not required to be multiples of the source shape values.
    When `scale_factor` is used, its values must be integers.

    Args:
        source (object): A Dr.Jit tensor or texture type.

        shape (list): The target shape (optional)

        scale_factor (list): The scale factor to apply to the current shape (optional)

    Returns:
        object: the up-sampled tensor or texture object. The type of the output will be the same as the type of the source.
    '''
    if  not getattr(t, 'IsTexture', False) and not _dr.is_tensor_v(t):
        raise TypeError("upsample(): unsupported input type, expected Dr.Jit "
                        "tensor or texture type!")

    if shape is not None and scale_factor is not None:
        raise TypeError("upsample(): shape and scale_factor arguments cannot "
                        "be defined at the same time!")

    if shape is not None:
        if not isinstance(shape, _Sequence):
            raise TypeError("upsample(): unsupported shape type, expected a list!")

        if len(shape) > len(t.shape):
            raise TypeError("upsample(): invalid shape size!")

        shape = list(shape) + list(t.shape[len(shape):])

        scale_factor = []
        for i, s in enumerate(shape):
            if type(s) is not int:
                raise TypeError("upsample(): target shape must contain integer values!")

            if s < t.shape[i]:
                raise TypeError("upsample(): target shape values must be larger "
                                "or equal to input shape! (%i vs %i)" % (s, t.shape[i]))

            if _dr.is_tensor_v(t):
                factor = s / float(t.shape[i])
                if factor != int(factor):
                    raise TypeError("upsample(): target shape must be multiples of "
                                    "the input shape! (%i vs %i)" % (s, t.shape[i]))
    else:
        if not isinstance(scale_factor, _Sequence):
            raise TypeError("upsample(): unsupported scale_factor type, expected a list!")

        if len(scale_factor) > len(t.shape):
            raise TypeError("upsample(): invalid scale_factor size!")

        scale_factor = list(scale_factor)
        for i in range(len(t.shape) - len(scale_factor)):
            scale_factor.append(1)

        shape = []
        for i, factor in enumerate(scale_factor):
            if type(factor) is not int:
                raise TypeError("upsample(): scale_factor must contain integer values!")

            if factor < 1:
                raise TypeError("upsample(): scale_factor values must be greater "
                                "than 0!")

            shape.append(factor * t.shape[i])

    if getattr(t, 'IsTexture', False):
        value_type = type(t.value())
        dim = len(t.shape) - 1

        if t.shape[dim] != shape[dim]:
            raise TypeError("upsample(): channel counts doesn't match input texture!")

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
        width = _dr.width(values[0])
        index = _dr.arange(_dr.uint32_array_t(value_type), width)
        data = _dr.zeros(value_type, width * channels)
        for c in range(channels):
            _dr.scatter(data, values[c], channels * index + c)

        # Create the up-sampled texture
        texture = type(t)(shape[:-1], channels,
                          use_accel=t.use_accel(),
                          filter_mode=t.filter_mode(),
                          wrap_mode=t.wrap_mode())
        texture.set_value(data)

        return texture
    else:
        dim = len(shape)
        size = _dr.prod(shape[:dim])
        base = _dr.arange(_dr.uint32_array_t(type(t.array)), size)

        index = 0
        stride = 1
        for i in reversed(range(dim)):
            ratio = shape[i] // t.shape[i]
            index += (base // ratio % t.shape[i]) * stride
            base //= shape[i]
            stride *= t.shape[i]

        return type(t)(_dr.gather(type(t.array), t.array, index), shape)
