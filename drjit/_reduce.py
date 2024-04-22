import drjit as dr
from typing import TypeVar, Tuple, Literal, List, Protocol, Union, cast

ArrayT = TypeVar("ArrayT", bound=dr.ArrayBase)

def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Turn a shape tuple into a C-style strides tuple"""
    val, ndim = 1, len(shape)
    strides = [0] * ndim
    for i in reversed(range(ndim)):
        strides[i] = val
        val *= shape[i]
    return tuple(strides)


class BinaryOp(Protocol):
    """Type signature of an array-valued binary reduction"""
    def __call__(self, arg0: ArrayT, arg1: ArrayT, /) -> ArrayT:
        ...

_reduce_ops: dict[dr.ReduceOp, BinaryOp] = {
    dr.ReduceOp.Add: cast(BinaryOp, lambda a, b: a + b),
    dr.ReduceOp.Mul: cast(BinaryOp, lambda a, b: a * b),
    dr.ReduceOp.Min: cast(BinaryOp, lambda a, b: dr.minimum(a, b)),
    dr.ReduceOp.Max: cast(BinaryOp, lambda a, b: dr.maximum(a, b)),
    dr.ReduceOp.Or: cast(BinaryOp, lambda a, b: a | b),
    dr.ReduceOp.And: cast(BinaryOp, lambda a, b: a & b),
}

def reduce_recursive(
    op: dr.ReduceOp,
    value: ArrayT,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    offset: dr.AnyArray,
    accum: ArrayT
) -> ArrayT:
    """
    Helper function that compiles a tensor reduction into nested loops
    containing a gather operation.
    """
    UInt32 = dr.uint32_array_t(value)
    if shape == ():
        return _reduce_ops[op](accum, dr.gather(type(value), value, offset))
    else:
        return dr.while_loop(
            label=f"Reduction {len(shape)}",
            labels=("k", "offset", "value", "accum"),
            state=(UInt32(0), UInt32(offset), value, accum),
            cond=lambda *args: args[0] < UInt32(shape[0]),
            body=lambda k, offset, value, accum: (
                k + 1,
                offset + UInt32(strides[0]),
                value,
                reduce_recursive(op, value, shape[1:], strides[1:], offset, accum),
            ),
            max_iterations=-1 if op is dr.ReduceOp.Add else None,
        )[3]


def tensor_reduce(
    op: dr.ReduceOp,
    value: ArrayT,
    axis: tuple[int, ...],
    mode: Literal["symbolic", "evaluated", None],
) -> ArrayT:
    """
    This function uses the operation ``op`` to reduce the tensor ``value``
    along the given axis/axes. It is an implementation detail of the top-level
    function ``drjit.reduce()`` used to handle tensor arguments.

    The function supports multiple evaluation strategies:

    1. If the desired reduction is over contiguous blocks, it recursively calls
       :py:func:`drjit.block_reduce` with the same ``mode`` parameter, which is
       potentially more efficient.

    2. ``mode="evaluated"``: Evaluate the input tensor, then gather and reduce
       within a symbolic loop to compute the elements of the output tensor.

       **Caveats**: can be slow when the reduced tensor is small, in which case
       there isn't enough parallelism to perform the operation efficiently.
       This strategy requires evaluating the input array, which is potentially
       costly in terms of CPU/GPU memory.

    3. ``mode="symbolic"``: Issue atomic scatter-reductions to populate the
       output tensor. Since explicit evaluation and storage are not required,
       this mode is preferable when the input tensor is very large (e.g., when
       it would not fit into memory).

       **Caveats**:  Sum reductions are subject to nondeterministic rounding
       error, and reductions to just a few elements can be subject to
       *contention*. See :py:func:`drjit.scatter_reduce` for a discussion of
       both points).
    """
    Tensor = type(value)
    Value = dr.array_t(value)
    Index = dr.uint32_array_t(Value)

    # Some information about the input shape
    in_shape = value.shape
    in_strides = _compute_strides(in_shape)

    # Compute the shape and strides of the reduced array, and the
    # shape and strides of the blocks over which the reduction takes place
    out_shape, out_strides_i = [], []
    block_shape, block_strides = [], []

    for i, size in enumerate(in_shape):
        if i in axis:
            if block_shape and in_shape[i] * in_strides[i] == block_strides[-1]:
                # Simplify by collapsing contiguous axes. This, e.g., reduces
                # the loop nesting level of the 'gather' strategy.
                block_shape[-1] *= size
                block_strides[-1] = in_strides[i]
            else:
                block_shape.append(size)
                block_strides.append(in_strides[i])
        else:
            out_shape.append(size)
            out_strides_i.append(in_strides[i])

    # Convert to tuples and compute strides
    block_shape = tuple(block_shape)
    block_strides = tuple(block_strides)
    out_shape = tuple(out_shape)
    out_strides_o = _compute_strides(out_shape)

    in_size = dr.prod(in_shape)
    out_size = dr.prod(out_shape)
    block_size = dr.prod(block_shape)
    in_array = value.array

    if mode == "symbolic":
        symbolic = True
    elif mode == "evaluated":
        symbolic = False
    elif mode is None:
        state = in_array.state

        if state is dr.VarState.Symbolic:
            # Reducing a symbolic variable is probably a bad idea.
            # As a default policy, let's error out here by trying
            # to evaluate the variable, which will display a long
            # and informative error message to the user.
            #
            # If a symbolic reduction of a symbolic variable is
            # truly desired, the user may specify mode="symbolic".

            dr.eval(in_array)

        can_reduce = dr.detail.can_scatter_reduce(Value, op)
        if not can_reduce:
            symbolic = False
        else:
            # Would it be reasonable to evaluate the input array?
            is_evaluated = state is dr.VarState.Evaluated or state is dr.VarState.Dirty
            is_big_array = dr.itemsize_v(Value) * in_size > 1024 * 1024 * 1024  # 1 GiB
            symbolic = not is_evaluated and is_big_array
    else:
        raise RuntimeError(
            'tensor_reduce(): \'mode\' must be "symbolic", "evaluated", or None.'
        )

    if in_size == out_size:
        # No-op
        out_array = in_array
    elif len(block_strides) == 1 and block_strides[0] == 1 and \
        (dr.backend_v(in_array) is not dr.JitBackend.CUDA or
         block_size & (block_size - 1) == 0):
        # The requested reduction is also doable via dr.block_reduce(), which
        # is going to be more optimized than the other strategies in this file.
        out_array = dr.block_reduce(op, in_array, block_size, mode)
    elif out_size == 1:
        # The requested reduction is also doable via dr.reduce() in 1D, which
        # is going to be more optimized than the other strategies in this file.
        out_array = dr.reduce(op, in_array, 0, mode)
    elif symbolic:
        index = dr.arange(Index, in_size)
        offset = dr.zeros(Index, in_size)
        ctr = 0

        for i, stride_i in enumerate(in_strides):
            pos = index // stride_i

            if i not in axis:
                offset = dr.fma(pos, out_strides_o[ctr], offset)
                ctr += 1

            index -= pos * stride_i

        out_array = dr.detail.reduce_identity(Value, op, out_size)

        dr.scatter_reduce(op, out_array, in_array, offset)
    else:
        index = dr.arange(Index, out_size)
        offset = dr.zeros(Index, out_size)

        for stride_o, stride_i in zip(out_strides_o, out_strides_i):
            if stride_i == stride_o:
                offset += index
                break

            pos = index // stride_o
            offset = dr.fma(pos, stride_i, offset)
            index -= pos * stride_o

        out_array = dr.detail.reduce_identity(Value, op)

        out_array = reduce_recursive(
            op=op,
            value=in_array,
            shape=block_shape,
            strides=block_strides,
            accum=out_array,
            offset=offset,
        )

    return Tensor(out_array, out_shape)
