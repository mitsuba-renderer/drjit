import drjit as dr
import sys
import typing

ArrayT = typing.TypeVar('ArrayT', bound=dr.ArrayBase)
Shape = typing.Union[int, typing.Tuple[int, ...]]
ArrayOrInt = typing.Union[int, dr.ArrayBase]
ArrayOrFloat = typing.Union[float, dr.ArrayBase]

class Generator:
    def random(self, dtype: typing.Type[ArrayT], shape: Shape) -> ArrayT:
        """
        Return a Dr.Jit array or tensor containing uniformly distributed
        pseudorandom variates.

        This function supports floating point arrays/tensors of various
        configurations and precisions, e.g.:

        .. code-block:: python

           from drjit.cuda import Float, TensorXf, Array3f, Matrix4f

           # Example usage
           rng = dr.rng(seed=0)
           rand_array = rng.random(Float, 128)
           rand_tensor = rng.random(TensorXf16, shape=(128, 128))
           rand_vec = rng.random(Array3f, (3, 128))
           rand_mat = rng.random(Matrix4f64, (4, 4, 128))

        The output is uniformly distributed the half-open interval :math:`[0, 1)`.
        Integer arrays are not supported.

        Args:
            source (type[ArrayT]): A Dr.Jit tensor or array type.

            shape (int | tuple[int, ...]): The target shape

        Returns:
            ArrayT: The generated array of random variates.
        """
        raise NotImplementedError("random(): use a subclass that implements this function")

    def uniform(self, dtype: typing.Type[ArrayT], shape: Shape, low: ArrayOrFloat = 0.0, high: ArrayOrFloat = 1.0):
        """
        Return a Dr.Jit array or tensor containing uniformly distributed
        pseudorandom variates.

        This function resembles :py:func:`random()` but additionally ensures
        that variates are distributed on the half-open interval
        :math:`[\texttt{low}, \texttt{high})`.

        Args:
            source (type[ArrayT]): A Dr.Jit tensor or array type.

            shape (int | tuple[int, ...]): The target shape

            low (float | drjit.ArrayBase): The low value of the desired interval

            high (float | drjit.ArrayBase): The high value of the desired interval

        Returns:
            ArrayT: The generated array of random variates.
        """

        return dr.fma(self.random(dtype, shape), high-low, low)

    def integers(self, dtype: typing.Type[ArrayT], shape: Shape, low: ArrayOrInt = 0, high: typing.Optional[ArrayOrInt] = None, endpoint: bool = False) -> ArrayT:
        """
        Return a Dr.Jit array or tensor containing uniformly distributed
        pseudorandom 32-bit integers.

        If ``high`` is ``None``, integers are drawn from ``[0, low)``.
        Otherwise, integers are drawn from ``[low, high)``, or ``[low, high]``
        if ``endpoint=True``.

        Args:
            dtype (type[ArrayT]): A Dr.Jit 32-bit integer array type
                (``Int32`` or ``UInt32``).

            shape (int | tuple[int, ...]): The target shape

            low (int | drjit.ArrayBase): Lowest integer to be drawn (inclusive).
                If ``high`` is ``None``, this parameter specifies the exclusive
                upper bound, and 0 becomes the lower bound.

            high (int | drjit.ArrayBase | None): If provided, one above the
                largest integer to be drawn. If ``endpoint=True``, this is the
                largest integer to be drawn (inclusive).

            endpoint (bool): If ``True``, ``high`` is inclusive. Default: ``False``.

        Returns:
            ArrayT: The generated array of random integers.
        """
        raise NotImplementedError("integers(): use a subclass that implements this function")

    def normal(self, dtype: typing.Type[ArrayT], shape: Shape, loc: ArrayOrFloat = 0.0, scale: ArrayOrFloat = 1.0) -> ArrayT:
        """
        Return a Dr.Jit array or tensor containing pseudorandom variates
        following a standard normal distribution

        This function supports arrays/tensors of various configurations and
        precisions--see the similar :py:func:`drjit.random()` for examples on
        how to call this function.

        Args:
            source (type[ArrayT]): A Dr.Jit tensor or array type.

            shape (int | tuple[int, ...]): The target shape

            loc (float | drjit.ArrayBase): The mean of the normal distribution (``0.0`` by default)

            scale (float | drjit.ArrayBase): The standard deviation of the normal distribution (``1.0`` by default)

        Returns:
            ArrayT: The generated array of random variates.
        """
        raise NotImplementedError("normal(): use a subclass that implements this function")

    def clone(self) -> 'Generator':
        raise NotImplementedError("clone(): use a subclass that implements this function")

class Philox4x32Generator(Generator):
    """
    Implementation of the :py:class:`Generator` interface based on the Philox4x32 RNG.
    """

    _seed : ArrayOrInt
    _counter : ArrayOrInt

    DRJIT_STRUCT = { '_seed' : ArrayOrInt, '_counter' : ArrayOrInt }

    def __init__(self, seed: ArrayOrInt = 0, counter: ArrayOrInt = 0, symbolic: bool = False):
        seed_tp = dr.uint64_array_t(seed)
        counter_tp = dr.uint32_array_t(seed)
        self._seed = seed_tp(seed)
        self._counter = counter_tp(counter)
        self._symbolic = symbolic

    def clone(self) -> 'Generator':
        return Philox4x32Generator(self._seed, self._counter)

    def _sample(self, dtype: typing.Type[ArrayT], shape: Shape, fn_name: str) -> ArrayT:
        counter, seed = self._counter, self._seed

        if isinstance(shape, int):
            shape = (shape, )

        is_jit = dr.is_jit_v(dtype)
        is_tensor = dr.is_tensor_v(dtype)

        if is_jit:
            # When JIT-compiling random number generation, ensure that
            # seed/counter have the right type and are opaque.

            leaf_tp = dr.leaf_t(dtype)
            mod = sys.modules[leaf_tp.__module__]
            seed_tp = dr.uint64_array_t(leaf_tp)
            counter_tp = dr.uint32_array_t(leaf_tp)

            symbolic = self._symbolic or dr.flag(dr.JitFlag.SymbolicScope)

            if type(seed) is not seed_tp:
                if symbolic:
                    raise RuntimeError('To generate random numbers within a symbolic loop, you must initialize the Generator with a seed of the underlying JIT backend, e.g.: rng = dr.rng(seed=dr.cuda.UInt32(0))')
                # When we record a frozen function, this will result in a
                # change of a type in the input of the function. The recording
                # were this change occurred cannot be replayed and has to be
                # discarded.
                dr.detail.freeze_discard(dr.backend_v(seed_tp), "Philox4x32Generator allocated seed")

                seed = seed_tp(seed)
                counter = counter_tp(counter)

            if not symbolic:
                dr.make_opaque(counter, seed)

            # Compute the number of lanes for parallel generation
            if is_tensor:
                lane_count = dr.prod(shape)
            else:
                lane_count = shape[-1]

            lane_index = dr.arange(counter_tp, lane_count)
        else:
            mod = dr.scalar
            lane_index = 0

        # Philox4x32 generates multiple values at once. Put them
        # into a pool and pop from this pool as needed
        pool = []
        def next_sample():
            nonlocal counter, pool

            # When the pool is empty, generate the next batch
            if not pool:
                rng = mod.Philox4x32(seed, counter, lane_index)
                pool += list(getattr(rng, fn_name)())
                counter += 1

            return pool.pop()

        if (is_jit and dr.depth_v(dtype) <= 1) or not dr.is_array_v(dtype):
            # Simple case: scalars and JIT-compiled 1D arrays/tensors
            s = next_sample()
            if is_tensor:
                value = dtype(s, shape)
            else:
                if len(shape) > 1:
                    raise RuntimeError('could not construct output: the provided "shape" and "dtype" parameters are incompatible.')
                value = dtype(s)
        else:
            # Complex case: vectors, matrices, non-JIT arrays, etc.
            def fill(v):
                if dr.depth_v(v) == 1:
                    if is_jit:
                        v[:] = next_sample()
                    else:
                        for i in range(len(v)):
                            v[i] = next_sample()
                else:
                    for i in range(len(v)):
                        fill(v[i])

            value = dr.empty(dtype, shape)
            fill(value)


        self._seed, self._counter = seed, counter

        return value

    def random(self, dtype: typing.Type[ArrayT], shape: Shape) -> ArrayT:
        tp = dr.type_v(dtype)
        if tp == dr.VarType.Float16:
            fn_name = 'next_float16x4'
        elif tp == dr.VarType.Float32:
            fn_name = 'next_float32x4'
        elif tp == dr.VarType.Float64 or dtype is float:
            fn_name = 'next_float64x2'
        else:
            raise RuntimeError('Unsupported "dtype": must be a Dr.Jit float16/float32/float64 array.')
        return self._sample(dtype, shape, fn_name)

    def integers(self, dtype: typing.Type[ArrayT], shape: Shape, low: ArrayOrInt = 0, high: typing.Optional[ArrayOrInt] = None, endpoint: bool = False) -> ArrayT:
        if high is None:
            high = low
            low = 0

        if endpoint:
            high = high + 1

        tp = dr.type_v(dtype)
        if tp not in (dr.VarType.Int32, dr.VarType.UInt32):
            raise RuntimeError('Unsupported "dtype": must be a Dr.Jit int32/uint32 array or tensor.')

        if isinstance(shape, int):
            shape = (shape,)

        UInt32 = dr.uint32_array_t(dtype)
        UInt64 = dr.uint64_array_t(dtype)
        bound = UInt32(high - low)
        threshold = (-bound) % bound

        x = self._sample(UInt32, shape, 'next_uint32x4')
        m = UInt64(x) * UInt64(bound)

        def loop_cond(m):
            return UInt32(m) < threshold

        def loop_body(m):
            x = self._sample(UInt32, shape, 'next_uint32x4')
            return (UInt64(x) * UInt64(bound),)

        (m,) = dr.while_loop(
            state=(m,),
            cond=loop_cond,
            body=loop_body
        )

        return dtype(m >> 32) + low

    def normal(self, dtype: typing.Type[ArrayT], shape: Shape, loc: ArrayOrFloat = 0.0, scale: ArrayOrFloat = 1.0) -> ArrayT:
        tp = dr.type_v(dtype)
        if tp == dr.VarType.Float16:
            fn_name = 'next_float16x4_normal'
        elif tp == dr.VarType.Float32:
            fn_name = 'next_float32x4_normal'
        elif tp == dr.VarType.Float64 or dtype is float:
            fn_name = 'next_float64x2_normal'
        else:
            raise RuntimeError('Unsupported "dtype": must be a Dr.Jit float16/float32/float64 array.')
        return dr.fma(self._sample(dtype, shape, fn_name), scale, loc)

    def __repr__(self) -> str:
        seed = self._seed if isinstance(self._seed, int) else self._seed[0]
        counter = self._counter if isinstance(self._counter, int) else self._counter[0]
        return f'Philox4x32Generator[seed={seed}, counter={counter}]'

