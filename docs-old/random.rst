.. _random:
.. cpp:namespace:: drjit

Random number generation
========================

Dr.Jit includes a fully vectorized implementation of the `PCG32 pseudorandom
number generator <http://www.pcg-random.org/>`_ developed by `Melissa O'Neill
<https://www.cs.hmc.edu/~oneill>`_. To use it, include the following header:

.. code-block:: cpp

    #include <drjit/random.h>

The following reference is partly based on the original `PCG32 documentation
<http://www.pcg-random.org/using-pcg-c.html>`_.

Usage
-----

The :cpp:struct:`PCG32` class is parameterized by a template parameter ``T``
that denotes the desired output type. Scalar values, SIMD packets, and
CUDA/LLVM arrays are all supported. The specific scalar type underlying ``T``
(e.g. integral, floating point, etc.) is ignored here, since only the shape
of the output is relevant at this point.

Using a scalar type like ``uint32_t`` leads to an implementation that generates
scalar variates:

.. code-block:: cpp

    // Scalar RNG
    using RNG = PCG32<uint32_t>;

    RNG my_rng;
    float value_f    = my_rng.next_float32(); // uniform on the interval [0, 1)
    uint32_t value_u = my_rng.next_uint32();  // uniform integer on [0, 2^32 - 1]

Specifying an Dr.Jit array type leads to a vectorized random number generator
that generates arrays of pseudorandom variates in parallel.

.. code-block:: cpp

    using Float = Packet<float, 16>;

    // Vector RNG -- generates 16 independent variates at once
    using RNG = PCG32<Float>;

    RNG my_rng;
    Float value = my_rng.next_float32();

PCG32 is *fast*: on a Skylake i7-6920HQ processor, the vectorized
implementation provided here generates around 1.4 billion single precision
variates per second. Note that the implementation uses a small amount of
internal storage to record the current RNG state and a stream selector.
Together, these two fields require 16 bytes per SIMD lane.

When used with LLVM or CUDA arrays, the arithmetic underlying pseudorandom
number generation will be JIT-compiled along with other computation performed
by the caller. Differentiable array types, e.g.:

.. code-block:: cpp

    using RNG = PCG32<DiffArray<LLVMArray<float>>>;

are supported but do not have an influence on the behavior of this class, since
the generated variates are the result of integral (i.e. non-differentiable)
computation. This can still be useful when a differentiable computation is
expressed using such types, in which case no special precautions must be taken
for the random number generator.

Python
------

Bindings of PCG32 are provided for all backends, see the discussion on
:ref:`Python types <python-types>` with regards to package naming conventions.

.. code-block:: pycon

    >>> from drjit.llvm import PCG32

    >>> rng = PCG32(size=100)

    >>> rng.next_float32()
    [0.10837864875793457, 0.15841352939605713, 0.9734833240509033, 0.006844520568847656, 0.05747580528259277, .. 90 skipped .., 0.209586501121521, 0.3716639280319214, 0.8550137281417847, 0.30228495597839355, 0.21239590644836426]

C++ Reference
-------------

.. cpp:struct:: template <typename T> PCG32

    This class implements the PCG32 random number generator. It has a period of
    :math:`2^{64}` and supports :math:`2^{63}` separate *streams*. Each stream
    produces a different unique sequence of pseudorandom numbers, which is
    particularly useful in the context of vectorized computations.

Member types
************

.. cpp:namespace:: template <typename T> drjit::PCG32

.. cpp:type:: Int64 = int64_array_t<T>

    Type alias for a signed 64-bit integer (or an array thereof).

.. cpp:type:: UInt64 = uint64_array_t<T>

    Type alias for a unsigned 64-bit integer (or an array thereof).

.. cpp:type:: UInt32 = uint32_array_t<T>

    Type alias for a unsigned 32-bit integer (or an array thereof).

.. cpp:type:: Float32 = float32_array_t<T>

    Type alias for a single precision float (or an array thereof).

.. cpp:type:: Float64 = float64_array_t<T>

    Type alias for a double precision float (or an array thereof).

.. cpp:type:: Mask = mask_t<UInt64>

    Type alias for masks that are internally used

Member variables
****************

.. cpp:member:: UInt64 state

    Stores the RNG state.  All values are possible.

.. cpp:member:: UInt64 inc

    Controls which RNG sequence (stream) is selected. Must *always* be odd,
    which is ensured by the constructor and :cpp:func:`seed()` method.

Constructors
************

.. cpp:function:: PCG32(size_t = 1, \
                        const UInt64 &initstate = PCG32_DEFAULT_STATE, \
                        const UInt64 &initseq   = PCG32_DEFAULT_STREAM)

     Seeds the PCG32 with the default state using the :cpp:func:`seed()`
     method.

Methods
*******

.. cpp:function:: void seed(size_t = 1, \
                            const UInt64 &initstate = PCG32_DEFAULT_STATE, \
                            const UInt64 &initseq = PCG32_DEFAULT_STREAM)

    This function initializes (a.k.a. "seeds") the random number generator, a
    required initialization step before the generator can be used. The provided
    arguments are defined as follows:

    - ``size`` denotes the number of parallel instances of random number
      generators that should be instantiated. This value is only relevant
      when ``T`` is a dynamic array type, in which case an appropriate
      offset is added to ``initseq`` for every entry.

    - ``initstate`` is the starting state for the RNG. Any 64-bit value is
      permissible.

    - ``initseq`` selects the output sequence for the RNG. Any 64-bit value is
      permissible, although only the low 63 bits are used.

    For this generator, there are :math:`2^{63}` possible sequences of
    pseudorandom numbers. Each sequence is entirely distinct and has a period
    of :math:`2^{64}`. The ``initseq`` argument selects which stream is used.
    The ``initstate`` argument specifies the location within the :math:`2^{64}`
    period.

    Calling :cpp:func:`PCG32::seed` with the same arguments produces the same
    output, allowing programs to use random number sequences repeatably.

.. cpp:function:: UInt32 next_uint32(const Mask &mask = true)

    Generate a uniformly distributed unsigned 32-bit random number (i.e.
    :math:`x`, where :math:`0\le x< 2^{32}`)

    If a mask parameter is provided, only the pseudorandom number generators
    of active SIMD lanes are advanced.

.. cpp:function:: UInt64 next_uint64(const Mask &mask = true)

    Generate a uniformly distributed unsigned 64-bit random number (i.e.
    :math:`x`, where :math:`0\le x< 2^{64}`)

    If a mask parameter is provided, only the pseudorandom number generators
    of active SIMD lanes are advanced.

    .. note::

        This function performs two internal calls to :cpp:func:`next_uint32()`.

.. cpp:function:: UInt32 next_uint32_bound(uint32_t bound, const Mask &mask = true)

    Generate a uniformly distributed unsigned 32-bit random number less
    than ``bound`` (i.e. :math:`x`, where :math:`0\le x<` ``bound``)

    If a mask parameter is provided, only the pseudorandom number generators
    of active SIMD lanes are advanced.

    .. note::

        This may involve multiple internal calls to
        :cpp:func:`next_uint32()`, in which case the RNG advances by
        several steps. This is only relevant when using the
        :cpp:func:`advance()` or :cpp:func:`operator-()` method.

.. cpp:function:: UInt64 next_uint64_bound(uint64_t bound, const Mask &mask = true)

    Generate a uniformly distributed unsigned 64-bit random number less
    than ``bound`` (i.e. :math:`x`, where :math:`0\le x<` ``bound``)

    If a mask parameter is provided, only the pseudorandom number generators of
    active SIMD lanes are advanced.

    .. note::

        This may involve multiple internal calls to
        :cpp:func:`next_uint64()`, in which case the RNG advances by
        several steps. This is only relevant when using the
        :cpp:func:`advance()` or :cpp:func:`operator-()` method.

.. cpp:function:: Float32 next_float32(const Mask &mask = true)

    Generate a single precision floating point value on the interval :math:`[0, 1)`

    If a mask parameter is provided, only the pseudorandom number generators of
    active SIMD lanes are advanced.

.. cpp:function:: Float64 next_float64(const Mask &mask = true)

    Generate a double precision floating point value on the interval :math:`[0, 1)`

    If a mask parameter is provided, only the pseudorandom number generators of
    active SIMD lanes are advanced.

    .. warning::

        Since the underlying random number generator produces 32 bit
        output, only the first 32 mantissa bits will be filled (however,
        the resolution is still finer than in :cpp:func:`next_float32`,
        which only uses 23 mantissa bits)

.. cpp:function:: void advance(const Int64 &delta)

    This operation provides jump-ahead; it advances the RNG by ``delta`` steps,
    doing so in :math:`\log(\texttt{delta})` time. Because of the periodic
    nature of generation, advancing by :math:`2^{64}-d` (i.e., passing
    :math:`-d`) is equivalent to backstepping the generator by :math:`d` steps.

.. cpp:function:: Int64 operator-(const PCG32 &other)

    Compute the distance between two PCG32 pseudorandom number generators

.. cpp:function:: bool operator==(const PCG32 &other)

    Equality operator

.. cpp:function:: bool operator!=(const PCG32 &other)

    Inequality operator

Macros
******

The following macros are defined in :file:`drjit/random.h`:

.. cpp:var:: uint64_t PCG32_DEFAULT_STATE = 0x853c49e6748fea9bULL

    Default initialization passed to :cpp:func:`PCG32::seed`.

.. cpp:var:: uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL

    Default stream index passed to :cpp:func:`PCG32::seed`.
