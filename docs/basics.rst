.. py:currentmodule:: drjit

.. _basics:

Basics
======

This fast-paced section reviews a wide range of standard operations. It assumes
the following import declarations

.. code-block:: python

   import drjit as dr
   from drjit.auto import Float, Array3f, UInt


Creating arrays
---------------

Recall that Dr.Jit array types are dynamically sized--for example,
:py:class:`Float <drjit.auto.Float>` refers to a 1D array of single
precision variables.

The simplest way to create such an array is to call its constructor with
a list of explicit values:

.. code-block:: python

   a = Float(1, 2, 3, 4)
   print(a) # [1, 2, 3, 4]

The constructor also accepts ``Sequence`` types (e.g. lists, tuples, NumPy
arrays, PyTorch tensors, etc.):

.. code-block:: python

   x = Float([1, 2, 3, 4])

Nested array types store several variables---for example, :py:class:`Array3f
<drjit.auto.Array3f>` is just a wrapper around 3 :py:class:`Float
<drjit.auto.Float>` instances. They can be passed to the constructor
explicitly, or via implicit conversion from constants, lists, etc.

.. code-block:: python

   a = Array3f([1, 2], 0, Float(10, 20))
   print(a)
   # Prints (with 'y' component broadcast to full size)
   # [[1, 0, 10],
   #  [2, 0, 20]]

Various functions can also create default-initialized arrays:

- :py:func:`dr.zeros() <zeros>`: ``[0, 0, ...]``.
- :py:func:`dr.ones() <ones>`: ``[1, 1, ...]``.
- :py:func:`dr.full() <full>`: ``[x, x, ...]`` given ``x``.
- :py:func:`dr.arange() <arange>`: ``[0, 1, 2, ...]``.
- :py:func:`dr.linspace() <linspace>`: linear interpolation of two endpoints.
- :py:func:`dr.empty() <empty>`: allocate uninitialized memory.

These always take the desired output type as first argument. You can optionally
request a given size along the dynamic axis, e.g.:

.. code-block:: python

   b = dr.zeros(Array3f)
   print(b.shape) # Prints: (3, 1)

   b = dr.zeros(Array3f, shape=(3, 1000))
   print(b.shape) # Prints: (3, 1000)


Element access
--------------

Use the default ``array[index]`` syntax to read/write array entries. Nested
static 1-4D arrays further expose equivalent ``.x`` / ``.y`` / ``.z`` / ``.w``
members:

.. code-block:: python

   a = Array3f(1, 2, 3)
   a.x += a.z + a[1]

Static 1-4D arrays also support `swizzling
<https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)>`__, which
arbitrarily reorders elements:

.. code-block:: python

   a.xy = a.xx + a.yx

Arithmetic operations
---------------------

Except for a few special cases (e.g., matrix multiplication), arithmetic
operations transform arrays element-wise. If needed, the system will implicitly
broadcast the operands and promote types.

.. code-block:: pycon

   >>> a = abs(Float(-1.25, 2) + UInt32(1))
   >>> type(a)
   <class 'drjit.cuda.Float'>
   >>> a
   [0.25, 3]

In the above example, *broadcasting* automatically extended the size of the
*scalar* (size-1) array, and the :py:class:`UInt32 <drjit.cuda.UInt>` type was
*promoted* to :py:class:`Float <drjit.cuda.Float>`. Type promotion follows the
rules of the C programming language.

Besides built-in arithmetic operators, the following standard functions are
available:

- :py:func:`dr.abs() <abs>`: Absolute value.
- :py:func:`dr.fma() <fma>`: Fused multiply-add.
- :py:func:`dr.minimum() <minimum>`, :py:func:`dr.maximum() <maximum>`:
  Element-wise minimum/maximum of two arrays.
- :py:func:`dr.ceil() <ceil>`, :py:func:`dr.floor() <floor>`,
  :py:func:`dr.round() <ceil>`, :py:func:`dr.trunc() <floor>`: Round up, down,
  to nearest, or to zero.
- :py:func:`dr.sqrt() <sqrt>`, :py:func:`dr.cbrt() <cbrt>`: Square and cube
  root.
- :py:func:`dr.rcp() <rcp>`: Reciprocal.
- :py:func:`dr.rsqrt() <rsqrt>`: Reciprocal square root.
- :py:func:`dr.sign() <sign>`: Extract the sign.
- :py:func:`dr.copysign() <copysign>`: Copy sign from one value to another.
- :py:func:`dr.clip() <clip>`: Clip a value to an interval.
- :py:func:`dr.lerp() <lerp>`: Linearly interpolate.

The library implements common transcendental functions:

- :py:func:`dr.sin() <sin>`, :py:func:`dr.cos() <cos>`, :py:func:`dr.tan()
  <tan>`: Trigonometric functions.
- :py:func:`dr.asin() <asin>`, :py:func:`dr.acos() <acos>`, :py:func:`dr.atan()
  <atan>`, :py:func:`dr.atan2() <atan2>`: .. and their inverses.
- :py:func:`dr.sinh() <sinh>`, :py:func:`dr.cosh() <cos>`, :py:func:`dr.tanh()
  <tanh>`: Hyperbolic trigonometric functions.
- :py:func:`dr.asinh() <asinh>`, :py:func:`dr.acosh() <acosh>`,
  :py:func:`dr.atanh() <atanh>`: .. and their inverses.
- :py:func:`dr.sincos() <sincos>`, :py:func:`dr.sincosh() <sincosh>`: Fast
  combined evaluation.
- :py:func:`dr.erf() <erf>`, :py:func:`dr.erfinv() <erfinv>`: Error function.
- :py:func:`dr.exp() <exp>`, :py:func:`dr.log() <log>`, :py:func:`dr.exp2()
  <exp2>`, :py:func:`dr.log2() <log2>`: Exponentials and logarithms.
- :py:func:`dr.power() <power>`: Power function.
- :py:func:`dr.lgamma() <lgamma>`: Gamma function.

Most of these support real and complex-valued inputs. A subset accepts
quaternions (see the section on :ref:`array types <special_arrays>` for
details). Integer arrays further support the following bit-level operations


- :py:func:`dr.lzcnt() <lzcnt>`, :py:func:`dr.tzcnt() <tzcnt>`:
  Leading/trailing zero count.
- :py:func:`dr.popcnt() <popcnt>`: Population count.
- :py:func:`dr.brev() <brev>`: Bit reverse.

Mask operations
---------------

Equality and inequality comparisons produce *masks* (i.e., boolean-valued
arrays) with support for binary arithmetic. The :py:func:`dr.select() <select>`
function blends results from two arrays based on a mask analogous to the
ternary ("``?``") operator in C/C++.

.. code-block:: pycon

   >>> a = dr.arange(Float, 5) - 3
   >>> mask = (a < 0) | (a == 2)
   >>> mask
   [True, True, False, False, True]
   >>> dr.select(mask, -1, a)        # select(mask, true_value, false_value)
   [-1, -1, 0, 1, -1]

Masks can also be applied to arrays in order to zero out the `False` indices by
using the `&` operator.

.. code-block:: pycon

   >>> a = Float([1, 2, 3])
   >>> mask = Bool([True, False, True])
   >>> a & mask
   [1, 0, 3]

Reductions
----------

Reductions use a given operation (e.g., addition) to combine values along one
or several dimensions.

- :py:func:`dr.sum() <sum>`, :py:func:`dr.prod() <prod>`: Sum and product reduction.
- :py:func:`dr.min() <min>`, :py:func:`dr.max() <max>`: Minimum/maximum reduction.
- :py:func:`dr.all() <all>`, :py:func:`dr.any() <any>`, :py:func:`dr.none()
  <none>`: Boolean reductions for mask arrays.
- :py:func:`dr.reduce() <reduce>`: Generalized reduction operator.

By default, they reduce arrays along the leading array dimension. For example,
the following reduction is equivalent to ``a.x + a.y + a.z``. By reducing this
value once more or specifying `axis=None`, we can sum over all entries.

.. code-block:: pycon

   >>> a = Array3f([1, 2], [10, 20], [100, 200])
   >>> dr.sum(a)
   [111, 222]
   >>> dr.sum(a, axis=None)
   [333]

Accessing memory: gather/scatter
--------------------------------

The function :py:func:`dr.gather() <gather>` fetches values from a 1D array
with positions specified by an index array. For example:

.. code-block:: pycon

   >>> buf = Float(10, 20, 30, 40, 50, 60)
   >>> index = UInt32(1, 0)
   >>> dr.gather(Float, buf, index)
   [20, 10]

Note how the operation takes the desired output type as first argument. We can
also gather nested arrays (assumed to be flattened in the source 1D
array using C-style order) by requesting a different result type.

.. code-block:: pycon

   >>> dr.gather(Array3f, buf, index)
   [[40, 50, 60],
    [10, 20, 30]]

Whereas gather reads memory, :py:func:`dr.scatter() <scatter>` realizes the
corresponding write operation.

.. code-block:: pycon

   >>> dr.scatter(buf, Array3f(0, 1, 2), UInt32(1))
   >>> buf
   [10, 20, 30, 0, 1, 2]

Finally, :py:func:`dr.scatter_add() <scatter_add>` (and the more
general :py:func:`dr.scatter_reduce() <scatter_reduce>`) atomically
accumulates values into an array.

.. code-block:: pycon

   >>> dr.scatter_add(buf, Array3f(100), UInt32(1))
   >>> buf
   [10, 20, 30, 100, 101, 102]

Random number generation
------------------------

Dr.Jit was originally developed for Monte Carlo simulation, and programs in
that domain require a source of (pseudo-) randomness. For this, the system
provides a member of the `PCG Family
<https://www.pcg-random.org/index.html>`__ of random number generators by
`Melissa O'Neill <https://www.cs.hmc.edu/~oneill/index.html>`__. To try it,
import the class :py:class:`drjit.*.PCG32 <drjit.auto.PCG32>` from your backend
of choice and initialize with the desired output array size.

.. code-block:: pycon

   >>> from drjit.auto import PCG32
   >>> rng = PCG32(10000)
   >>> rng.next_float32()
   [0.108379, 0.533909, 0.00684452, .. 9994 skipped .., 0.511698, 0.600626, 0.219648]
   >>> rng.next_uint32_bounded(4)
   [1, 0, 0, .. 9994 skipped .., 0, 3, 3]

Please see the documentation of this class for a review of its features.
