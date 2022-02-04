.. cpp:namespace:: drjit
.. py:module:: drjit
.. _autodiff:

Automatic differentiation
=========================

*Automatic differentiation* (AD) refers to a set of techniques to numerically
evaluate the gradient of a computer program. Dr.Jit realizes such differentiable
computations using the :cpp:struct:`DiffArray` C++ template class and Python
bindings in the :py:mod:`drjit.cuda.ad` and :py:mod:`drjit.llvm.ad` modules.

The remainder of this section assumes the following import declarations:

.. tabs::

    .. code-tab:: python

        # Differentiable float type, which performs computations on the GPU
        from drjit.cuda.ad import Float

        # ... alternatively, the LLVM backend performs computations on the CPU
        # from drjit.llvm.ad import Float

    .. code-tab:: cpp

        #include <drjit/autodiff.h>

        // Declare a base type that performs computation on the GPU
        using BaseFloat = dr::CUDAArray<float>;

        // Alternatively, the following base type performs computation on the CPU
        // using BaseFloat = dr::LLVMArray<float>;

        // The differentiable array wraps the base type
        using Float = dr::DiffArray<BaseFloat>;


Basics
------

AD is based on a remarkably simple idea: no matter how big and complex a
computer program becomes, it will ultimately be built from operations that are
individually simple to understand, like additions and multiplications. Each of
these smaller operations is easily differentiated on its own, at which point
the `chain rule <https://en.wikipedia.org/wiki/Chain_rule>`_ can be used to
assemble the individual derivatives into a derivative of the entire program.
This can be done at a moderate extra cost (usually no more than 3-4x the cost
of the original computation). Many flavors and specializations of AD have been
proposed in the past, we refer to the excellent book by Griewank and Walther
[GrWa08]_ for a thorough overview of this topic.

Dr.Jit implements two symmetrically opposite modes of AD:

1. **Forward mode**. Given computation with certain inputs and outputs, forward
   mode propagates derivatives *from the inputs towards the outputs*.

   This mode is most efficient when the computation to be differentiated has
   few inputs (ideally only a single one). Otherwise, multiple AD passes may be
   needed to deal with each input separately; the section on :ref:`tangent
   mode <autodiff-tangent-adjoint-mode>` provides details on this.

   For example, suppose we want to differentiate both outputs of the function
   :math:`\mathrm{sincos}(\theta)=(\sin\theta, \cos\theta)` with respect to a
   single input :math:`\theta` at :math:`\theta=1`. Using Dr.Jit, this can be
   done as follows:

   .. tabs::

       .. code-tab:: python

           theta = Float(1.0)       # Declare input value and enable gradient tracking
           dr.enable_grad(theta)
           s, c = dr.sincos(theta)  # The computation to be differentiated
           dr.forward(theta)        # Forward-propagate derivatives through the computation
           grad_s, grad_c = dr.grad(s), dr.grad(c) # Extract derivatives wrt. both outputs

       .. code-tab:: cpp

           Float theta = 1.f;               // Declare input value and enable gradient tracking
           dr::enable_grad(theta);
           auto [s, c] = dr::sincos(theta); // The computation to be differentiated
           dr::forward(theta);              // Forward-propagate derivates through the computation
           BaseFloat grad_s = dr::grad(s),  // Extract derivatives wrt. both outputs
                     grad_c = dr::grad(c);

   Forward-mode AD simultaneously computes derivatives with respect to all
   outputs. There were only two in this example (``grad_s`` and ``grad_c``)
   though the approach remains efficient even when there are vast numbers of
   them.

2. **Reverse/Backward mode**. In contrast, reverse mode propagates derivatives *from the
   outputs towards the inputs*.

   This mode is most efficient when the computation to be differentiated has
   few outputs (ideally only a single one). Otherwise, multiple AD passes may be
   needed to deal with each input separately; the section on :ref:`adjoint
   mode <autodiff-tangent-adjoint-mode>` provides details on this.

   For example, suppose we want to differentiate both inputs of the function
   :math:`\theta=\mathrm{atan2}(y, x)` with respect to a single output
   :math:`\theta`. Using Dr.Jit, this can be done as follows:

   .. tabs::

       .. code-tab:: python

           from drjit.cuda.ad import Array2f # Array composed of 2 differentiable Floats
           p = Array2f(1, 2) # Declare the input value and enable gradient tracking
           dr.enable_grad(p)
           theta = dr.atan2(p.y, p.x) # The computation to be differentiated
           dr.backward(theta) # Reverse-propagate derivatives through the computation
           grad_x, grad_y = dr.grad(p.x), dr.grad(p.y) # Extract derivatives wrt. both inputs

       .. code-tab:: cpp

           using Array2f = dr::Array<Float, 2>; // Array composed of 2 differentiable Floats
           Array2f p = Array2f(1, 2); // Declare the input value and enable gradient tracking
           dr::enable_grad(p);
           Float theta = dr::sincos(p.y(), p.x()); // The computation to be differentiated
           dr::backward(theta); // Reverse-propagate derivates through the computation
           BaseFloat grad_x = dr::grad(p.x()), // Extract derivatives wrt. both inputs
                     grad_y = dr::grad(p.y());

   Reverse-mode AD simultaneously computes derivatives with respect to all
   inputs and remains efficient even when there are vast numbers of them.

   Reverse mode is particuarly useful for gradient-based optimization
   techniques, where one often encounters functions with many input parameters
   and a single output ("loss") that characterizes the quality of the current
   solution.

API reference (Python)
----------------------

.. py:function:: enable_grad(*args)

   Recurses through :ref:`pytrees <pytrees>`.

   :param args: List of variables for which gradient tracking should be
                enabled. Recurses through :ref:`pytrees <pytrees>`.

.. py:function:: disable_grad(*args)

   Recurses through :ref:`pytrees <pytrees>`.

   :param args: List of variables for which gradient tracking should be
                enabled.



.. _pytrees:
Functions that operate on general python object trees
-----------------------------------------------------

Many Dr.Jit operations support arbitrary
(tuples, lists, dicts, custom data
structures).


How does it work?
-----------------

Dr.Jit

There is one major catch with reverse mode: derivative propagation proceeds in
the opposite direction (i.e. ) A partial record of intermediate computations
must be kept in memory to enable this, which can become costly for long-running
computations.

.. tabs::

   .. code-tab:: python

       dr::graphviz(f);
       from drjit.cuda.ad import Array2f # Array composed of 2 differentiable Floats
       p = Array2f(1, 2) # Declare the input value and enable gradient tracking

   .. code-tab:: cpp

       using Array2f = dr::Array<Float, 2>; // Array composed of 2 differentiable Floats
       Array2f p = Array2f(1, 2); // Declare the input value and enable gradient tracking
       dr::enable_grad(p);

Visualizing computation graphs
------------------------------

It is possible to visualize the graph of the currently active computation using
the :cpp:func:`graphviz` function. You may also want to assign explicit
variable names via  :cpp:func:`set_label` to make the visualization easier to
parse. An example is shown below:

.. code-block:: python

    >>> a = FloatD(1.0)
    >>> set_requires_gradient(a)
    >>> b = erf(a)
    >>> set_label(a, 'a')
    >>> set_label(b, 'b')

    >>> print(graphviz(b))
    digraph {
      rankdir=RL;
      fontname=Consolas;
      node [shape=record fontname=Consolas];
      1 [label="'a' [s]\n#1 [E/I: 1/5]" fillcolor=salmon style=filled];
      3 [label="mul [s]\n#3 [E/I: 0/4]"];
      ... 111 lines skipped ...
      46 -> 12;
      46 [fillcolor=cornflowerblue style=filled];
    }

The resulting string can be visualized via Graphviz, which reveals the
numerical approximation used to evaluate the error function :cpp:func:`erf`.

.. figure:: autodiff-01.svg
    :width: 800px
    :align: center

.. _autodiff-tangent-adjoint-mode:
Tangent and adjoint mode
------------------------

Advanced usage
--------------

.. rubric:: References

.. [GrWa08] Andreas Griewank and Andrea Walther. 2008. Evaluating derivatives: principles and techniques of algorithmic differentiation. Vol. 105. SIAM.


