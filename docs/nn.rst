.. py:currentmodule:: drjit.nn

.. _neural_nets:

Neural Networks
---------------

The module :py:mod:`drjit.nn` provides convenient modular abstractions to
construct, evaluate,  and optimize neural networks. Their design resembles the
PyTorch `nn.Module
<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__ classes.

The :py:class:`nn.Module <Module>` class takes a cooperative vector as input
and produces another cooperative vector. Modules can be chained to form longer
sequential pipelines. Please review the section on :ref:`cooperative vectors
<coop_vec>` for details on cooperative vectors and their applications.

.. warning::

   The neural network classes are considered experimental and subject to change
   in future versions of Dr.Jit.

Example:
--------

Below is an example demonstrating how to use it to declare and optimize a small
`multilayer perceptron <https://en.wikipedia.org/wiki/Multilayer_perceptron>`__
(MLP).

.. code-block:: python

    import drjit as dr
    import drjit.nn as nn

    from drjit.llvm.ad import TensorXf16
    from drjit.opt import Adam

    # Establish the network structure
    net = nn.Sequential(
        nn.Linear(-1, 32, bias=False),
        nn.ReLU(),
        nn.Linear(-1, -1),
        nn.ReLU(),
        nn.Linear(-1, 3, bias=False),
        nn.Tanh()
    )

    # Instantiate the network for a specific backend + input size
    net = net.alloc(TensorXf16, 2)

    # Pack coefficients into a training-optimal layout
    coeffs, net = nn.pack(net, layout='training')

    # Optimize a float32 version of the packed coefficients
    opt = Adam(lr=1e-3, params={'coeffs': Float32(coeffs)})

    # Update network state from optimizer
    for i in range(1000):
        # Update neural network state
        coeffs[:] = Float16(opt['coeffs'])

        # Create input
        out = net(nn.CoopVec(...))

        # Unpack
        out = Array3f16(result)

        # Backpropagate
        dr.backward(dr.square(reference-out))

        # Take a gradient step
        opt.step()

List
----

Neural network module classes currently include:

- Sequential evaluation of a list of models: :py:class:`Sequential`.

- Linear/affine layers: :py:class:`Linear`.

- Encoding layers: :py:class:`SinEncode`, :py:class:`TriEncode`.

- Activation functions and other nonlinear transformations: :py:class:`ReLU`, :py:class:`LeakyReLU`,
  :py:class:`Exp`, :py:class:`Exp2`, :py:class:`Tanh`.

- Miscellaneous: :py:class:`Cast`, :py:class:`ScaleAdd`.
