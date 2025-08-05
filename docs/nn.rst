.. py:currentmodule:: drjit.nn

.. _neural_nets:

Neural Networks
===============

Dr.Jit's neural network infrastructure builds on :ref:`cooperative vectors
<coop_vec>`. Please review their documentation before reading this section.

The module :py:mod:`drjit.nn` provides convenient modular abstractions to
construct, evaluate,  and optimize neural networks. Their design resembles the
PyTorch `nn.Module
<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__ classes.
The Dr.Jit :py:class:`nn.Module <Module>` class takes a cooperative vector as input
and produces another cooperative vector. Modules can be chained to form longer
sequential pipelines.

.. warning::

   The neural network classes are experimental and subject to change in future
   versions of Dr.Jit.

List
----

The set of neural network module currently includes:

- Sequential evaluation of a list of models: :py:class:`nn.Sequential <Sequential>`.

- Linear and affine layers: :py:class:`nn.Linear <Linear>`.

- Encoding layers: :py:class:`nn.SinEncode <SinEncode>`, :py:class:`nn.TriEncode <TriEncode>`, :py:class:`nn.HashEncodingLayer <HashEncodingLayer>`.

- Activation functions and other nonlinear transformations: :py:class:`nn.ReLU <ReLU>`, :py:class:`nn.LeakyReLU <LeakyReLU>`,
  :py:class:`nn.Exp <nn.Exp>`, :py:class:`nn.Exp2 <Exp2>`, :py:class:`nn.Tanh <Tanh>`.

- Miscellaneous: :py:class:`nn.Cast <Cast>`, :py:class:`nn.ScaleAdd <ScaleAdd>`.

Example
-------

Below is a fully worked out example demonstrating how to use it to declare and
optimize a small `multilayer perceptron
<https://en.wikipedia.org/wiki/Multilayer_perceptron>`__ (MLP). This network
implements a 2D neural field (right) that we then fit to a low-resolution image of `The
Great Wave off Kanagawa
<https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa>`__ (left).

.. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/coopvec-screenshot.png
  :width: 600
  :align: center

The optimization uses the *Adam* optimizer (:py:class:`dr.opt.Adam
<drjit.opt.Adam>`) optimizer and a *gradient scaler*
(:py:class:`dr.opt.GradScaler <drjit.opt.GradScaler>`) for adaptive
mixed-precision training.

.. code-block:: python

    from tqdm.auto import tqdm
    import imageio.v3 as iio
    import drjit as dr
    import drjit.nn as nn
    from drjit.opt import Adam, GradScaler
    from drjit.auto.ad import Texture2f, TensorXf, TensorXf16, Float16, Float32, Array2f, Array3f

    # Load a test image and construct a texture object
    ref = TensorXf(iio.imread("https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/wave-128.png") / 256)
    tex = Texture2f(ref)

    # Establish the network structure
    net = nn.Sequential(
        nn.TriEncode(16, 0.2),
        nn.Cast(Float16),
        nn.Linear(-1, -1, bias=False),
        nn.LeakyReLU(),
        nn.Linear(-1, -1, bias=False),
        nn.LeakyReLU(),
        nn.Linear(-1, -1, bias=False),
        nn.LeakyReLU(),
        nn.Linear(-1, 3, bias=False),
        nn.Exp()
    )

    # Instantiate a random number generator to initialize the network weights
    rng = dr.rng(seed=0)

    # Instantiate the network for a specific backend + input size
    net = net.alloc(
        dtype=TensorXf16,
        size=2,
        rng=rng
    )

    # Convert to training-optimal layout
    weights, net = nn.pack(net, layout='training')
    print(net)

    # Optimize a single-precision copy of the parameters
    opt = Adam(lr=1e-3, params={'weights': Float32(weights)})

    # This is an adaptive mixed-precision (AMP) optimization, where a half
    # precision computation runs within a larger single-precision program.
    # Gradient scaling is required to make this numerically well-behaved.
    scaler = GradScaler()

    res = 256

    for i in tqdm(range(40000)):
        # Update network state from optimizer
        weights[:] = Float16(opt['weights'])

        # Generate jittered positions on [0, 1]^2
        t = dr.arange(Float32, res)
        p = (Array2f(dr.meshgrid(t, t)) + rng.random(Array2f, (2, res * res))) / res

        # Evaluate neural net + L2 loss
        img = Array3f(net(nn.CoopVec(p)))
        loss = dr.squared_norm(tex.eval(p) - img)

        # Mixed-precision training: take suitably scaled steps
        dr.backward(scaler.scale(loss))
        scaler.step(opt)

    # Done optimizing, now let's plot the result
    t = dr.linspace(Float32, 0, 1, res)
    p = Array2f(dr.meshgrid(t, t))
    img = Array3f(net(nn.CoopVec(p)))

    # Convert 'img' with shape 3 x (N*N) into a N x N x 3 tensor
    img = dr.reshape(TensorXf(img, flip_axes=True), (res, res, 3))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(ref)
    ax[1].imshow(dr.clip(img, 0, 1))
    fig.tight_layout()
    plt.show()

Hasgrid encodings
-----------------

In the above example, we use a neural network with 64 activations per layer.
Such small networks are usually not able to represent complex signals, and we
rely on the :py:class:`nn.TriEncode <TriEncode>` encoding, to improve the
quality of the learned signal. Dr.Jit also provides a hash grid encoding
(:py:class:`nn.HashGridEncoding <HashGridEncoding>`), proposed in `Instant NGP
<https://nvlabs.github.io/instant-ngp>`__. This encoding stores trainable
features in a multi-level grid. When evaluating the encoding, this grid is
indexed with a hashing funciton, features are interpolated between the
adjacient vertices. Dr.Jit also supports permotohedral encodings (:py:class:`nn.PermutoEncoding
<PermutoEncoding>`). They have been introduced in `PermutoSDF
<https://radualexandru.github.io/permuto_sdf>`__, and use triangles,
tetrahedrons and their higher dimensional equivalents. This improves their
performance compared to hash grid encodings for high dimensional inputs,
because the number of memory lookups scales linearly in dimension instead of
exponentially.

Because the features of the hashgrid can be stored in a different precision or
layout, than the weights of the network, we cannot simply pack them together
into the same weight layout. Using a hashgrid as a encoding layer in a
sequential network is therefore not directly possible. Instead, Dr.Jit provides
a :py:class:`nn.HashEncodingLayer <HashEncodingLayer>`, wrapping a hashgrid,
while not traversing its features. The following snippet shows a working
example learning the same image, but with a hashgrid encoding.

.. code-block:: python

    from tqdm.auto import tqdm
    import imageio.v3 as iio
    import drjit as dr
    import drjit.nn as nn
    from drjit.opt import Adam, GradScaler
    from drjit.auto.ad import Texture2f, TensorXf, TensorXf16, Float16, Float32, Array2f, Array3f

    # Load a test image and construct a texture object
    ref = TensorXf(iio.imread("https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/wave-128.png") / 256)
    tex = Texture2f(ref)

    # Instantiate a random number generator to initialize the network weights
    rng = dr.rng(seed=0)

    # Create a two dimensional hash grid encoding, with 8 levels, 2 features per
    # level and a scaling factor between levels of 1.5.
    enc = nn.HashGridEncoding(
        Float16,
        2,
        n_levels=8,
        n_features_per_level=2,
        per_level_scale=1.5,
        rng=rng,
    )

    # Alternatively we can also use a permutohedral encoding. In contrast to a hash
    # grid, it uses triangles, tetrahedrons and their higher dimensional
    # equivalences as simplexes. Their vertex count scales linearly with dimension,
    # allowing for higher dimensional inputs, while keeping the memory lookup
    # overhead minimal.
    # Uncomment the following lines to enable the permutohedral encoding.
    # enc = nn.PermutoEncoding(
    #     Float16,
    #     2,
    #     n_levels=8,
    #     n_features_per_level=2,
    #     per_level_scale=1.5,
    # )
    print(enc)


    # Establish the network structure.
    # In contrast to the previous example, we use a HashEncodingLayer, referencing
    # the previously created hash grid. Its parameters will not be part of the
    # packed weights, and have to be handled separately.
    net = nn.Sequential(
        nn.HashEncodingLayer(enc),
        nn.Cast(Float16),
        nn.Linear(-1, -1, bias=False),
        nn.LeakyReLU(),
        nn.Linear(-1, -1, bias=False),
        nn.LeakyReLU(),
        nn.Linear(-1, -1, bias=False),
        nn.LeakyReLU(),
        nn.Linear(-1, 3, bias=False),
        nn.Exp()
    )

    # Instantiate the network for a specific backend + input size
    net = net.alloc(TensorXf16, 2, rng=rng)

    # Convert to training-optimal layout
    weights, net = nn.pack(net, layout='training')
    print(net)

    # Optimize a single-precision copy of the parameters.
    # In addition to the network weights, we also add the parameters of the
    # encoding.
    opt = Adam(
        lr=1e-3,
        params={
            "mlp.weights": Float32(weights),
            "enc.params": Float32(enc.params),
        },
    )

    # This is an adaptive mixed-precision (AMP) optimization, where a half
    # precision computation runs within a larger single-precision program.
    # Gradient scaling is required to make this numerically well-behaved.
    scaler = GradScaler()

    res = 256

    for i in tqdm(range(40000)):
        # Update network state from optimizer
        weights[:] = Float16(opt['mlp.weights'])
        # Update the encoding parameters as well
        enc.params[:] = Float16(opt['enc.params'])

        # Generate jittered positions on [0, 1]^2
        t = dr.arange(Float32, res)
        p = (Array2f(dr.meshgrid(t, t)) + rng.random(Array2f, (2, res * res))) / res

        # Evaluate neural net + L2 loss
        img = Array3f(net(nn.CoopVec(p)))
        loss = dr.squared_norm(tex.eval(p) - img)

        # Mixed-precision training: take suitably scaled steps
        dr.backward(scaler.scale(loss))
        scaler.step(opt)

    # Done optimizing, now let's plot the result
    t = dr.linspace(Float32, 0, 1, res)
    p = Array2f(dr.meshgrid(t, t))
    img = Array3f(net(nn.CoopVec(p)))

    # Convert 'img' with shape 3 x (N*N) into a N x N x 3 tensor
    img = dr.reshape(TensorXf(img, flip_axes=True), (res, res, 3))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(ref)
    ax[1].imshow(dr.clip(img, 0, 1))
    fig.tight_layout()
    plt.show()
