import drjit as dr
from typing import (
    Any,
    Optional,
    Union,
    Mapping,
    MutableMapping,
    Iterator,
    Type,
    Tuple,
    Dict,
)
import sys

if sys.version_info < (3, 11):
    try:
        from typing_extensions import Unpack
    except ImportError:
        raise RuntimeError(
            "Dr.Jit requires the 'typing_extensions' package on Python <3.11"
        )
else:
    from typing import Unpack

# Learning rate can be a Python scalar or a Dr.Jit array
LearningRate = Union[float, dr.ArrayBase]


class Optimizer(MutableMapping[str, dr.ArrayBase]):
    """
    Gradient-based optimizer base class

    This class resembles a Python dictionary enabling retrieval and assignment
    of parameter values by name. It furthermore implements common functionality
    used by gradient-based optimizers, such as stochastic gradient descent
    (:py:class:`SGD`) and Adam (:py:class:`Adam`).

    Typically, optimizers are used as follows:

    .. code-block:: python

       # Create an optimizer object
       opt = Adam(lr=1e-3)

       # Register one or more parameters
       opt["x"] = Float(1, 2, 3)

       # Alternative syntax
       # opt = Adam(
       #     lr=1e-3,
       #     params={'x': Float(1, 2, 3)}
       # )

       # For some number of iterations..
       for i in range(1000):
           # Fetch the current parameter value
           x = opt["x"]

           # Compute a scalar loss value and backpropagate
           loss = my_optimization_task(x)
           dr.backward(loss)

           # Take a gradient step (details depend on the choice of optimizer)
           opt.step()

    Following :py:func:`opt.step() <Optimizer.step>`, some applications may
    need to project the parameter values onto a valid subset, e.g.:

    .. code-block:: python

       # Values outside of the range [0, 1] are not permitted
       opt['x'] = clip(opt['x'], 0, 1)

    You may register additional parameters or delete existing ones during an
    optimization.

    .. note::

       There are several notable differences compared to optimizers in PyTorch:

       - PyTorch keeps references to the original parameters and manipulates them
         in-place. The user must call ``opt.zero_grad()`` to clear out remaining
         gradients from the last iteration.

       - Dr.Jit optimizers *own* the parameters being optimized. The function
         ``opt.step()`` only updates this internal set of parameters without
         causing changes elsewhere.

         Compared to PyTorch, an optimization loop therefore involves some
         boilerplate code (e.g., ``my_param = opt["my_param"]``) to fetch
         parameter values and and use them to compute a differentiable
         objective.

       In general, it is recommended that you optimize Dr.Jit code using Dr.Jit
       optimizers, rather than combining frameworks with differentiable bridges
       (e.g., via the :py:func:`@dr.wrap <drjit.wrap>` decorator), which can add
       significant inter-framework communication and bookkeeping overheads.

    .. note::

       It is tempting to print or plot the loss decay during the optimization.
       However, doing so forces the CPU to wait for asynchronous execution to
       finish on the device (e.g., GPU), impeding the system's ability to
       schedule new work during this time. In other words, such a seemingly
       minor detail can actually have a detrimental impact on device
       utilization. As a workaround, consider printing *asynchronously* using
       :py:func:`dr.print(loss, symbolic=True) <print>`.
    """

    # Global learning rate parameter
    lr: LearningRate

    # Maps the parameter name to a tuple containing
    # - the current parameter value
    # - an parameter-specific learning rate (or None)
    # - an arbitrary sequence of additional optimizer-dependent state values
    state: Dict[
        str, Tuple[dr.ArrayBase, Optional[LearningRate], Unpack[Tuple[Any, ...]]]
    ]

    def __init__(
        self, lr: LearningRate, params: Optional[Mapping[str, dr.ArrayBase]] = None
    ):
        """
        Create an empty Optimizer object with the learning rate ``lr`` and initial
        parameter set ``params``.
        """

        if isinstance(lr, float) and lr < 0:
            raise RuntimeError("'lr' must be >0")

        self.lr = lr
        self.state = {}

        if params:
            self.update(params)

    def __contains__(self, key: object, /) -> bool:
        """Check whether the optimizer contains a parameter with the name ``key``."""
        return self.state.__contains__(key)

    def __getitem__(self, key: str, /) -> dr.ArrayBase:
        """Retrieve a parameter value from the optimizer."""
        return self.state[key][0]

    def __setitem__(self, key: str, value: dr.ArrayBase, /):
        """
        Overwrite a parameter value or register a new parameter.

        Supported parameter types includes:

        - Differentiable Dr.Jit arrays and nested arrays
        - Differentiable Dr.Jit tensors
        - Special array types (matrices, quaternions, complex numbers). These
          will be optimized component-wise.

        In contrast to assignment in a regular dictionary, this
        function conceptually creates a copy of the input parameter
        (conceptual because the :ref:`Copy-On-Write <cow>`
        optimization avoids an actual on-device copy).

        When ``key`` refers to a known parameter, the optimizer
        will overwrite it with ``value``. In doing so, it will
        preserve any associated optimizer state, such as momentum,
        adaptive step size, etc. When the new parameter value is
        substantially different (e.g., as part of a different
        optimization run), the previous momentum value may be
        meaningless, in which case a call to :py:func:`reset()` is
        advisable.

        When the new parameter value's :py:func:`.shape <drjit.ArrayBase.shape>`
        differs from the current setting, the implementation automatically
        calls :py:func:`reset()` to discard the associated optimizer state.

        When ``key`` does not refer to a known parameter, the optimizer will
        register it. Note that only differentiable parameters are
        supported---incompatible types will raise a ``TypeError``.
        """

        if not (dr.is_diff_v(value) and dr.is_float_v(value)):
            raise TypeError(
                f'Optimizer.__setitem__(): parameter "{key}" is not differentiable!'
            )

        if dr.width(value) == 0:
            raise RuntimeError(f'Optimizer.__setitem__(): parameter "{key}" is empty!')

        prev = self.state.get(key, None)

        # Make a detached copy and subsequently reattach it to the AD graph
        value = dr.detach(value)
        dr.enable_grad(value)

        if prev is not None and prev[0].shape == value.shape:
            self.state[key] = value, *prev[1:]
        else:
            self._reset(key, value)

    def __len__(self) -> int:
        """Return the number of number of registered parameters."""
        return len(self.state)

    def __delitem__(self, key: str, /) -> None:
        """Remove a parameter from the optimizer."""
        del self.state[key]

    def learning_rate(self, key: Optional[str] = None) -> Optional[LearningRate]:
        """
        Return the learning rate (globally, or of a specific parameter).

        When ``key`` is provided, the function returns the
        associated parameter-specific learning rate (or ``None``,
        if no learning rate was set for this parameter).

        When ``key`` is not provided, the function returns the default learning rate.
        """
        if key is None:
            return self.lr
        else:
            return self.state[key][1]

    def set_learning_rate(
        self,
        value: Union[LearningRate, Mapping[str, Optional[LearningRate]], None] = None,
        /,
        **kwargs: Optional[LearningRate],
    ) -> None:
        """
        Set the learning rate (globally, or of a specific parameter).

        This function can be used as follows:

        1. To modify the default learning rate of the optimizer:

           .. code-block:: python

              opt = Adam(lr=1e-3)

              # ... some time later:
              opt.set_learning_rate(1e-4)


        2. To modify the learning rate of a specific parameter:

           .. code-block:: python

              opt = Adam(lr=1e-3, params={'x': x, 'y': y})
              opt.set_learning_rate({'y': 1e-4})

              # Alternative calling convention
              opt.set_learning_rate(y=1e-4)

        Note that once the learning rate of a specific parameter is set, it
        *always* takes precedence over the global setting. You must remove the
        parameter-specific setting to return to the global default:

        .. code-block:: python

           opt.set_learning_rate(y=None)
        """

        if isinstance(value, float) or isinstance(value, dr.ArrayBase):
            self.lr = value
        elif isinstance(value, Mapping):
            for k, lr in value.items():
                state = self.state[k]
                self.state[k] = (state[0], lr, *state[2:])
        if kwargs:
            self.set_learning_rate(kwargs)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator traversing the names of registered parameters."""
        return iter(self.state.keys())

    def keys(self) -> Iterator[str]:  # type: ignore
        """Return an iterator traversing the names of registered parameters."""
        return iter(self.state.keys())

    def values(self) -> Iterator[dr.ArrayBase]:  # type: ignore
        """Return an iterator traversing the values of registered parameters."""
        return (v[0] for v in self.state.values())

    def items(self) -> Iterator[Tuple[str, dr.ArrayBase]]:  # type: ignore
        """Return an iterator traversing the names and values of registered parameters."""
        return ((k, v[0]) for k, v in self.state.items())

    def update(self, params: Optional[Mapping[str, dr.ArrayBase]] = None, **args: dr.ArrayBase) -> None:  # type: ignore
        """
        Overwrite multiple parameter values at once.

        This function simply calls :py:func:`__setitem__` multiple
        times. Like :py:func:`dict.update()`, :py:func:`update()
        <update>` supports two calling conventions:

        .. code-block:: python

           # Update using a dictionary
           opt.update({'key_1': value_1, 'key_2': value_2})

           # Update using a variable keyword arguments
           opt.update(key_1=value_1, key_2=value_2)
        """
        if params:
            for k, v in params.items():
                self[k] = v
        if args:
            self.update(args)

    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset the internal state (e.g., momentum, adaptive learning rate, etc.)
        associated with the parameter ``key``. When ``key=None``, the
        implementation resets the state of *all* parameters.
        """

        if key is not None:
            self._reset(key, self[key])
        else:
            for k in self.state.keys():
                self._reset(k, self[k])

    # --------------------------------------------------------------------
    #    Functionality that must be provided by subclasses
    # --------------------------------------------------------------------

    # To be provided by subclasses
    def step(
        self,
        /,
        eval: bool = True,
        grad_scale: Optional[LearningRate] = None,
        active: Optional[dr.ArrayBase] = None,
    ) -> None:
        """
        Take a gradient step.

        This function visits each registered parameter in turn and

        1. extracts the associated gradient (:py:attr:`.grad <drjit.ArrayBase.grad>`),
        2. takes a step using an optimizer-dependent update rule, and
        3. reattaches the resulting new state with the AD graph.

        Args:
            eval (bool):
                If ``eval=True`` (the default), the system will explicitly
                evaluate the resulting parameter state, causing a kernel lauch.
                Set ``eval=False`` if you wish to perform further computation
                that should be fused into the optimizer step.

            grad_scale (float | drjit.ArrayBase):
                Use this parameter to scale all gradients by a custom amount.
                Dr.Jit uses this parameter for automatic mixed-precision training.

            active (drjit.ArrayBase | None):
                This parameter can be used to pass a 1-element boolean mask. A
                value of ``Bool(False)`` disables the optimizer state update.
                Dr.Jit uses this parameter for automatic mixed-precision training.
        """
        raise Exception("Optimizer.step(): missing implementation!")

    # To be provided by subclasses
    def _reset(self, key: str, value: dr.ArrayBase, /) -> None:
        raise Exception(f"Optimizer._reset({key}, {value}): missing implementation!")


class _LRCache(Dict[Tuple[Type[dr.ArrayBase], float], dr.ArrayBase]):
    """
    Implementation detail: learning rate cache.

    A parameter's combined learning rate (consisting of a global or
    parameter-specific value, debiasing factors, automatic mixed-precision
    scaling, etc.) tends to vary from iteration to iteration.

    It is important to express these scale factors using opaque arrays to
    prevent them from being baked into generated kernel code, as this would
    interfere with kernel caching. At the same time, many parameters will
    generally reuse the same scale factor in a specific optimization iteration.
    It is nice to reuse the opaque array once it has been created.

    The _LRCache class is a simple internal cache to enable this reuse.
    """

    def product(
        self, tp: Type[dr.ArrayT], *args: Union[float, dr.ArrayBase]
    ) -> dr.ArrayT:
        """
        Compute the product of the entries of ``*args`` and return a
        result of type ``tp``.
        """
        scale_f, scale_o = 1.0, None

        for arg in args:
            if isinstance(arg, float) or isinstance(arg, int):
                scale_f *= arg
            elif scale_o is None:
                scale_o = arg
            else:
                scale_o *= arg

        key = (tp, scale_f)
        result = self.get(key, None)

        if result is None:
            result = dr.opaque(tp, scale_f)
            self[key] = result

        if scale_o is not None:
            dr.make_opaque(scale_o)
            result = result * scale_o

            if not isinstance(scale_o, tp):
                raise TypeError(
                    f"Scaled step size has type {type(scale_o)}, expected {tp}"
                )

        return result  # type: ignore


class SGD(Optimizer):
    """
    Implements basic *stochastic gradient descent* (SGD) with a fixed learning
    rate and, optionally, momentum (0.9 is a typical parameter value for the
    ``momentum`` parameter).

    The default initailization (``momentum=0``) uses the following update equation:

    .. math::

       \\begin{align*}
       \\mathbf{p}_{i+1} &= \\mathbf{p}_i - \\eta\\cdot\\mathbf{g}_{i+1},
       \\end{align*}

    where :math:`\\mathbf{p}_i` is the parameter value at iteration :math:`i`,
    :math:`\\mathbf{g}_i` denotes the associated gradient, and :math:`\\eta` is
    the learning rate.

    Momentum-based SGD (with ``momentum>0``, ``nesterov=False``) uses the
    update equation:

    .. math::

       \\begin{align*}
       \\mathbf{v}_{i+1} &= \\mu\\cdot\\mathbf{v}_i + \\mathbf{g}_{i+1}\\\\
       \\mathbf{p}_{i+1} &= \\mathbf{p}_i - \\eta \\cdot \\mathbf{v}_{i+1},
       \\end{align*}

    where :math:`\\mathbf{v}` is the velocity and :math:`\\mu` is the momentum
    parameter. Nesterov-style SGD (``nesterov=True``) switches to the following
    update rule:

    .. math::

       \\begin{align*}
       \\mathbf{v}_{i+1} &= \\mu \\cdot \\mathbf{v}_i + \\mathbf{g}_{i+1}\\\\
       \\mathbf{p}_{i+1} &= \\mathbf{p}_i - \\eta \\cdot (\\mathbf{g}_{i+1} + \\mu \\mathbf{v}_{i+1}).
       \\end{align*}

    Some frameworks implement variations of the above quations. The code in
    Dr.Jit was designed to reproduce the behavior of `torch.optim.SGD
    <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`__.
    """

    def __init__(
        self,
        lr: LearningRate,
        momentum: float = 0.0,
        nesterov: bool = False,
        mask_updates: bool = False,
        params: Optional[Mapping[str, dr.ArrayBase]] = None,
    ):
        """
        Args:

            lr (float | drjit.ArrayBase):
                Learning rate parameter. You may want to try different values
                (e.g. powers of two) to find the best setting for a specific
                problem. Use :py:func:`Optimizer.set_learning_rate` to
                later adjust this value globally, or for specific parameters.

            momentum (float):
                The momentum factor as described above. Larger values will
                cause past gradients to persist for a longer amount of time.

            mask_updates (bool):
                Set this parameter to ``True`` to mask updates to zero-valued
                gradient components. See the documentation of :py:class:`Adam`
                for further detail details.

            params (Mapping[str, drjit.ArrayBase] | None):
                Optional dictionary-like object containing an initial set of
                parameters.
        """

        if momentum < 0:
            raise RuntimeError("'momentum' must be >= 0")

        if nesterov and momentum == 0:
            raise RuntimeError("Nesterov acceleration assumes momentum > 0.")

        self.momentum = momentum
        self.nesterov = nesterov
        self.mask_updates = mask_updates

        super().__init__(lr, params)

    def step(
        self,
        /,
        eval: bool = True,
        grad_scale: Optional[LearningRate] = None,
        active: Optional[dr.ArrayBase] = None,
    ) -> None:
        cache = _LRCache()

        for key, state in self.state.items():
            value: dr.ArrayBase  # Current parameter value
            lr: Optional[LearningRate]  # Parameter-specific learning rate
            v: Optional[dr.ArrayBase]  # Momentum accumulator

            # Unpack optimizer state
            value, lr, v = state

            # Fetch the parameter gradient and convert special array types
            # (e.g. complex numbers) into ones with element-wise semantics
            grad = value.grad.array
            if grad_scale is not None:
                grad *= grad_scale

            if self.momentum == 0:
                assert v is None
                step = grad
                v_next = None
            else:
                assert v is not None

                # Check for shape incompatibilities and potentially reset
                # the parameter state
                if grad.shape != v.shape:
                    self._reset(key, value)
                    value, lr, v = state
                v_next = dr.fma(self.momentum, v, grad)

                if self.nesterov:
                    step = dr.fma(self.momentum, v_next, grad)
                else:
                    step = v_next

            # Compute the step size scale, which is a product of
            # - Adaptive/parameter-specific scaling
            # - Optional: a global scale
            scale = cache.product(
                dr.leaf_t(grad),  # Desired type
                lr if lr is not None else self.lr,
                -1.0,
            )

            # Optional: mask updates to components with zero-valued gradients
            mask = False
            if self.mask_updates and self.momentum != 0:
                mask |= grad == 0

            # Optional: mask updates, e.g., due to adaptive multi precision training
            if active is not None:
                mask |= ~active

            if mask is not False:
                assert v_next is not None
                v_next[mask] = v
                step[mask] = 0

            # Construct new parameter value and reattach to AD graph
            new_value = dr.fma(step, scale, dr.detach(value).array)
            value_tp = type(value)
            if type(new_value) is not value_tp:
                new_value = value_tp(new_value)
            dr.enable_grad(new_value)

            # Update the optimizer state and schedule it for evaluation
            state = new_value, lr, v_next
            dr.schedule(state)
            self.state[key] = state

        # Submit a kernel containing queued parameter updates
        if eval:
            dr.eval()

    def _reset(self, key: str, value: dr.ArrayBase, /) -> None:
        tp = dr.array_t(value)
        m = None if self.momentum == 0 else dr.opaque(tp, 0, value.shape)
        self.state[key] = value, None, m

    def __repr__(self):
        """Return a human-readable string representation"""
        lr_dict: Dict[str, LearningRate] = dict(default=self.lr)
        for k, state in self.state.items():
            lr = state[1]
            if lr is not None:
                lr_dict[k] = lr

        return (
            "SGD[\n"
            "  state = %s,\n"
            "  lr = %s,\n"
            "  momentum = %g\n"
            "  nesterov = %s\n"
            "]" % (list(self.keys()), lr_dict, self.momentum, self.nesterov)
        )


class RMSProp(Optimizer):
    """
    Implements the RMSProp optimizer explained in `lecture notes
    <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`__
    by G. Hinton.

    RMSProp scales the learning rate by the reciprocal of a running average of
    the *magnitude* of past gradients:

    .. math::

       \\begin{align*}
           \\mathbf{m}_{i+1} &= \\alpha \\cdot \\mathbf{m}_i + (1-\\alpha)\\cdot\\mathbf{g}_{i+1}^2\\\\
           \\mathbf{p}_{i+1} &= \\mathbf{p}_i - \\frac{\\eta}{\\sqrt{\\mathbf{m}_{i+1}+\\varepsilon}}\\, \\mathbf{g}_{i+1},
       \\end{align*}


    where :math:`\\mathbf{p}_i` is the parameter value at iteration :math:`i`,
    :math:`\\mathbf{g}_i` denotes the associated gradient,
    :math:`\\mathbf{m}_i` accumulates the *second moment*, :math:`\\eta` is the
    learning rate, and :math:`\\varepsilon` is a small number to avoid division
    by zero.

    The implementation reproduces the behavior of `torch.optim.RMSprop
    <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html>`__.
    """

    # Second moment EMA weight
    alpha: float

    # Fudge factor to avoid division by zero
    epsilon: float

    # Mask updates of parameters that did not receive a gradient?
    mask_updates: bool

    def __init__(
        self,
        lr: LearningRate,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        mask_updates: bool = False,
        params: Optional[Mapping[str, dr.ArrayBase]] = None,
    ):
        """
        Construct a RMSProp optimizer instance.

        Args:

            lr (float | drjit.ArrayBase):
                Learning rate parameter. You may want to try different values
                (e.g. powers of two) to find the best setting for a specific
                problem. Use :py:func:`Optimizer.set_learning_rate` to
                later adjust this value globally, or for specific parameters.

            alpha (float):
                Weight of the second-order moment exponential moving average
                (EMA). Values approaching ``1`` will cause past gradients to
                persist for a longer amount of time.

            mask_updates (bool):
                Set this parameter to ``True`` to mask updates to zero-valued
                gradient components. See the documentation of :py:class:`Adam`
                for further detail details.

            params (Mapping[str, drjit.ArrayBase] | None):
                Optional dictionary-like object containing an initial set of
                parameters.
        """

        super().__init__(lr, params)

        if alpha < 0 or alpha >= 1:
            raise RuntimeError("'alpha' must be on the interval [0, 1)")
        if epsilon < 0:
            raise RuntimeError("'epsilon' must be >0")

        self.alpha = alpha
        self.epsilon = epsilon
        self.mask_updates = mask_updates

    def step(
        self,
        /,
        eval: bool = True,
        grad_scale: Optional[LearningRate] = None,
        active: Optional[dr.ArrayBase] = None,
    ) -> None:
        cache = _LRCache()

        for key, state in self.state.items():
            value: dr.ArrayBase  # Current parameter value
            lr: Optional[LearningRate]  # Parameter-specific learning rate
            m_tp: dr.ArrayBase  # Second moment EMA state from previous iteration

            # Unpack optimizer state
            value, lr, m_tp = state

            # Fetch the parameter gradient and convert special array types
            # (e.g. complex numbers) into ones with element-wise semantics
            grad = value.grad.array
            if grad_scale is not None:
                grad *= grad_scale

            # Check for shape incompatibilities and potentially reset
            # the parameter state
            if grad.shape != m_tp.shape:
                self._reset(key, value)
                value, lr, m_tp = state

            # Update second moment EMA
            m_t = dr.lerp(dr.square(grad), m_tp, self.alpha)

            # Compute the step size scale, which is a product of
            # - Adaptive/parameter-specific scaling
            # - Optional: a global scale
            scale = cache.product(
                dr.leaf_t(grad),  # Desired type
                lr if lr is not None else self.lr,
                -1.0,
            )

            # Use a faster approximation for tiny epsilon values
            if self.epsilon <= 1e-6:
                step = grad * dr.rsqrt(m_t + self.epsilon**2)
            else:
                step = grad / (dr.sqrt(m_t) + self.epsilon)

            # Optional: mask updates to components with zero-valued gradients
            mask = False
            if self.mask_updates:
                mask |= grad == 0

            # Optional: mask updates, e.g., due to automatic mixed precision training
            if active is not None:
                mask |= ~active

            if mask is not False:
                m_t[mask] = m_tp
                step[mask] = 0

            # Construct new parameter value and reattach to AD graph
            new_value = dr.fma(step, scale, dr.detach(value).array)
            value_tp = type(value)
            if type(new_value) is not value_tp:
                new_value = value_tp(new_value)
            dr.enable_grad(new_value)

            # Update the optimizer state and schedule it for evaluation
            state = new_value, lr, m_t
            dr.schedule(state)
            self.state[key] = state

        # Submit a kernel containing queued parameter updates
        if eval:
            dr.eval()

    # Implementation detail of Optimizer.reset()
    def _reset(self, key: str, value: dr.ArrayBase, /) -> None:
        tp = dr.array_t(value)
        m_t = dr.opaque(tp, 0, value.shape)
        self.state[key] = value, None, m_t

    def __repr__(self):
        """Return a human-readable string representation"""
        lr_dict: Dict[str, LearningRate] = dict(default=self.lr)
        for k, state in self.state.items():
            lr = state[1]
            if lr is not None:
                lr_dict[k] = lr

        return (
            "RMSProp[\n"
            "  state = %s,\n"
            "  lr = %s,\n"
            "  alpha = %g,\n"
            "  epsilon = %g\n"
            "]"
            % (
                list(self.keys()),
                lr_dict,
                self.alpha,
                self.epsilon,
            )
        )


class Adam(Optimizer):
    """
    This class implements the Adam optimizer as presented in the paper *Adam: A
    Method for Stochastic Optimization* by Kingman and Ba, ICLR 2015.

    Adam effectively combines momentum (as in :py:class:`SGD` with
    ``momentum>0``) with the adaptive magnitude-based scale factor from
    :py:class:`RMSProp`. To do so, it maintains two *exponential moving
    averages* (EMAs) per parameter: :math:`\\mathbf{m}_i` for the first moment,
    and :math:`\\mathbf{v}_i` for the second moment. This triples the memory
    usage and should be considered when optimizing very large representations.

    The method uses the following update equation:

    .. math::

       \\begin{align*}
           \\mathbf{m}_{i+1} &= \\beta_1 \\cdot \\mathbf{m}_i + (1-\\beta_1)\\cdot\\mathbf{g}_{i+1}\\\\
           \\mathbf{v}_{i+1} &= \\beta_2 \\cdot \\mathbf{v}_i + (1-\\beta_2)\\cdot\\mathbf{g}_{i+1}^2\\\\
           \\mathbf{p}_{i+1} &= \\mathbf{p}_i - \\eta \\frac{1-\\beta_2^{i+1}}{1-\\beta_1^{i+1}} \\frac{\\mathbf{v}_{i+1}}{\\sqrt{\\mathbf{m}_{i+1}+\\varepsilon}},
       \\end{align*}

    where :math:`\\mathbf{p}_i` is the parameter value at iteration :math:`i`,
    :math:`\\mathbf{g}_i` denotes the associated gradient, :math:`\\eta` is the
    learning rate, and :math:`\\varepsilon` is a small number to avoid division
    by zero.


    The scale factor :math:`\\frac{1-\\beta_2^{i+1}}{1-\\beta_1^{i+1}}`
    corrects for the zero-valued initialization of the
    moment accumulators :math:`\\mathbf{m}_i`
    and :math:`\\mathbf{v}_i` at :math:`i=0`.

    This class also implements two extensions that are turned off by default.
    See the descriptions of the ``mask_updates`` and ``uniform`` parameters below.
    """

    # First moment EMA weight
    beta_1: float

    # Second moment EMA weight
    beta_2: float

    # Fudge factor to avoid division by zero
    epsilon: float

    # Mask updates of parameters that did not receive a gradient?
    mask_updates: bool

    # Uniform Adam: use maximum of second moment [Nicolet et al. 2021]
    uniform: bool

    def __init__(
        self,
        lr: LearningRate,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        mask_updates: bool = False,
        uniform: bool = False,
        params: Optional[Mapping[str, dr.ArrayBase]] = None,
    ):
        """
        Construct a new Adam optimizer object. The default parameters
        replicate the behavior of the original method.

        Args:

            lr (float | drjit.ArrayBase):
                Learning rate parameter. You may want to try different values
                (e.g. powers of two) to find the best setting for a specific
                problem. Use :py:func:`Optimizer.set_learning_rate` to
                later adjust this value globally, or for specific parameters.

            beta_1 (float):
                Weight of the first-order moment exponential moving average
                (EMA). Values approaching ``1`` will cause past gradients to
                persist for a longer amount of time.

            beta_2 (float):
                Weight of the second-order moment EMA. Values approaching ``1``
                will cause past gradients to persist for a longer amount of
                time.

            mask_updates (bool):
                Set this parameter to ``True`` to mask updates to zero-valued
                gradient components. This can can be preferable in some types
                of differentiable Monte Carlo simulations, where only a subset
                of the parameters is observed during a typical optimization
                step.

                In such a situation, the original Adam method will behave as
                follows:

                1. Momentum accumulated during previous iterations will
                   continue to affect the parameter.

                2. The optimizer will accumulate zero-valued gradient
                   components into the first and second moment EMAs.

                When ``mask_updates`` is ``True``, the optimizer interprets
                zero-valued derivatives as a lack of information and skips both
                of the above steps, resembling PyTorch's `SparseAdam optimizer
                <https://pytorch.org/docs/1.9.0/generated/torch.optim.SparseAdam.html>`_.

            uniform (bool):
                If enabled, the optimizer will use the *UniformAdam* variant of
                Adam [Nicolet et al. 2021], where the update rule uses the
                *maximum* of the second moment estimates at the current step
                instead of the per-element second moments.

            params (Mapping[str, drjit.ArrayBase] | None):
                Optional dictionary-like object containing an initial set of
                parameters.
        """

        super().__init__(lr, params)

        if beta_1 < 0 or beta_1 >= 1:
            raise RuntimeError("'beta_1' must be on the interval [0, 1)")
        if beta_2 < 0 or beta_2 >= 1:
            raise RuntimeError("'beta_2' must be on the interval [0, 1)")
        if epsilon < 0:
            raise RuntimeError("'epsilon' must be >0")

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.mask_updates = mask_updates
        self.uniform = uniform

    def step(
        self,
        /,
        eval: bool = True,
        grad_scale: Optional[LearningRate] = None,
        active: Optional[dr.ArrayBase] = None,
    ) -> None:
        cache = _LRCache()

        for key, state in self.state.items():
            value: dr.ArrayBase  # Current parameter value
            lr: Optional[LearningRate]  # Parameter-specific learning rate
            t: int  # Integer time/iteration value
            m_tp: dr.ArrayBase  # First moment EMA state from previous iteration
            v_tp: dr.ArrayBase  # Second moment EMA state from previous iteration

            # Unpack optimizer state
            value, lr, t, m_tp, v_tp = state

            # Fetch the parameter gradient and convert special array types
            # (e.g. complex numbers) into ones with element-wise semantics
            grad = value.grad.array
            if grad_scale is not None:
                grad *= grad_scale

            # Check for shape incompatibilities and potentially reset
            # the parameter state
            if grad.shape != m_tp.shape:
                self._reset(key, value)
                value, lr, t, m_tp, v_tp = state

            # Update moment EMAs
            m_t = dr.lerp(grad, m_tp, self.beta_1)
            v_t = dr.lerp(dr.square(grad), v_tp, self.beta_2)

            # Increase the iteration count
            t += 1

            # Compute the step size scale, which is a product of
            # - EMA debiasing factor
            # - Adaptive/parameter-specific scaling
            # - Optional: a global scale
            scale = cache.product(
                dr.leaf_t(grad),  # Desired type
                -dr.sqrt(1 - self.beta_2**t) / (1 - self.beta_1**t),
                lr if lr is not None else self.lr,
            )

            # Optional: use maximum of second order term
            v_tm = dr.max(v_t) if self.uniform else v_t

            # Use a faster approximation for tiny epsilon values
            if self.epsilon <= 1e-6:
                step = m_t * dr.rsqrt(v_tm + self.epsilon**2)
            else:
                step = m_t / (dr.sqrt(v_tm) + self.epsilon)

            # Optional: mask updates to components with zero-valued gradients
            mask = False
            if self.mask_updates:
                mask |= grad == 0

            # Optional: mask updates, e.g., due to automatic mixed precision training
            if active is not None:
                mask |= ~active

            if mask is not False:
                # Known issue: we don't mask the update to 't' here. That would
                # require moving this parameter to the GPU, with a whole bunch
                # of downsides. Oh well.
                m_t[mask] = m_tp
                v_t[mask] = v_tp
                step[mask] = 0

            # Construct new parameter value and reattach to AD graph
            new_value = dr.fma(step, scale, dr.detach(value).array)
            value_tp = type(value)
            if type(new_value) is not value_tp:
                new_value = value_tp(new_value)
            dr.enable_grad(new_value)

            # Update the optimizer state and schedule it for evaluation
            state = new_value, lr, t, m_t, v_t
            dr.schedule(state)
            self.state[key] = state

        # Submit a kernel containing queued parameter updates
        if eval:
            dr.eval()

    # Implementation detail of Optimizer.reset()
    def _reset(self, key: str, value: dr.ArrayBase, /) -> None:
        tp = dr.array_t(value)
        m_t = dr.opaque(tp, 0, value.shape)
        v_t = dr.opaque(tp, 0, value.shape)
        self.state[key] = value, None, 0, m_t, v_t

    def __repr__(self):
        """Return a human-readable string representation"""
        lr_dict: Dict[str, LearningRate] = dict(default=self.lr)
        for k, state in self.state.items():
            lr = state[1]
            if lr is not None:
                lr_dict[k] = lr

        return (
            "Adam[\n"
            "  state = %s,\n"
            "  lr = %s,\n"
            "  beta = (%g, %g),\n"
            "  epsilon = %g\n"
            "]"
            % (
                list(self.keys()),
                lr_dict,
                self.beta_1,
                self.beta_2,
                self.epsilon,
            )
        )


class GradScaler:
    """
    Gradient scaler for automatic mixed-precision training.

    It is sometimes necessary to perform some part of a computation using
    lower precision (e.g., :py:class:`drjit.auto.Float16`) to improve storage
    and runtime efficiency. One issue with such lower-precision arithmetic is
    that gradients tend to underflow to zero, which can break the optimization.

    The :py:class:`GradientScaler` class implements a strategy for *automatic
    mixed precision* (AMP) training to prevent such numerical issues. A
    comprehensive overview of AMP can be found `here
    <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>`_

    AMP in Dr.Jit works as follows:

    1. Construct a :py:class:`GradientScaler` instance prior to the
       optimization loop. Suppose it is called ``scaler``.

    1. Invoke :py:func:`scaler.scale(loss)` function to scale the
       optimization loss by a suitable value to prevent gradient underflow,
       and then propagate derivatives using :py:func:`drjit.backward()` or
       a similar AD function.

    2. Replace the call to :py:func:`opt.step() <Optimizer.step>` with
       :py:func:`scale.step(opt)` function, which removes the scaling
       prior to the gradient step.

    Concretely, this might look as follows:

    .. code-block:: python

        opt = Adam(lr=1e-3)
        opt['my_param'] = 0
        scaler = GradScaler()
        for i in range(1000):
            my_param = opt['my_param']
            loss = my_func(my_param)
            dr.backward(scaler.scale(loss))
            scaler.step(opt)

    A large scale factor can also cause the opposite problem: *gradient
    overflow*, which manifests in the form of infinity and NaN-valued gradient
    components. :py:class:`GradientScaler` automatically detects this, skips
    the optimizer step, and decreases the step size.

    The implementation starts with a relatively aggressive scale factor that is
    likely to cause overflows, hence it may appear that the optimizer is
    initially stagnant for a few iterations. This is expected.
    """

    scale_factor: Union[float, dr.ArrayBase]
    backoff_factor: float
    growth_factor: float
    growth_interval: int

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """
        The Dr.Jit :py:class:`GradScaler` class follows the high-level API of
        `pytorch.amp.GradScaler <https://pytorch.org/docs/stable/notes/amp_examples.html>__`.

        Args:

            init_scale (float):
                The initial scale factor.

            growth_factor (float):
                When ``growth_interval`` optimization steps have taken place
                without overflows, :py:class:`GradScaler` will begin to
                progressively increase the scale by multiplying it with
                ``growth_factor`` at every iteration until an overflow is
                again detected.

            backoff_factor (float):
                When an overflow issue is detected, :py:class:`GradScaler`
                will decrease the scale factor by multiplying it with
                ``backoff_factor``.

            growth_interval (int):
                A large iteration count, following which it can be helpful
                to begin exploring larger scale factors.
        """

        self.scale_factor = init_scale
        self.it = 0
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

        if self.backoff_factor >= 1:
            raise Exception("'backoff_factor' must be < 1.")
        if self.growth_factor < 1:
            raise Exception("'growth_factor' must be >= 1.")

    def scale(self, arg: dr.ArrayBase, /):
        """
        Multiply ``arg`` by the current scale factor.
        """

        return arg * self.scale_factor

    def unscale(self, arg: dr.ArrayBase, /):
        """
        Multiply ``arg`` by the reciprocal of the current scale factor.

        When using :py:func:`.step()`, explicit unscaling is usually not needed.
        """

        return arg / self.scale_factor

    def step(self, opt: Optimizer, **kwargs) -> None:
        """
        Take a gradient step via ``opt``, while being careful to remove the
        previously introduced gradient scale factor.
        """
        if len(opt) == 0:
            return

        scale, it = self.scale_factor, self.it

        good = True
        for v in opt.values():
            good &= dr.all(dr.isfinite(v.grad))
        grow = it >= self.growth_interval

        next_scale = dr.select(
            good,
            dr.select(grow, scale * self.growth_factor, scale),
            scale * self.backoff_factor,
        )
        next_it = dr.select(good, it + 1, 0)

        dr.schedule(next_scale, next_it)
        opt.step(grad_scale=dr.rcp(scale), active=good, **kwargs)  # type: ignore

        self.scale_factor = next_scale
        self.it = next_it
