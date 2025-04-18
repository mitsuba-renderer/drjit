# ruff: noqa: E721  -- don't warn about exact type comparisons done via 'is'

import drjit as dr
import typing
import types

def pytorch_check(value, /):
    '''Returns ``True`` if ``value`` is a PyTorch tensor'''
    return type(value).__module__ == 'torch' and type(value).__name__ == 'Tensor'


def pytorch_fp_check(value, /):
    '''Returns ``True`` if ``value`` is a PyTorch floating point tensor'''
    return type(value).__module__ == 'torch' and type(value).__name__ == 'Tensor' and value.dtype.is_floating_point


def jax_check(value, /):
    '''Returns ``True`` if ``value`` is a JAX tensor'''
    return type(value).__module__.startswith('jaxlib')


def tf_check(value, /):
    '''Returns ``True`` if ``value`` is a TensorFlow tensor'''
    return type(value).__module__.startswith('tensorflow')


def tf_var_check(value, /):
    '''Returns ``True`` if ``value`` is a TensorFlow variable'''
    return type(value).__name__ == 'ResourceVariable'


def tf_fp_check(value):
    '''Returns True if value is a TensorFlow floating point tensor'''
    return tf_check(value) and value.dtype.is_floating


def pytree_check(value, /):
    '''Returns ``True`` if ``value`` is a structural element of a PyTree'''
    tp = type(value)
    return tp is list or tp is tuple or \
           tp is dict or getattr(tp, 'DRJIT_STRUCT', None) is not None


def apply(fn, a, /):
    '''Helper function to recursively map a PyTree through the function ``fn``'''
    tp = type(a)

    result = fn(a)
    if result is not Ellipsis:
        return result
    elif tp is list:
        return [apply(fn, v) for v in a]
    elif tp is tuple:
        return tuple(apply(fn, v) for v in a)
    elif tp is dict:
        return {k: apply(fn, v) for k, v in a.items()}
    else:
        desc = getattr(tp, 'DRJIT_STRUCT', None)
        if type(desc) is dict:
            result = tp()
            for k in desc:
                setattr(result, k, apply(fn, getattr(a, k)))
            return result
        else:
            return a


def apply2(fn, a, b, /):
    '''
    Helper function to recursively map two compatible PyTrees
    through the function ``fn``
    '''
    result = fn(a, b)
    if result is not Ellipsis:
        return result

    ta, tb = type(a), type(b)
    if ta is not tb:
        raise TypeError(f'Incompatible types: {ta} and {tb}.')

    if ta is list:
        assert len(a) == len(b)
        return [apply2(fn, a[i], b[i]) for i in range(len(a))]
    elif ta is tuple:
        assert len(a) == len(b)
        return tuple(apply2(fn, a[i], b[i]) for i in range(len(a)))
    elif ta is dict:
        assert a.keys() == b.keys()
        return {k: apply2(fn, a[k], b[k]) for k, v in a.items()}
    else:
        desc = getattr(ta, 'DRJIT_STRUCT', None)
        if type(desc) is dict:
            result = type(a)()
            for k in desc:
                setattr(result, k, apply2(fn, getattr(a, k), getattr(b, k)))
            return result
        else:
            return a


def from_drjit(value, target, enable_grad = False, /):
    '''
    Convert a PyTree containing Dr.Jit arrays/tensors to another array
    programming framework as identified by ``target``.

    The function returns the output PyTree as well as a sequence capturing the
    original type of each converted Dr.Jit type. This is useful when those same
    exact types should be restored in a subsequent conversion by ``to_drjit``.
    '''

    value_tp = []

    def fn(h, /):
        tp = type(h)
        value_tp.append(tp)

        if dr.is_array_v(tp):
            if not dr.is_tensor_v(h):
                h = dr.tensor_t(tp)(h)
            r = getattr(h, target)()
            if enable_grad and target == 'torch' and r.dtype.is_floating_point:
                r.requires_grad = True
            return r
        return ...

    return apply(fn, value), value_tp


def to_drjit(value, source, value_tp = None, enable_grad = None):
    '''
    Convert a PyTree containing tensors from another array programming
    framework identified by ``source`` into Dr.Jit tensors.

    Optionally, the function can restore the array types within an input PyTree
    previously captured by ``from_drjit``.
    '''

    tp_index = 0

    def fn(h, /):
        nonlocal tp_index
        tp = value_tp[tp_index] if value_tp is not None else None
        tp_index += 1
        if (source == 'torch' and pytorch_check(h)) or \
           (source == 'jax'   and jax_check(h)) or \
           (source == 'tf'    and tf_check(h)):

            if source == 'tf' and tf_var_check(h) :
                r = dr.detail.import_tensor(h.value(), True)
            else:
                r = dr.detail.import_tensor(h, True)
            if type(r) is not tp and dr.is_array_v(tp):
                r = tp(r)
            if source == 'torch' and enable_grad:
                if h.requires_grad:
                    dr.enable_grad(r)
            if source == 'tf' and enable_grad:
                # There is no TF equivalent to h.requires_grad.
                # We hence enable gradients for all trainable variables
                # and tensors of floating dtype.
                if tf_var_check(h):
                    if h.trainable:
                        dr.enable_grad(r)
                elif h.dtype.is_floating:
                    dr.enable_grad(r)
            return r
        return ...

    return apply(fn, value)


def pytorch_filter_fp(value, /):
    '''Extract a flat list of floating point PyTorch tensors from the PyTree ``value``'''

    result = []

    def fn(h, /):
        if pytorch_fp_check(h):
            result.append(h)
        return ...

    apply(fn, value)
    return result


def tf_filter_fp(value, /):
    '''Extract a flat list of floating point TensorFlow tensors from the PyTree ``value``'''

    result = []

    def fn(h, /):
        if tf_fp_check(h):
            result.append(h)
        return ...

    apply(fn, value)
    return result


def pytorch_grad(value, /):
    '''Extract a the gradients of PyTorch tensors from the PyTree ``value``'''

    def fn(h, /):
        if pytorch_check(h):
            return h.grad
        return ...

    return apply(fn, value)


def pytorch_tangent(value, /):
    '''Extract a the tangents of PyTorch arrays from the PyTree ``value``'''

    def fn(h, /):
        if pytorch_fp_check(h):
            from torch.autograd.forward_ad import unpack_dual
            return unpack_dual(h).tangent
        return ...

    return apply(fn, value)


def pytorch_make_dual(a, b, /):
    '''Build combined primal/tangent PyTrees for PyTorch forward-mode AD'''

    def fn(a, b):
        if type(b) is float and b == 0.0:
            # Allow non-differentiable types when assigning tangents to unknown types
            # (In this case, the function will be called by pytorch_make_dual
            # with the output of drjit.grad, which equals float(0))
            return a

        if type(a) is type(b):
            if pytorch_fp_check(a):
                from torch.autograd.forward_ad import make_dual
                return make_dual(a, b)
        else:
            return a
        return ...

    return apply2(fn, a, b)


def fixup_grad(a, b, target, /, tf_add_zero = False):
    '''
    Fix up gradients so that they are accepted by routines like ``jax.vjp``,
    ``torch.autograd.backward``, etc.

    Specifically, the function

    - ensures that tensors in PyTree `a` have the same shape as those in `b`,
      which is important to correctly handle the dimension 0 case.

    - replaces gradients for non-differentiable arrays with special objects
      that JAX expects.
    '''
    if (target == 'tf') and not tf_add_zero:
        # Nothing to do, let's save the trouble
        return a

    def fn(a, b):
        # Ignore structural PyTree elements
        if pytree_check(a):
            return ...

        is_jax = target == 'jax' and jax_check(a)
        is_torch = target == 'torch' and pytorch_check(a)
        is_tf_fp = target == 'tf' and tf_add_zero and tf_fp_check(a)

        # JAX really doesn't like receiving gradients/tangents for non-diff.
        # elements. It wants a special array with dtype `jax.float0`. Such
        # arrays cannot be created with JAX, so we actually have to revert to
        # NumPy, of all things! (https://github.com/google/jax/issues/4433)
        if target == 'jax' and ((not is_jax) or 'float' not in a.dtype.name) \
            and not isinstance(a, float):
            import jax
            import numpy
            return numpy.zeros(getattr(b, 'shape', ()), dtype=jax.float0)

        if is_tf_fp and tf_add_zero:
            import tensorflow as tf
            with tf.device(a.device):
                return tf.zeros_like(a) + a

        if type(a) is type(b):
            if is_jax or (is_torch and a.dtype.is_floating_point):
                return a.reshape(b.shape)
        else:
            return 0

        return ...

    return apply2(fn, a, b)


def _flatten(a, flat, desc, /):
    '''Helper function to flatten a PyTree and rebuild it later`'''
    tp = type(a)
    desc.append(tp)

    if tp is list or tp is tuple:
        desc.append(len(a))
        for v in a:
            _flatten(v, flat, desc)
    elif tp is dict:
        desc.append(tuple(a.keys()))
        for v in a.values():
            _flatten(v, flat, desc)
    else:
        desc = getattr(tp, 'DRJIT_STRUCT', None)
        if type(desc) is dict:
            for k in desc:
                _flatten(getattr(a, k), flat, desc)
        else:
            flat.append(a)


def _unflatten(flat, desc, /):
    tp = desc.pop()

    if tp is list or tp is tuple:
        n = desc.pop()
        return tp(_unflatten(flat, desc) for _ in range(n))
    elif tp is dict:
        keys = desc.pop()
        return { k : _unflatten(flat, desc) for k in keys }
    else:
        desc = getattr(tp, 'DRJIT_STRUCT', None)
        if type(desc) is dict:
            result = tp()
            for k in desc:
                setattr(result, k, _unflatten(flat, desc))
            return result
        else:
            return flat.pop()


def flatten(a, /):
    flat, desc = [], []
    _flatten(a, flat, desc)
    return desc, *flat


def unflatten(desc, *flat):
    return _unflatten(
        list(reversed(flat)),
        list(reversed(desc)))


def wrap_into_dr_tensor(value):
    '''Helper to transform a PyTree's members to Dr.Jit tensors'''
    def fn(h):
        tp = type(h)
        if dr.is_array_v(tp):
            if not dr.is_tensor_v(h):
                h = dr.tensor_t(tp)(h)
            return h
        return ...
    return apply(fn, value)


def wrap_into_tf_tensor(value):
    '''Helper to transform a PyTree's members to TF tensors'''
    import tensorflow as tf
    def fn(h):
        try:
            return tf.convert_to_tensor(h)
        except ValueError:
            return tf.constant(-1, tf.float32)
    return tf.nest.map_structure(fn, value)


def find_first_tf_tensor(value):
    '''Finds the first TensorFlow tensor in a PyTree'''
    if isinstance(value, (list, tuple)):
        for item in value:
            result = find_first_tf_tensor(item)
            if tf_check(result):
                return result
    elif isinstance(value, dict):
        for key in value:
            result = find_first_tf_tensor(value[key])
            if tf_check(result):
                return result
    else:
        if tf_check(value):
            return value
        return None


class WrapADOp(dr.CustomOp):
    '''
    Dr.Jit custom operation that wraps differentiable computation performed
    using another AD framework (e.g., PyTorch, TensorFlow)
    '''
    def eval(self, func, target, *args, **kwargs):
        # Convert input PyTrees from Dr.Jit
        self.args,   self.args_tp   = from_drjit(args,   target, True)
        self.kwargs, self.kwargs_tp = from_drjit(kwargs, target, True)
        self.target = target
        self.func = func

        # Evaluate the function using another array programming framework
        if target == 'tf':
            import tensorflow as tf
            self.watched_vars = wrap_into_tf_tensor([self.args, self.kwargs])
            self.device = find_first_tf_tensor([self.args, self.kwargs]).device
            with tf.device(self.device): # Ensure that TF runs on the correct device
                with tf.GradientTape() as tape:
                    tape.watch(self.watched_vars)
                    self.out = func(*self.args, **self.kwargs)
                self.tape = tape
        else:
            self.out = func(*self.args, **self.kwargs)

        # Convert the out PyTree to Dr.Jit
        return to_drjit(self.out, target)

    def forward(self):
        target = self.target

        grad_args, _   = from_drjit(self.grad_in('args'),   target)
        grad_kwargs, _ = from_drjit(self.grad_in('kwargs'), target)
        grad_args      = fixup_grad(grad_args, self.args, target)
        grad_kwargs    = fixup_grad(grad_kwargs, self.kwargs, target)
        if target == 'torch':
            import torch.autograd.forward_ad as fa

            with fa.dual_level():
                out = self.func( *pytorch_make_dual(self.args,   grad_args),
                                **pytorch_make_dual(self.kwargs, grad_kwargs))

                grad_out = pytorch_tangent(out)
        elif target == 'jax':
            import jax

            def wrapper(args, kwargs):
                return self.func(*args, **kwargs)

            _, grad_out = jax.jvp(
                wrapper, (self.args, self.kwargs), (grad_args, grad_kwargs)
            )
        elif target == 'tf':
            import tensorflow as tf
            primals = list(self.args) + list(self.kwargs.values())
            tangents = list(grad_args) + list(grad_kwargs.values())
            with tf.device(self.device):
                with tf.autodiff.ForwardAccumulator(
                    primals=tf_filter_fp(primals),
                    tangents=tf_filter_fp(tangents)
                ) as acc:
                    out = self.func(*self.args, **self.kwargs)
                grad_out = acc.jvp(wrap_into_tf_tensor(out))
        else:
            raise RuntimeError('WrapADOp.forward(): unsupported framework!')
        self.set_grad_out(to_drjit(grad_out, target))

    def backward(self):
        target = self.target
        grad_out, _ = from_drjit(self.grad_out(), target)
        # Sever link to DrJit (via DLPack `owner` field) to avoid leaks (TF only).
        grad_out    = fixup_grad(grad_out, self.out, target, tf_add_zero=True)

        if target == 'torch':
            import torch
            torch.autograd.backward(pytorch_filter_fp(self.out),
                                    pytorch_filter_fp(grad_out))
            grad_args = pytorch_grad(self.args)
            grad_kwargs = pytorch_grad(self.kwargs)
        elif target == 'jax':
            import jax

            def wrapper(args, kwargs):
                return self.func(*args, **kwargs)

            _, vjp_fun = jax.vjp(wrapper, self.args, self.kwargs)
            grad_args, grad_kwargs = vjp_fun(grad_out)
        elif target == 'tf':
            import tensorflow as tf
            with tf.device(self.device): # Ensure that TF runs on the correct device
                out = wrap_into_tf_tensor(self.out)

                grad_args, grad_kwargs = self.tape.gradient(out, self.watched_vars,
                                                            output_gradients=grad_out)
        else:
            raise RuntimeError('WrapADOp.backward(): unsupported framework!')
        self.set_grad_in('args',   to_drjit(grad_args,   target, self.args_tp))
        self.set_grad_in('kwargs', to_drjit(grad_kwargs, target, self.kwargs_tp))

torch_wrapper = None

# Temporary storage of 'desc_o' needed for torch->drjit PyTorch forward-AD
# See https://github.com/pytorch/pytorch/issues/117491
torch_desc_o = None

def new_drjit_scope():
    if dr.has_backend(dr.JitBackend.LLVM):
        dr.detail.new_scope(dr.JitBackend.LLVM)
    if dr.has_backend(dr.JitBackend.CUDA):
        dr.detail.new_scope(dr.JitBackend.CUDA)

def create_torch_wrapper():
    from torch import set_grad_enabled as torch_set_grad_enabled
    from torch.autograd import Function, function

    class TorchWrapper(Function):
        @staticmethod
        def forward(ctx, func, desc, *inputs):
            new_drjit_scope()

            # Convert and unflatten the input PyTrees
            inputs = to_drjit(inputs, 'torch', enable_grad=True)
            args, kwargs = unflatten(desc, *inputs)

            # Run the function, flatten the output PyTree and convert its members to tensors
            global torch_desc_o
            with torch_set_grad_enabled(True):
                # Torch autograd tracing is disabled within `Function.forward`
                # we turn it back on here in case func itself uses torch
                torch_desc_o, *output = flatten(func(*args, **kwargs))
            output = wrap_into_dr_tensor(output)

            # Stash inputs and outputs for later use
            ctx.inputs, ctx.output = inputs, output

            # Convert the output and return
            output_conv = from_drjit(output, 'torch')[0]

            new_drjit_scope()

            return tuple(output_conv)

        @staticmethod
        @function.once_differentiable
        def backward(ctx, *grad_outputs):
            new_drjit_scope()

            grad_outputs = to_drjit(grad_outputs, 'torch')
            dr.set_grad(ctx.output, grad_outputs)

            # Backpropagate, mask non-differentiable elements
            grad_inputs = dr.backward_to(ctx.inputs)
            grad_inputs = [
                (grad_inputs[i] if dr.grad_enabled(ctx.inputs[i]) else None) \
                    for i in range(len(ctx.inputs))
            ]

            # Convert
            grad_inputs = from_drjit(grad_inputs, 'torch')[0]

            new_drjit_scope()

            return None, None, *grad_inputs

        @staticmethod
        @function.once_differentiable
        def jvp(ctx, func, desc, *grad_inputs):
            new_drjit_scope()

            grad_inputs = to_drjit(grad_inputs, 'torch')
            dr.set_grad(ctx.inputs, grad_inputs)

            # Forward propagate, mask non-differentiable elements
            grad_output = dr.forward_to(ctx.output)
            grad_output = tuple(
                (grad_output[i] if dr.grad_enabled(ctx.output[i]) else None) \
                    for i in range(len(ctx.output))
            )

            # Convert
            grad_output = from_drjit(grad_output, 'torch')[0]

            new_drjit_scope()

            return grad_output


    return TorchWrapper

T = typing.TypeVar("T")

def wrap(source: typing.Union[str, types.ModuleType],
         target: typing.Union[str, types.ModuleType]) -> typing.Callable[[T], T]:
    r'''
    Differentiable bridge between Dr.Jit and other array programming
    frameworks.

    This function wraps computation performed using one array programming
    framework to expose it in another. Currently, `PyTorch
    <https://pytorch.org>`__, `TensorFlow <https://tensorflow.org>`__,
    and `JAX <https://jax.readthedocs.io>`__ are supported, though other
    frameworks may be added in the future.

    Annotating a function with :py:func:`@drjit.wrap <wrap>` adds code
    that suitably converts arguments and return values. Furthermore, it
    stitches the operation into the *automatic differentiation* (AD) graph of
    the other framework to ensure correct gradient propagation.

    When exposing code written using another framework, the wrapped function
    can take and return any :ref:`PyTree <pytrees>` including flat or nested
    Dr.Jit arrays, tensors, and arbitrary nested lists/tuples, dictionaries,
    and custom data structures. The arguments don't need to be
    differentiable---for example, integer/boolean arrays that don't carry
    derivative information can be passed as well.

    The wrapped function should be *pure*: in other words, it should read its
    input(s) and compute an associated output so that re-evaluating the
    function again produces the same answer. Multi-framework derivative
    tracking of impure computation will likely not behave as expected.

    The following table lists the currently supported conversions:

    .. |nbsp| unicode:: 0xA0
       :trim:

    .. list-table::
       :widths: 1 5 5 5 50
       :header-rows: 1

       * - Direction
         - Primal
         - Forward-mode |nbsp| AD
         - Reverse-mode |nbsp| AD
         - Remarks

       * - ``drjit`` → ``torch``
         - .. centered:: ✅
         - .. centered:: ✅
         - .. centered:: ✅
         - Everything just works.

       * - ``torch`` → ``drjit``
         - .. centered:: ✅
         - .. centered:: ✅
         - .. centered:: ✅

         - **Limitation**: The passed/returned :ref:`PyTrees <pytrees>` can
           contain arbitrary arrays or tensors, but other types
           (e.g., a custom Python object not understood by PyTorch) will
           raise errors when differentiating in *forward mode* (backward mode
           works fine).

           An `issue <https://github.com/pytorch/pytorch/issues/117491>`__ was
           filed on the PyTorch bugtracker.

       * - ``drjit`` → ``tf``
         - .. centered:: ✅
         - .. centered:: ✅
         - .. centered:: ✅
         - You may want to further annotate the wrapped function with
           ``tf.function`` to trace and just-in-time compile it in the
           Tensorflow environment, i.e.,

           .. code-block:: python

              @dr.wrap(source='drjit', target='tf')
              @tf.function(jit_compile=False) # Set to True for XLA mode

           **Limitation**: There is an issue for tf.int32 tensors which are
           wrongly placed on CPU by DLPack. This can lead to inconsistent device
           placement of tensors.

           An `issue <https://github.com/tensorflow/tensorflow/issues/78091>`__
           was filed on the TensorFlow bugtracker.

       * - ``tf`` → ``drjit``
         - .. centered:: ✅
         - .. centered:: ❌
         - .. centered:: ✅
         - TensorFlow has some limitiations with respect to custom gradients
           in foward-mode AD.

           **Limitation**: TensorFlow does not allow for non-tensor
           input structures in fuctions with
           `custom gradients
           <https://www.tensorflow.org/api_docs/python/tf/custom_gradient>`__.

           TensorFlow has a bug for functions with custom gradients and
           keyword arguments.

           An `issue <https://github.com/tensorflow/tensorflow/issues/77559>`__
           was filed on the TensorFlow bugtracker.

       * - ``drjit`` → ``jax``
         - .. centered:: ✅
         - .. centered:: ✅
         - .. centered:: ✅
         - You may want to further annotate the wrapped function with
           ``jax.jit`` to trace and just-in-time compile it in the JAX
           environment, i.e.,

           .. code-block:: python

              @dr.wrap(source='drjit', target='jax')
              @jax.jit

           **Limitation**: The passed/returned :ref:`PyTrees <pytrees>` can
           contain arbitrary arrays or Python scalar types, but other types
           (e.g., a custom Python object not understood by JAX) will raise
           errors.

       * - ``jax`` → ``drjit``
         - .. centered:: ❌
         - .. centered:: ❌
         - .. centered:: ❌
         - This direction is currently unsupported. We plan to add it in
           the future.

    Please also refer to the documentation sections on :ref:`multi-framework
    differentiation <interop_ad>` :ref:`associated caveats <interop_caveats>`.

    .. note::

       Types that have no equivalent on the other side (e.g. a quaternion
       array) will convert to generic tensors.

       Data exchange is limited to representations that exist on both sides.
       There are a few limitations:

       - PyTorch `lacks support for most unsigned integer types
         <https://github.com/pytorch/pytorch/issues/58734>`__ (``uint16``,
         ``uint32``, or ``uint64``-typed arrays). Use signed integer types to
         work around this issue.

       - TensorFlow has limitations with respect to forward-mode AD for
         functions with custom gradients.There is also an `issue for functions
         with keyword arguments
         <https://github.com/tensorflow/tensorflow/issues/77559>`__.

       - Dr.Jit currently lacks support for most 8- and 16-bit numeric types
         (besides half precision floats).

       - JAX `refuses to exchange
         <https://github.com/google/jax/issues/19352>`__ boolean-valued
         tensors with other frameworks.

    Args:
        source (str | module): The framework used *outside* of the wrapped
          function. The argument is currently limited to either ``'drjit'``,
          ``'torch'``, ``'tf'``, or ``jax'``. For convenience, the associated Python
          module can be specified as well.

        target (str | module): The framework used *inside* of the wrapped
          function. The argument is currently limited to either ``'drjit'``,
          ``'torch'``, ``'tf'``, or ``'jax'``. For convenience, the associated Python
          module can be specified as well.

    Returns:
        The decorated function.
    '''

    # Get module names if source and target are not already strings
    source = source.__name__ if not isinstance(source, str) else source
    target = target.__name__ if not isinstance(target, str) else target
    valid_types = ('drjit', 'torch', 'jax', 'tf')

    if source not in valid_types:
        raise ValueError("drjit.wrap(): unknown 'source' argument.")

    if target not in valid_types:
        raise ValueError("drjit.wrap(): unknown 'target' argument.")

    if source != 'drjit' and target != 'drjit':
        raise ValueError("drjit.wrap(): at least one of 'source' and "
                         "'target' must equal \"drjit\".")

    if source == target:
        # Nothing to do
        return lambda x: x

    if source == 'drjit':
        def wrapper(func):
            return lambda *args, **kwargs: \
                dr.custom(WrapADOp, func, target, *args, **kwargs)
        return wrapper
    elif target == 'drjit' and source == 'torch':
        global torch_wrapper

        if torch_wrapper is None:
            torch_wrapper = create_torch_wrapper()

        def wrapper(func):
            def wrapper_2(*args, **kwargs):
                global torch_desc_o
                rv = torch_wrapper.apply(func, *flatten((args, kwargs)))
                rv = unflatten(torch_desc_o, *rv)
                torch_desc_o = None
                return rv

            return wrapper_2

        return wrapper

    elif target == 'drjit' and source == 'tf':

        import tensorflow as tf

        def wrapper(func):

            @tf.custom_gradient
            def wrapper_2(*args, **kwargs):
                inputs = to_drjit((args, kwargs), 'tf', enable_grad=True)
                outputs = func(*inputs[0], **inputs[1])
                def grad(*dy):
                    if kwargs:
                        raise TypeError("Keyword arguments are not allowed for 'tf->drjit'")
                    grad_outputs = to_drjit(dy, 'tf')
                    out = flatten(outputs)[1:]
                    out = wrap_into_dr_tensor(out)
                    dr.set_grad(out, grad_outputs)
                    vars = flatten(inputs[0])[1:] # Only gradients for args are computed due
                                                  # to a TF bug https://github.com/tensorflow/tensorflow/issues/77559
                    grads = dr.backward_to(vars)
                    grads = from_drjit(grads, 'tf')[0]
                    # Set gradients for non-differentiable tensors to None
                    grads = [(g if dr.grad_enabled(vars[i]) else None) \
                             for i, g in enumerate(flatten(grads)[1:])]
                    return grads
                return from_drjit(outputs, 'tf')[0], grad

            return wrapper_2

        return wrapper
    else:
        raise ValueError("drjit.wrap(): unsupported combination of 'source' and 'target'.")
