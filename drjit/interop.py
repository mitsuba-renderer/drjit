import drjit as dr
import typing
import types

def pytorch_check(value, /):
    '''Returns ``True`` if ``value`` is a PyTorch tensor'''
    return type(value).__module__ == 'torch' and type(value).__name__ == 'Tensor'


def apply(fn, value, /):
    '''Helper function to recursively map a PyTree through the function ``fn``'''
    tp = type(value)

    result = fn(value)
    if result is not None:
        return result
    elif tp is list:
        return [apply(fn, v) for v in value]
    elif tp is tuple:
        return tuple(apply(fn, v) for v in value)
    elif tp is dict:
        return {k: apply(fn, v) for k, v in value.items()}
    else:
        desc = getattr(tp, "DRJIT_STRUCT", None)
        if type(desc) is dict:
            result = tp()
            for k in desc: 
                setattr(result, k, apply(fn, getattr(value, k)))
            return result
        else:
            return value


def from_drjit(value, target, enable_grad, /):
    '''
    Convert a PyTree containing Dr.Jit arrays/tensors to another array
    programming framework as identified by ``target``.

    The function return sthe output PyTree as well as a sequence capturing the
    original type of each converted Dr.Jit type. This is useful when those same
    exact types should be restored in a subsequent conversion by ``to_drjit``.
    '''

    value_tp = []

    def fn(h, /):
        if dr.is_array_v(h):
            value_tp.append(type(h))
            if not dr.is_tensor_v(h):
                h = dr.tensor_t(h)(h)
            r = getattr(h, target)()
            if enable_grad and target == 'torch' and r.dtype.is_floating_point:
                r.requires_grad = True
            return r
        return None

    return apply(fn, value), value_tp


def to_drjit(value, source, value_tp, /):
    '''
    Convert a PyTree containing tensors from another array programming
    framework identified by ``source`` into Dr.Jit tensors.

    Optionally, the function can restore the array types within an input PyTree
    previously captured by ``from_drjit``.
    '''

    tp_it = iter(value_tp) if value_tp is not None else None

    def fn(h, /):
        if source == 'torch' and pytorch_check(h):
            r = dr.detail.import_tensor(h, True)
            if tp_it:
                r = next(tp_it)(r)
            return r

        return None

    return apply(fn, value)


def pytorch_flatten(value, /):
    '''Extract a flat list of PyTorch arrays from the PyTree ``value``'''

    result = []

    def fn(h, /):
        if pytorch_check(h):
            result.append(h)
        return None

    apply(fn, value)
    return result

def pytorch_grad(value, /):
    '''Extract a the gradients of PyTorch arrays from the PyTree ``value``'''

    def fn(h, /):
        if pytorch_check(h):
            return h.grad
        return None

    return apply(fn, value)

def pytorch_reshape(a, b, /):
    '''Ensure that tensors in PyTree `a` have the same shape as those in `b`'''
    tp = type(a)
    assert tp is type(b)

    if pytorch_check(a):
        if a.dtype.is_floating_point:
            return a.reshape(b)
        else:
            return a
    elif tp is list:
        assert len(a) == len(b)
        return [pytorch_reshape(a[i], b[i]) for i in range(len(a))]
    elif tp is tuple:
        assert len(a) == len(b)
        return tuple(pytorch_reshape(a[i], b[i]) for i in range(len(a)))
    elif tp is dict:
        assert a.keys() == b.keys()
        return {k: pytorch_reshape(a[k], b[k]) for k, v in a}
    else:
        desc = getattr(tp, "DRJIT_STRUCT", None)
        if type(desc) is dict:
            result = type(a)()
            for k in desc: 
                setattr(result, k, pytorch_reshape(getattr(a, k), getattr(b, k)))
            return result
        else:
            return a


def wrap_ad(source: typing.Union[str, types.ModuleType],
            target: typing.Union[str, types.ModuleType]):
    r'''
    Differentiable bridge between Dr.Jit and other array programming
    frameworks.

    This function decorator enables programs that combine Dr.Jit code with
    other array programming frameworks. Currently, only PyTorch is supported,
    though this set may be extended in the future.

    Annotating a function with :py:func:`@drjit.wrap_ad <wrap_ad>` adds code
    that suitably converts arguments and return values. Furthermore, it
    stitches the operation into the *automatic differentiation* (AD) graph of
    the other framework to ensure correct gradient propagation, which motivates
    the name of the decorator.

    When exposing code written using another framework, the wrapped function
    can take and return any PyTree including flat or nested Dr.Jit arrays and
    tensors.

    Args:
        source (str | module): The framework used *outside* of the wrapped
          function. The argument is currently limited to either ``'drjit'`` or
          ``'torch'``. For convenience, the associated Python module can
          be specified as well.

        target (str | module): The framework used *inside* of the wrapped
          function. The argument is currently limited to either ``'drjit'`` or
          ``'torch'``. For convenience, the associated Python module can
          be specified as well.

    Returns:
        The decorated function.
    '''

    # Get module names if source and target are not already strings
    source = source.__name__ if not isinstance(source, str) else source
    target = target.__name__ if not isinstance(target, str) else target
    valid_types = ('drjit', 'torch')

    if source not in valid_types:
        raise Exception("drjit.wrap_ad(): unknown 'source' argument.")

    if target not in valid_types:
        raise Exception("drjit.wrap_ad(): unknown 'target' argument.")

    if source == target:
        # Nothing to do
        return lambda x: x

    if source == 'drjit':
        def wrapper(f):
            class WrapADOp(dr.CustomOp):
                def eval(self, *args, **kwargs):
                    # Convert input PyTrees from Dr.Jit
                    self.args, self.args_tp = from_drjit(args, target, True)
                    self.kwargs, self.kwargs_tp = from_drjit(kwargs, target, True)

                    # Evaluate the function using another array programming framework
                    self.out = f(*self.args, **self.kwargs)

                    # Convert the out PyTree to Dr.Jit
                    return to_drjit(self.out, target, None)

                def forward(self):
                    raise RuntimeError('drjit.wrap_ad(): The PyTorch interface currently '
                                       'lacks support for forward mode differentiation.')

                def backward(self):
                    grad_out = from_drjit(self.grad_out(), target, False)

                    if target == 'torch':
                        import torch
                        torch.autograd.backward(pytorch_flatten(self.out),
                                                pytorch_flatten(grad_out))

                        grad_args = pytorch_grad(self.args)
                        grad_kwargs = pytorch_grad(self.kwargs)
                    else:
                        raise RuntimeError("WrapADOp.backward(): unsupported framework!")

                    grad_args   = to_drjit(grad_args, target, self.args_tp)
                    grad_kwargs = to_drjit(grad_kwargs, target, self.kwargs_tp)

                    self.set_grad_in('args', grad_args)
                    self.set_grad_in('kwargs', grad_kwargs)

            return lambda *args, **kwargs: dr.custom(WrapADOp, *args, **kwargs)

        return wrapper

    return None
