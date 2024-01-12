import drjit as dr
import typing
import types

def pytorch_check(value, /):
    '''Returns ``True`` if ``value`` is a PyTorch tensor'''
    return type(value).__module__ == 'torch' and type(value).__name__ == 'Tensor'


def apply(fn, a, /):
    '''Helper function to recursively map a PyTree through the function ``fn``'''
    tp = type(a)

    result = fn(a)
    if result is not None:
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


def apply_pair(fn, a, b, /):
    '''
    Helper function to recursively map two compatible PyTrees
    through the function ``fn``
    '''
    ta, tb = type(a), type(b)

    if ta is not tb:
        if tb is float and b == 0.0:
            # Allow incompatible types when assigning tangents to unknown types
            # (In this case, the function will be called by pytorch_make_dual
            # with the output of drjit.grad, which equals float(0))
            return a
        raise TypeError(f'Incompatible types: {ta} and {tb}.')

    result = fn(a, b)
    if result is not None:
        return result
    elif ta is list:
        assert len(a) == len(b)
        return [apply_pair(fn, a[i], b[i]) for i in range(len(a))]
    elif ta is tuple:
        assert len(a) == len(b)
        return tuple(apply_pair(fn, a[i], b[i]) for i in range(len(a)))
    elif ta is dict:
        assert a.keys() == b.keys()
        return {k: apply_pair(fn, a[k], b[k]) for k, v in a.items()}
    else:
        desc = getattr(ta, 'DRJIT_STRUCT', None)
        if type(desc) is dict:
            result = type(a)()
            for k in desc:
                setattr(result, k, apply_pair(fn, getattr(a, k), getattr(b, k)))
            return result
        else:
            return a


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

def pytorch_tangent(value, /):
    '''Extract a the tangents of PyTorch arrays from the PyTree ``value``'''

    def fn(h, /):
        if pytorch_check(h):
            from torch.autograd.forward_ad import unpack_dual
            return unpack_dual(h).tangent
        return None

    return apply(fn, value)

def pytorch_make_dual(a, b, /):
    '''Build combined primal/tangent PyTrees for PyTorch forward-mode AD'''

    def fn(a, b):
        if pytorch_check(a) and a.dtype.is_floating_point:
            from torch.autograd.forward_ad import make_dual
            return make_dual(a, b)
        return None

    return apply_pair(fn, a, b)

def pytorch_reshape(a, b, /):
    '''Ensure that tensors in PyTree `a` have the same shape as those in `b`'''

    def fn(a, b):
        if pytorch_check(a) and a.dtype.is_floating_point:
            return a.reshape(b)
        return None

    return apply_pair(fn, a, b)


class WrapADOp(dr.CustomOp):
    '''
    Dr.Jit custom operation that wraps differentiable computation performed
    using another AD framework (e.g., PyTorch)
    '''
    def eval(self, f, target, *args, **kwargs):
        # Convert input PyTrees from Dr.Jit
        self.args, self.args_tp = from_drjit(args, target, True)
        self.kwargs, self.kwargs_tp = from_drjit(kwargs, target, True)
        self.target = target
        self.f = f

        # Evaluate the function using another array programming framework
        self.out = f(*self.args, **self.kwargs)

        # Convert the out PyTree to Dr.Jit
        return to_drjit(self.out, target, None)

    def forward(self):
        target = self.target

        grad_args, _   = from_drjit(self.grad_in('args'), target, False)
        grad_kwargs, _ = from_drjit(self.grad_in('kwargs'), target, False)

        if target == 'torch':
            import torch.autograd.forward_ad as fa

            with fa.dual_level():
                out = self.f( *pytorch_make_dual(self.args,   grad_args),
                             **pytorch_make_dual(self.kwargs, grad_kwargs))

                grad_out = pytorch_tangent(out)

        self.set_grad_out(to_drjit(grad_out, target, None))

    def backward(self):
        target = self.target

        grad_out, _ = from_drjit(self.grad_out(), target, False)

        if target == 'torch':
            import torch
            torch.autograd.backward(pytorch_flatten(self.out),
                                    pytorch_flatten(grad_out))

            grad_args = pytorch_grad(self.args)
            grad_kwargs = pytorch_grad(self.kwargs)
        else:
            raise RuntimeError('WrapADOp.backward(): unsupported framework!')

        self.set_grad_in('args', to_drjit(grad_args, target, self.args_tp))
        self.set_grad_in('kwargs', to_drjit(grad_kwargs, target, self.kwargs_tp))


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
    can take and return any :ref:`Pytree <pytrees>` including flat or nested
    Dr.Jit arrays, tensors, and arbitrary nested lists/tuples, dictionaries,
    and custom data structures.

    The wrapped function should be *pure*: in other words, it should read its
    input(s) and compute an associated output so that re-evaluating the
    function again still produces the same answer. Multi-framework derivative
    tracking of impure computation may not behave as expected.

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
            return lambda *args, **kwargs: dr.custom(WrapADOp, f, target, *args, **kwargs)
        return wrapper

    return None
