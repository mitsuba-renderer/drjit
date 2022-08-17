import drjit as _dr
import numpy as _np
import torch as _torch


def to_torch(arg):
    '''
    Convert a differentiable Dr.Jit array into a PyTorch tensor while keeping
    track of derivative computation.

    Using this function is it possible to mix AD-aware computation in Dr.Jit
    and PyTorch. As shown in the code example below, a differentiable array
    (tracking derivatives) resulting from some Dr.Jit arithmetic can be converted
    into a PyTorch tensor to perform further computation. A subsequent call to
    `torch.tensor.backward() <https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html>`_
    will properly backpropagate gradients across the frameworks, pushing
    gradients all the way to the original Dr.Jit array.

    .. code-block:: python

        # Start with a Dr.Jit array
        a = dr.llvm.ad.Float(...)
        dr.enable_grad(a)

        # Some Dr.Jit arithmetic
        b = dr.sin(a)

        # Convert it into a PyTorch tensor
        c = dr.to_torch(b)

        # Some PyTorch arithmetic
        d = c.sum()

        # Propagate gradients to variable a
        d.backward()

        # Inspect the resulting gradients
        print(dr.grad(a))

    This function relies on the `torch.autograd.Function <https://pytorch.org/docs/stable/notes/extending.html>`_
    class to implement the conversion as a custom differentiable operation in PyTorch.

    .. danger::

        Forward-mode AD isn't currently supported by this operation.

    Args:
        arg (drjit.ArrayBase): differentiable Dr.Jit array or tensor type.

    Returns:
        torch.tensor: The PyTorch tensor representing the input Dr.Jit array.
    '''

    class ToTorch(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, arg, handle):
            ctx.drjit_arg = arg
            return _torch.tensor(_np.array(arg))

        @staticmethod
        @_torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            _dr.set_grad(ctx.drjit_arg, grad_output)
            _dr.enqueue(_dr.ADMode.Backward, ctx.drjit_arg)
            _dr.traverse(type(ctx.drjit_arg), _dr.ADMode.Backward)
            del ctx.drjit_arg
            return None, None

    handle = _torch.empty(0, requires_grad=True)
    return ToTorch.apply(arg, handle)


def from_torch(dtype, arg):
    '''
    Convert a differentiable PyTorch tensor into a Dr.Jit array while keeping
    track of derivative computation.

    Using this function is it possible to mix AD-aware computation in Dr.Jit
    and PyTorch. As shown in the code example below, an differentiable tensor
    (tracking derivatives) resulting from some PyTorch arithmetic can be converted
    into a Dr.Jit array to perform further computation. A subsequent call to
    :py:func:`drjit.backward()` will properly backpropagate gradients across the
    frameworks, pushing gradients all the way to the original PyTorch tensor.

    .. code-block:: python

        # Start with a PyTorch tensor
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Some PyTorch arithmetic
        b = torch.sin(a)

        # Convert it into a Dr.Jit array
        c = dr.from_torch(m.Float, b)

        # Some Dr.Jit arithmetic
        e = dr.sum(d)

        # Propagate gradients to variable a
        dr.backward(e)

        # Inspect the resulting gradients
        print(a.grad)

    This function relies on the :py:class:`CustomOp` class to implement the
    conversion as a custom differentiable operation in Dr.Jit.

    .. danger::

        Forward-mode AD isn't currently supported by this operation.

    Args:
        dtype (type): Desired differentiable Dr.Jit array type.
        arg (torch.tensor): PyTorch tensor type

    Returns:
        drjit.ArrayBase: The differentiable Dr.Jit array representing the input
                         PyTorch tensor.
    '''
    if not _dr.is_diff_v(dtype) or not _dr.is_array_v(dtype):
        raise TypeError("from_torch(): expected a differentiable Dr.Jit array type!")

    class FromTorch(_dr.CustomOp):
        def eval(self, arg, handle):
            self.torch_arg = arg
            return dtype(arg)

        def forward(self):
            raise TypeError("from_torch(): forward-mode AD is not supported!")

        def backward(self):
            grad = _torch.tensor(_np.array(self.grad_out()))
            self.torch_arg.backward(grad)

    handle = _dr.zeros(dtype)
    _dr.enable_grad(handle)
    return _dr.custom(FromTorch, arg, handle)
