from argparse import ArgumentError
import drjit as _dr

import inspect
from collections.abc import Mapping as _Mapping, \
                            Sequence as _Sequence


def warp_ad(func, to: str, arg):
    '''

    '''

    # Detect input framework
    def detect(o):
        if isinstance(o, _Sequence):
            return detect(o[0])
        elif isinstance(o, _Mapping):
            return detect(o.items()[0][1])
        elif _dr.is_array_v(o):
            return 'drjit'
        elif o.__module__ == 'torch' and type(o).__name__ == 'Tensor':
            return 'torch'
        else:
            return None

    def cast_to_generic(cast):
        def func(a):
            if _dr.is_array_v(a):
                return cast(a)
            elif isinstance(a, _Sequence):
                return [func(b) for b in a]
            elif isinstance(a, _Mapping):
                return {k: func(v) for k, v in a.items()}
        return func

    def cast_from_generic(cast):
        def func(a, dtype):
            if _dr.is_array_v(dtype):
                return cast(a, dtype)
            elif isinstance(dtype, _Sequence):
                return [func(a[i], dtype[i]) for i in range(len(dtype))]
            elif isinstance(dtype, _Mapping):
                return {k: func(a[k], dtype[k]) for k in dtype.keys()}
        return func

    def cast_to_torch(a):
        def cast(a):
            b = _torch.tensor(_np.array(a))
            return b if _dr.is_llvm_v(a) else b.cuda()
        return cast_to_generic(cast)(a)

    def cast_from_torch(a, dtype):
        def cast(a, dtype):
            if _dr.is_llvm_v(dtype):
                return dtype(a.cpu())
            elif _dr.is_cuda_v(dtype):
                return dtype(a.cuda())
        return cast_from_generic(cast)(a, dtype)

    if detect(arg) == 'torch':
        import numpy as _np
        import torch as _torch

        if not to == 'drjit':
            raise ArgumentError('wrap_ad(): invalid combination of frameworks, '
                                'expected to=drjit!')

        class ToDrJit(_torch.autograd.Function):
            @staticmethod
            def forward(ctx, arg):
                ctx.arg = arg
                # Get list of func arguments types from its signature
                sig = inspect.signature(func)
                arg_type = [sig.parameters[k].annotation for k in sig.parameters][0]

                ctx.arg_drjit = cast_from_torch(arg, arg_type)
                _dr.enable_grad(ctx.arg_drjit)

                ctx.res_drjit = func(ctx.arg_drjit)

                res_torch = cast_to_torch(ctx.res_drjit)

                return res_torch

            @staticmethod
            @_torch.autograd.function.once_differentiable
            def backward(ctx, grad_output):
                _dr.set_grad(ctx.res_drjit, grad_output)
                _dr.enqueue(_dr.ADMode.Backward, ctx.res_drjit)
                _dr.traverse(type(ctx.res_drjit), _dr.ADMode.Backward)

                arg_grad = cast_to_torch(_dr.grad(ctx.arg_drjit))

                del ctx.res_drjit, ctx.arg_drjit
                return arg_grad

        return ToDrJit.apply(arg)

    if to == 'torch':
        import numpy as _np
        import torch as _torch

        if not detect(arg) == 'drjit':
            raise ArgumentError('wrap_ad(): invalid combination of frameworks, '
                                'expected args to be Dr.Jit array types!')

        class ToTorch(_dr.CustomOp):
            def eval(self, arg):
                self.arg = arg
                self.arg_torch = _torch.tensor(_np.array(arg))

                if _dr.is_cuda_v(arg):
                    self.arg_torch = self.arg_torch.cuda()

                self.arg_torch.requires_grad = True

                self.res_torch = func(self.arg_torch)

                res_dtype = inspect.signature(func).return_annotation

                return cast_from_torch(self.res_torch, res_dtype)

            def forward(self):
                raise TypeError("warp_ad(): forward-mode AD is not supported!")

            def backward(self):
                grad_out_torch = cast_to_torch(self.grad_out())

                _torch.autograd.backward(self.res_torch, grad_out_torch)

                arg_grad = cast_from_torch(self.arg_torch.grad, type(self.arg))

                self.set_grad_in('arg', arg_grad)

        return _dr.custom(ToTorch, arg)








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
    import numpy as _np
    import torch as _torch

    if not _dr.is_diff_v(arg) or not _dr.is_array_v(arg):
        raise TypeError("from_torch(): expected a differentiable Dr.Jit array type!")

    class ToTorch(_torch.autograd.Function):
        @staticmethod
        def forward(ctx, arg, handle):
            print('to_torch(): forward')
            print(f'Thread ID: {threading.get_native_id()}')
            ctx.drjit_arg = arg
            ctx.thread = threading.current_thread()
            if _dr.is_llvm_v(arg):
                return _torch.tensor(_np.array(arg))
            elif _dr.is_cuda_v(arg):
                return _torch.tensor(_np.array(arg)).cuda()
            else:
                raise TypeError("to_torch(): expected an LLVM or CUDA Dr.Jit array type!")

        @staticmethod
        @_torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            print(dir(ctx.thread))
            print('to_torch(): backward -> set_grad')
            print(f'VCallRecord: {_dr.flag(_dr.JitFlag.VCallRecord)}')
            _dr.set_grad(ctx.drjit_arg, grad_output)
            print(f'VCallRecord: {_dr.flag(_dr.JitFlag.VCallRecord)}')
            _dr.enqueue(_dr.ADMode.Backward, ctx.drjit_arg)
            print(f'VCallRecord: {_dr.flag(_dr.JitFlag.VCallRecord)}')
            print(_dr)
            print(f'Thread ID: {threading.get_native_id()}')
            _dr.traverse(type(ctx.drjit_arg), _dr.ADMode.Backward)
            del ctx.drjit_arg
            print('to_torch(): backward -> DONE')
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
    import numpy as _np
    import torch as _torch

    if not _dr.is_diff_v(dtype) or not _dr.is_array_v(dtype):
        raise TypeError("from_torch(): expected a differentiable Dr.Jit array type!")

    class FromTorch(_dr.CustomOp):
        def eval(self, arg, handle):
            print('from_torch(): eval')
            self.torch_arg = arg
            if _dr.is_llvm_v(dtype):
                return dtype(arg.cpu())
            elif _dr.is_cuda_v(dtype):
                return dtype(arg.cuda())
            else:
                raise TypeError("from_torch(): expected an LLVM or CUDA Dr.Jit array type!")

        def forward(self):
            raise TypeError("from_torch(): forward-mode AD is not supported!")

        def backward(self):
            print('from_torch(): backward')
            grad_out = self.grad_out()
            grad = _torch.tensor(_np.array(grad_out))
            if self.torch_arg.is_cuda:
                grad = grad.cuda()
            self.torch_arg.backward(grad)

    handle = _dr.zeros(dtype)
    _dr.enable_grad(handle)
    return _dr.custom(FromTorch, arg, handle)
