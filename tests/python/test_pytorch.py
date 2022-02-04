# import drjit as dr
# import numpy as np
# import pytest
# import torch

# class DrJitAtan2(torch.autograd.Function):
#     """PyTorch function example from the documentation."""
#     @staticmethod
#     def forward(ctx, arg1, arg2):
#         ctx.in1 = dr.FloatD(arg1)
#         ctx.in2 = dr.FloatD(arg2)
#         dr.set_requires_gradient(ctx.in1, arg1.requires_grad)
#         dr.set_requires_gradient(ctx.in2, arg2.requires_grad)
#         ctx.out = dr.atan2(ctx.in1, ctx.in2)
#         out_torch = ctx.out.torch()
#         dr.cuda_flush_malloc_cache()
#         return out_torch

#     @staticmethod
#     def backward(ctx, grad_out):
#         dr.set_gradient(ctx.out, dr.FloatC(grad_out))
#         dr.FloatD.backward()
#         result = (dr.gradient(ctx.in1).torch()
#                   if dr.requires_gradient(ctx.in1) else None,
#                   dr.gradient(ctx.in2).torch()
#                   if dr.requires_gradient(ctx.in2) else None)
#         del ctx.out, ctx.in1, ctx.in2
#         dr.cuda_flush_malloc_cache()
#         return result


# def test01_set_gradient():
#     a = dr.FloatD(42, 10)
#     dr.set_requires_gradient(a)

#     with pytest.raises(TypeError):
#         grad = dr.FloatD(-1, 10)
#         dr.set_gradient(a, grad)

#     grad = dr.FloatC(-1, 10)
#     dr.set_gradient(a, grad)
#     assert np.allclose(grad.numpy(), dr.gradient(a).numpy())

#     # Note: if `backward` is not called here, test03 segfaults later.
#     # TODO: we should not need this, there's most likely some missing cleanup when `a` is destructed
#     dr.FloatD.backward()
#     del a, grad


# def test02_array_to_torch():
#     a = dr.FloatD(42, 10)
#     a_torch = a.torch()
#     assert isinstance(a_torch, torch.Tensor)
#     a_torch += 8
#     a_np = a_torch.cpu().numpy()
#     assert isinstance(a_np, np.ndarray)
#     assert np.allclose(a_np, 50)


# def test03_pytorch_function():
#     drjit_atan2 = DrJitAtan2.apply

#     y = torch.tensor(1.0, device='cuda')
#     x = torch.tensor(2.0, device='cuda')
#     y.requires_grad_()
#     x.requires_grad_()

#     o = drjit_atan2(y, x)
#     o.backward()
#     assert np.allclose(y.grad.cpu(), 0.4)
#     assert np.allclose(x.grad.cpu(), -0.2)
