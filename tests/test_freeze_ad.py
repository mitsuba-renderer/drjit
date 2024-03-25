# Copyright (c) 2024 NVIDIA CORPORATION.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import drjit as dr
import pytest

from test_freeze import FLOAT_TYPES, ARRAY_TYPES, ARRAY_TYPES_AD


def test01_simple_ad_fully_inside():
    Float = dr.cuda.ad.Float32
    log_level = dr.log_level()
    # dr.set_log_level(dr.LogLevel.Debug)

    def my_kernel(x):
        dr.enable_grad(x)

        result = x * x
        dr.backward(result)

        return result

    for start_enabled in (True, False):
        # Re-freeze
        my_kernel_frozen = dr.kernel(my_kernel)

        for i in range(3):
            print(f'------------------------------ {i=}, {start_enabled=}')
            x = Float([1., 2., 3.]) + dr.opaque(Float, i)
            if start_enabled:
                dr.enable_grad(x)

            y = my_kernel_frozen(x)
            grad_x = dr.grad(x)
            grad_y = dr.grad(y)
            dr.schedule(y, grad_x, grad_y)
            print(f'Input was: {x=}')
            print(f'Outputs were: {y=}, {grad_y=}, {grad_x=}')
            assert dr.allclose(y, dr.sqr(x))
            assert dr.allclose(grad_y, 0)
            assert dr.allclose(grad_x, 2 * x)

            # Status of grad_enabled should be restored (side-effect of the function),
            #  even if it wasn't enabled at first
            assert dr.grad_enabled(x)
            print(f'------------------------------')

    dr.set_log_level(log_level)


@pytest.mark.parametrize('set_some_literal_grad', (False, True))
@pytest.mark.parametrize('inputs_end_enabled', (False, True))
@pytest.mark.parametrize('inputs_start_enabled', (False, True))
@pytest.mark.parametrize('params_end_enabled', (False, True))
@pytest.mark.parametrize('params_start_enabled', (False, True))
def test02_suspend_resume(
    params_start_enabled, params_end_enabled, inputs_start_enabled, inputs_end_enabled, set_some_literal_grad
):

    Float = dr.cuda.ad.Float32
    UInt32 = dr.cuda.ad.UInt32
    log_level = dr.log_level()
    # dr.set_log_level(dr.LogLevel.Debug)

    # TODO: remove this
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)

    class MyModel():
        def __init__(self, params):
            self.params = params
            self.frozen_eval = dr.kernel(type(self).eval, state=(lambda self, **_: (self.params,)))

        def eval(self, x: Float, params_end_enabled: bool, inputs_end_enabled: bool, set_some_literal_grad: bool):
            idx = dr.arange(UInt32, dr.width(x)) % dr.width(self.params)
            latents = dr.gather(Float, self.params, idx)
            result = x * latents

            with dr.resume_grad():
                dr.set_grad_enabled(self.params, params_end_enabled)
                dr.set_grad_enabled(x, inputs_end_enabled)
                if set_some_literal_grad:
                    # If grads are not enabled, this will get ignored, which is fine
                    dr.set_grad(x, Float(6.66))

            return result

    model = MyModel(params=Float([1, 2, 3, 4, 5]))

    for i in range(3):
        print(f'------------------------------ {i=}')
        # Inputs of different widths
        x = Float([0.1, 0.2, 0.3, 0.4, 0.5] * (i + 1)) + dr.opaque(Float, i)

        dr.set_grad_enabled(model.params, params_start_enabled)
        dr.set_grad_enabled(x, inputs_start_enabled)

        # print(f'Before: {model.params.index=}, {dr.grad(model.params).index=}, {dr.grad(model.params).is_literal_()}')

        with dr.suspend_grad():
            result = model.frozen_eval(model, x, params_end_enabled, inputs_end_enabled, set_some_literal_grad)

        # dr.eval(result, model.params, dr.grad(model.params))
        # print(f'After: {model.params.index=}, {dr.grad(model.params).index=}, {dr.grad(model.params).is_literal_()}')
        assert not dr.grad_enabled(result)
        assert dr.grad_enabled(model.params) == params_end_enabled
        assert dr.grad_enabled(x) == inputs_end_enabled

        # The frozen function should restore the right width, even for a zero-valued literal.
        # When grads are enabled, the default gradients are a zero-valued literal array
        # with a width equal to the array's width. Otherwise, it has width 1.
        grads = dr.grad(model.params)
        assert dr.width(grads) == (dr.width(model.params) if params_end_enabled else 1)
        assert dr.all(dr.eq(grads, 0))

        grads = dr.grad(x)
        assert dr.width(grads) == (dr.width(x) if inputs_end_enabled else 1)
        if inputs_end_enabled and set_some_literal_grad:
            assert dr.all(dr.eq(grads, 6.66))
        else:
            assert dr.all(dr.eq(grads, 0))

    dr.set_log_level(log_level)


@pytest.mark.parametrize('change_params_width', (False, True))
def test03_with_grad_scatter(change_params_width):
    Float = dr.cuda.ad.Float32
    UInt32 = dr.cuda.ad.UInt32
    log_level = dr.log_level()
    # dr.set_log_level(dr.LogLevel.Debug)
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)

    class Model:
        def __init__(self, n):
            self.params = Float(list(range(1, n+1)))
            assert dr.width(self.params) == n
            dr.enable_grad(self.params)

        def __call__(self):
            # Cheeky workaround for the frozen kernel signature checking
            pass

    def my_kernel(model, x, opaque_params_width):
        idx = dr.arange(UInt32, dr.width(x)) % opaque_params_width

        with dr.resume_grad():
            latents = dr.gather(Float, model.params, idx)
            contrib = x * latents
            dr.backward_from(contrib)

        return dr.detach(contrib)

    model = Model(5)
    my_kernel_frozen = dr.kernel(my_kernel, state=(lambda model, **_: (model.params,)))

    for i in range(6):
        print(f'------------------------------ {i=}')
        # Different width at each iteration
        x = Float([1., 2., 3.] * (i + 1)) + dr.opaque(Float, i)

        # The frozen kernel should also support the params (and therefore its gradient buffer)
        # changing width without issues.
        if change_params_width and (i == 3):
            model = Model(10)
        # Reset gradients
        dr.set_grad(model.params, 0)

        with dr.suspend_grad():
            y = my_kernel_frozen(model, x, dr.opaque(UInt32, dr.width(model.params)))
        assert not dr.grad_enabled(x)
        assert not dr.grad_enabled(y)
        assert dr.grad_enabled(model.params)

        grad_x = dr.grad(x)
        grad_y = dr.grad(y)
        grad_p = dr.grad(model.params)
        print(f'Input was: {x=}')
        print(f'Outputs were: {y=}, {grad_y=}, {grad_x=}, {grad_p=}')
        # assert dr.allclose(y, dr.sqr(x))

        # Expected grads
        assert dr.allclose(grad_y, 0)
        assert dr.allclose(grad_x, 0)
        grad_p_expected = dr.zeros(Float, dr.width(model.params))
        idx = dr.arange(UInt32, dr.width(x)) % dr.width(model.params)
        dr.scatter_reduce(dr.ReduceOp.Add, grad_p_expected, x, idx)
        assert dr.allclose(grad_p, grad_p_expected)
        print(f'------------------------------')

    dr.set_log_level(log_level)

def test04_tutorial_example():
    Float = dr.cuda.ad.Float32
    UInt32 = dr.cuda.ad.UInt32

    @dr.kernel()
    def frozen_eval(inputs, idx, params, target_value, grad_factor):
        intermediate = dr.gather(Float, params, idx)
        result = 0.5 * dr.sqr(intermediate) * inputs

        # Since reductions are not supported yet, we cannot compute a single
        # loss value here. It's not really a problem though, since DrJit can
        # backpropagate starting from arrays of any widths.
        loss_per_entry = dr.sqr(result - target_value) * grad_factor

        # The gradients resulting from backpropagation will be directly accumulated
        # (via dr.scatter_add()) into the gradient buffer of `params` (= `dr.grad(params)`).
        dr.backward_from(loss_per_entry)

        # It's fine to return the primal values of `result`, but keep in mind that they will
        # not be differentiable w.r.t. `params`.
        return dr.detach(result)

    params = Float([1, 2, 3, 4, 5])

    for _ in range(3):
        dr.disable_grad(params)
        dr.enable_grad(params)
        assert dr.all(dr.eq(dr.grad(params), 0))

        inputs = Float([0.1, 0.2, 0.3])
        idx = UInt32([1, 2, 3])
        # Represents the optimizer's loss scale
        grad_factor = 4096 / dr.opaque(Float, dr.width(inputs))

        result = frozen_eval(inputs, idx, params,
                             target_value=0.5, grad_factor=grad_factor)
        assert not dr.grad_enabled(result)
        # Gradients were correctly accumulated to `params`'s gradients.
        assert not dr.all(dr.eq(dr.grad(params), 0))



# TODO: test that we don't destroy gradients that were already present in the inputs
# TODO: test gradients that get created w.r.t. input literals, etc
# TODO: test gradient buffers getting upgraded from a literal (0) to a full buffer
# TODO: test gradients being created by a first kernel launch, and consummed / used by a subsequent launch in the same frozen function
# TODO: test grad_enabled being temporarily set inside, gradients used by a subsequent launch (still inside the frozen function), and then disabled before some other lauch and / or before we return from the frozen function
# TODO: test with input variables that are also part of the output
# TODO: test with a new input grad that happens to be a new literal value (e.g. 5)
# TODO: test with resume / suspend scopes
# TODO: test `set_grad` inside of a frozen function (including setting a 0-valued literal gradient), with and without suspend scopes

if __name__ == '__main__':
    test01_simple_ad_fully_inside()
